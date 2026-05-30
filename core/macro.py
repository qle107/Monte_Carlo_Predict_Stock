"""Macroeconomic indicator fetcher."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Macro data is released monthly / quarterly - 4-hour TTL is generous but prevents
# hammering free public APIs on every poll loop pass.

_CACHE_TTL = 4 * 3600.0  # seconds
_cache_lock = threading.RLock()
_macro_cache: dict = {}  # key → (data, expire_monotonic)

def _cache_get(key: str):
    with _cache_lock:
        entry = _macro_cache.get(key)
        if entry is None:
            return None
        data, exp = entry
        if time.monotonic() > exp:
            del _macro_cache[key]
            return None
        return data

def _cache_put(key: str, data) -> None:
    with _cache_lock:
        now = time.monotonic()
        stale = [k for k, (_, exp) in _macro_cache.items() if now > exp]
        for k in stale:
            del _macro_cache[k]
        _macro_cache[key] = (data, now + _CACHE_TTL)

# Each function maps (current, previous) → "bullish" | "bearish" | "neutral".
# Thresholds are based on widely-used rule-of-thumb market heuristics.

def _cpi_impact(cur: float | None, _prev) -> str:
    """High CPI → Fed hikes → equity multiples compress → bearish."""
    if cur is None:
        return "neutral"
    if cur > 3.5:
        return "bearish"
    if cur < 2.0:
        return "bullish"
    return "neutral"

def _ppi_impact(cur: float | None, _prev) -> str:
    """High PPI → upstream cost pressure → margin squeeze → bearish."""
    if cur is None:
        return "neutral"
    if cur > 3.0:
        return "bearish"
    if cur < 1.5:
        return "bullish"
    return "neutral"

def _pce_impact(cur: float | None, _prev) -> str:
    """Fed's 2 % target - above target → rate hikes → bearish."""
    if cur is None:
        return "neutral"
    if cur > 2.5:
        return "bearish"
    if cur < 1.8:
        return "bullish"
    return "neutral"

def _fed_rate_impact(cur: float | None, prev: float | None) -> str:
    """High absolute level AND recent hikes → bearish; cuts → bullish."""
    if cur is None:
        return "neutral"
    if cur > 5.0:
        return "bearish"
    if prev is not None and cur < prev - 0.05:
        return "bullish"  # cutting cycle
    if cur < 2.0:
        return "bullish"
    return "neutral"

def _yield_10y_impact(cur: float | None, prev: float | None) -> str:
    """Rising yields compress equity P/E multiples → bearish for stocks."""
    if cur is None:
        return "neutral"
    rising = prev is not None and cur > prev + 0.15
    if cur > 4.5 or rising:
        return "bearish"
    falling = prev is not None and cur < prev - 0.15
    if cur < 3.0 or falling:
        return "bullish"
    return "neutral"

def _gdp_impact(cur: float | None, _prev) -> str:
    """Strong growth → earnings expansion → bullish; contraction → bearish."""
    if cur is None:
        return "neutral"
    if cur > 3.0:
        return "bullish"
    if cur < 0.0:
        return "bearish"
    return "neutral"

def _unemployment_impact(cur: float | None, prev: float | None) -> str:
    """Rising unemployment → weakening consumer → bearish; tight labour → bullish."""
    if cur is None:
        return "neutral"
    rising = prev is not None and cur > prev + 0.2
    if cur > 5.0 or rising:
        return "bearish"
    if cur < 4.0:
        return "bullish"
    return "neutral"

def _pmi_impact(cur: float | None, _prev) -> str:
    """PMI >50 = expansion (bullish); <50 = contraction (bearish)."""
    if cur is None:
        return "neutral"
    if cur > 52.0:
        return "bullish"
    if cur < 48.0:
        return "bearish"
    return "neutral"

def _trend_arrow(cur: float | None, prev: float | None) -> str:
    """Visual arrow comparing current reading to prior period."""
    if cur is None or prev is None:
        return "→"
    if cur > prev * 1.001:
        return "↑"
    if cur < prev * 0.999:
        return "↓"
    return "→"

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def _fetch_fred_levels(series_id: str, limit: int = 14) -> list[float]:
    """
    Return the most recent `limit` numeric observations (newest first).
    Returns empty list if FRED_API_KEY is absent or the request fails.
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        resp = httpx.get(
            _FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": str(limit),
            },
            timeout=8.0,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        vals: list[float] = []
        for o in obs:
            try:
                v = float(o["value"])  # raises ValueError for "."
                vals.append(v)
            except (ValueError, KeyError):
                pass
        return vals
    except Exception as exc:
        logger.debug("[macro] FRED %s failed: %s", series_id, exc)
        return []

def _fred_latest_two(series_id: str) -> tuple[float | None, float | None]:
    """Convenience: return (current, previous) from a FRED level series."""
    vals = _fetch_fred_levels(series_id, limit=2)
    cur = vals[0] if len(vals) > 0 else None
    prev = vals[1] if len(vals) > 1 else None
    return cur, prev

def _fred_yoy(series_id: str) -> tuple[float | None, float | None]:
    """
    Compute YoY % change from a FRED level series.
    Returns (current_yoy, previous_month_yoy).
    Needs 14 months: current vs 12 months ago, previous vs 13 months ago.
    """
    vals = _fetch_fred_levels(series_id, limit=14)
    if len(vals) < 13:
        return None, None
    try:
        cur_yoy = (vals[0] / vals[12] - 1.0) * 100.0
        prev_yoy = (vals[1] / vals[13] - 1.0) * 100.0 if len(vals) >= 14 else None
        return round(cur_yoy, 2), (round(prev_yoy, 2) if prev_yoy is not None else None)
    except ZeroDivisionError:
        return None, None

# Registration is optional for higher rate limits.

_BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

_BLS_SERIES = {
    "CPI": "CUUR0000SA0",  # CPI-U All Items (index level)
    "PPI": "WPSFD49104",  # PPI Final Demand (index level)
    "Unemployment": "LNS14000000",  # Civilian Unemployment Rate (%)
}

def _fetch_bls(series_key: str) -> tuple[float | None, float | None]:
    """
    Fetch the latest two values for a BLS series (no API key required).
    For CPI/PPI returns the index level; unemployment returns the rate directly.
    """
    series_id = _BLS_SERIES.get(series_key)
    if not series_id:
        return None, None
    try:
        now = datetime.now()
        payload = {
            "seriesid": [series_id],
            "startyear": str(now.year - 1),
            "endyear": str(now.year),
        }
        resp = httpx.post(
            _BLS_API_URL,
            json=payload,
            timeout=10.0,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        series_list = resp.json().get("Results", {}).get("series", [])
        if not series_list:
            return None, None
        # BLS returns newest first
        obs = series_list[0].get("data", [])
        vals: list[float] = []
        for o in obs:
            with contextlib.suppress(ValueError, KeyError):
                vals.append(float(o["value"]))
        cur = vals[0] if len(vals) > 0 else None
        prev = vals[1] if len(vals) > 1 else None
        return cur, prev
    except Exception as exc:
        logger.debug("[macro] BLS %s failed: %s", series_key, exc)
        return None, None

def _bls_yoy(series_key: str) -> tuple[float | None, float | None]:
    """
    Compute YoY % change from the BLS index level series.
    Fetches 14 months of data, same logic as _fred_yoy.
    """
    series_id = _BLS_SERIES.get(series_key)
    if not series_id:
        return None, None
    try:
        now = datetime.now()
        payload = {
            "seriesid": [series_id],
            "startyear": str(now.year - 2),  # two years for 14+ months
            "endyear": str(now.year),
        }
        resp = httpx.post(
            _BLS_API_URL,
            json=payload,
            timeout=10.0,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        series_list = resp.json().get("Results", {}).get("series", [])
        if not series_list:
            return None, None
        obs = series_list[0].get("data", [])
        vals = []
        for o in obs:
            with contextlib.suppress(ValueError, KeyError):
                vals.append(float(o["value"]))
        if len(vals) < 13:
            return None, None
        cur_yoy = round((vals[0] / vals[12] - 1.0) * 100.0, 2)
        prev_yoy = round((vals[1] / vals[13] - 1.0) * 100.0, 2) if len(vals) >= 14 else None
        return cur_yoy, prev_yoy
    except Exception as exc:
        logger.debug("[macro] BLS YoY %s failed: %s", series_key, exc)
        return None, None

def _fetch_yf_rate(ticker: str) -> tuple[float | None, float | None]:
    """
    Fetch current and ~1-month-ago close from yfinance.
    Used for ^TNX (10Y yield) and ^IRX (3-month T-bill ≈ Fed funds proxy).
    """
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
        if hist.empty:
            return None, None
        cur = round(float(hist["Close"].iloc[-1]), 3)
        prev_idx = max(0, len(hist) - 22)  # ~22 trading days = 1 month
        prev = round(float(hist["Close"].iloc[prev_idx]), 3)
        return cur, prev
    except Exception as exc:
        logger.debug("[macro] yfinance %s failed: %s", ticker, exc)
        return None, None

def _fetch_worldbank_gdp() -> tuple[float | None, float | None]:
    """
    Fetch US annual real GDP growth (%) from the World Bank open API.
    Free, no key required.  Returns most recent two years.
    """
    try:
        resp = httpx.get(
            "https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.KD.ZG"
            "?format=json&per_page=5&mrv=5",
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            return None, None
        records = [r for r in data[1] if r.get("value") is not None]
        if not records:
            return None, None
        cur = round(float(records[0]["value"]), 2)
        prev = round(float(records[1]["value"]), 2) if len(records) > 1 else None
        return cur, prev
    except Exception as exc:
        logger.debug("[macro] World Bank GDP failed: %s", exc)
        return None, None

def _indicator(
    name: str,
    full_name: str,
    cur: float | None,
    prev: float | None,
    unit: str,
    description: str,
    impact_fn,
) -> dict:
    return {
        "name": name,
        "full_name": full_name,
        "current": round(cur, 2) if cur is not None else None,
        "previous": round(prev, 2) if prev is not None else None,
        "unit": unit,
        "description": description,
        "impact": impact_fn(cur, prev),  # "bullish" | "bearish" | "neutral"
        "arrow": _trend_arrow(cur, prev),  # "↑" | "↓" | "→"
    }

def fetch_macro_indicators(force_refresh: bool = False) -> dict:
    """
    Fetch all macroeconomic indicators.

    Results are cached for 4 hours - macro data is released monthly/quarterly
    so re-fetching every poll cycle would be wasteful and may hit rate limits.

    Set force_refresh=True to bypass the cache (e.g. on user-initiated reload).

    Returns:
        {
          "indicators": [ { name, full_name, current, previous, unit,
                            description, impact, arrow }, ... ],
          "fetched_at": ISO-8601 UTC string,
          "fred_active": bool   # whether FRED API key is configured
        }
    """
    cache_key = "macro_indicators"
    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("[macro] returning cached indicators")
            return cached

    logger.info("[macro] fetching fresh macro indicators …")
    fred_key = bool(os.getenv("FRED_API_KEY", "").strip())
    indicators: list[dict] = []

    # FRED CPIAUCSL is a monthly index level; compute YoY % ourselves.
    # BLS CUUR0000SA0 is the same series via the BLS API.
    if fred_key:
        cpi_cur, cpi_prev = _fred_yoy("CPIAUCSL")
    else:
        cpi_cur, cpi_prev = _bls_yoy("CPI")
    indicators.append(
        _indicator(
            "CPI",
            "Consumer Price Index (YoY %)",
            cpi_cur,
            cpi_prev,
            "% YoY",
            "Measures consumer price inflation. High CPI → Fed raises rates → equity valuations fall.",
            _cpi_impact,
        )
    )

    # FRED PPIFID is a level; compute MoM % change for display.
    if fred_key:
        ppi_levels = _fetch_fred_levels("PPIFID", limit=14)
        if len(ppi_levels) >= 13:
            ppi_cur = round((ppi_levels[0] / ppi_levels[12] - 1.0) * 100.0, 2)
            ppi_prev = (
                round((ppi_levels[1] / ppi_levels[13] - 1.0) * 100.0, 2) if len(ppi_levels) >= 14 else None
            )
        else:
            ppi_cur, ppi_prev = None, None
    else:
        ppi_cur, ppi_prev = _bls_yoy("PPI")
    indicators.append(
        _indicator(
            "PPI",
            "Producer Price Index - Final Demand (YoY %)",
            ppi_cur,
            ppi_prev,
            "% YoY",
            "Upstream cost gauge. Rising PPI signals future consumer inflation and margin pressure.",
            _ppi_impact,
        )
    )

    # FRED PCEPILFE is a monthly index level.
    if fred_key:
        pce_cur, pce_prev = _fred_yoy("PCEPILFE")
    else:
        pce_cur, pce_prev = None, None
    indicators.append(
        _indicator(
            "Core PCE",
            "Core PCE Price Index (YoY %)",
            pce_cur,
            pce_prev,
            "% YoY",
            "Fed's preferred inflation gauge (excludes food & energy). 2 % is the policy target.",
            _pce_impact,
        )
    )

    # FRED FEDFUNDS: effective federal funds rate (monthly average, %).
    # Fallback: ^IRX is the 13-week T-bill, a close proxy.
    if fred_key:
        fed_cur, fed_prev = _fred_latest_two("FEDFUNDS")
    else:
        fed_cur, fed_prev = _fetch_yf_rate("^IRX")
    indicators.append(
        _indicator(
            "Fed Rate",
            "Federal Funds Rate",
            fed_cur,
            fed_prev,
            "%",
            "Current target rate. High rates = tighter financial conditions = lower equity multiples.",
            _fed_rate_impact,
        )
    )

    # FRED DGS10: daily 10-year constant maturity yield (%).
    # Fallback: yfinance ^TNX (10-year T-note yield %).
    if fred_key:
        y10_cur, y10_prev = _fred_latest_two("DGS10")
    else:
        y10_cur, y10_prev = _fetch_yf_rate("^TNX")
    indicators.append(
        _indicator(
            "10Y Yield",
            "10-Year US Treasury Yield",
            y10_cur,
            y10_prev,
            "%",
            "Risk-free rate benchmark. Rising yields compress equity P/E multiples and raise cost of capital.",
            _yield_10y_impact,
        )
    )

    # FRED A191RL1Q225SBEA: quarterly real GDP growth rate (annualised, %).
    # Fallback: World Bank annual GDP growth.
    if fred_key:
        gdp_cur, gdp_prev = _fred_latest_two("A191RL1Q225SBEA")
    else:
        gdp_cur, gdp_prev = _fetch_worldbank_gdp()
    indicators.append(
        _indicator(
            "GDP Growth",
            "Real GDP Growth Rate",
            gdp_cur,
            gdp_prev,
            "% QoQ ann.",
            "Economic output growth. Strong GDP supports corporate earnings; contraction signals recession.",
            _gdp_impact,
        )
    )

    # FRED UNRATE: monthly civilian unemployment rate (%).
    # BLS LNS14000000: same series via BLS API.
    if fred_key:
        unemp_cur, unemp_prev = _fred_latest_two("UNRATE")
    else:
        unemp_cur, unemp_prev = _fetch_bls("Unemployment")
    indicators.append(
        _indicator(
            "Unemployment",
            "Civilian Unemployment Rate",
            unemp_cur,
            unemp_prev,
            "%",
            "Labour market health. Rising unemployment weakens consumer spending; low rate supports growth.",
            _unemployment_impact,
        )
    )

    # FRED NAPM: ISM Manufacturing PMI (index, 50 = neutral).
    # Only available via FRED (no free public alternative).
    if fred_key:
        ism_cur, ism_prev = _fred_latest_two("NAPM")
    else:
        ism_cur, ism_prev = None, None
    indicators.append(
        _indicator(
            "ISM Mfg PMI",
            "ISM Manufacturing PMI",
            ism_cur,
            ism_prev,
            "index",
            "Survey of factory activity. Above 50 = expansion (bullish); below 50 = contraction (bearish).",
            _pmi_impact,
        )
    )

    result = {
        "indicators": indicators,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "fred_active": fred_key,
        # Hint for dashboard: if FRED key is absent, some fields will be None
        "data_note": (
            "Full coverage active (FRED API key detected)."
            if fred_key
            else "Partial coverage - set FRED_API_KEY env var for CPI/PPI/PCE/GDP/ISM data. "
            "10Y yield and Fed rate proxy are sourced from yfinance (^TNX / ^IRX)."
        ),
    }
    _cache_put(cache_key, result)
    logger.info("[macro] indicators fetched and cached (%d indicators)", len(indicators))
    return result
