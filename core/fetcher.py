"""
core/fetcher.py
Fetches OHLCV candles. Priority: Alpaca (IEX, free) → Polygon → yfinance.

Extended hours (pre-market 4am–9:30am ET, after-hours 4pm–8pm ET):
- Auto-enabled when the current time is outside regular session.
- Can also be forced on/off via the `extended` parameter.
- yfinance: prepost=True includes pre/post bars.
- Alpaca IEX: naturally includes extended bars when available.
- Polygon: always includes extended bars in aggregate data.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# ── Session timing constants (US Eastern) ────────────────────────────────────
# Pre-market:  04:00 – 09:29
# Regular:     09:30 – 15:59
# After-hours: 16:00 – 19:59
# Closed:      20:00 – 03:59

_REGULAR_START = (9,  30)   # hour, minute ET
_REGULAR_END   = (16,  0)
_PRE_START     = (4,   0)
_AFTER_END     = (20,  0)

# ── Lookback window calculator ────────────────────────────────────────────────

_YF_MAX_DAYS = {
    "1m":  7,     # yfinance hard limit for 1m
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "4h":  730,
    "1d":  3650,
}

_INTERVAL_MINUTES = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}

def _lookback_days(interval: str, n_candles: int, buffer: float = 1.6) -> int:
    """
    How many calendar days to request to reliably get n_candles.
    buffer=1.6 accounts for weekends, holidays, and session gaps.
    """
    mins   = _INTERVAL_MINUTES.get(interval, 15)
    cpd    = 390 / mins                           # candles per trading day
    needed = max(int((n_candles / cpd) * buffer) + 1, 2)
    return min(needed, _YF_MAX_DAYS.get(interval, 60))


# ── Session detection ─────────────────────────────────────────────────────────

def current_session() -> str:
    """Return the current US market session based on ET wall-clock time."""
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except ImportError:
        # fallback: UTC-5 (rough ET, ignores DST)
        et = datetime.now(timezone.utc) - timedelta(hours=5)

    h, m = et.hour, et.minute
    weekday = et.weekday()   # 0=Mon … 6=Sun

    if weekday >= 5:
        return "closed"
    if (h > _REGULAR_START[0] or (h == _REGULAR_START[0] and m >= _REGULAR_START[1])) \
       and (h < _REGULAR_END[0]):
        return "regular"
    if h >= _PRE_START[0] and (h < _REGULAR_START[0] or (h == _REGULAR_START[0] and m < _REGULAR_START[1])):
        return "pre-market"
    if h >= _REGULAR_END[0] and h < _AFTER_END[0]:
        return "after-hours"
    return "closed"


def should_use_extended(user_extended: bool) -> bool:
    """
    Return True if extended-hours data should be fetched.
    Always True when outside regular session so live data is never missed.
    """
    if user_extended:
        return True
    session = current_session()
    return session in ("pre-market", "after-hours")


def _session_label(df: pd.DataFrame) -> str:
    """Tag the last candle's session."""
    if df.empty:
        return current_session()   # fall back to clock-based detection
    try:
        last_et = df.index[-1].tz_convert("America/New_York")
    except Exception:
        return current_session()

    h, m = last_et.hour, last_et.minute
    if (h > 9 or (h == 9 and m >= 30)) and h < 16:
        return "regular"
    if 4 <= h < 9 or (h == 9 and m < 30):
        return "pre-market"
    if 16 <= h < 20:
        return "after-hours"
    return "closed"


# ── Regular-hours filter ─────────────────────────────────────────────────────

def _filter_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Strip candles outside 9:30am–4:00pm ET."""
    try:
        et   = df.index.tz_convert("America/New_York")
        mask = (
            (et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))
        ) & (
            (et.hour < 16) | ((et.hour == 16) & (et.minute == 0))
        )
        return df[mask]
    except Exception:
        return df


# ── Data sources ─────────────────────────────────────────────────────────────

def _yfinance(ticker: str, interval: str, n: int, extended: bool) -> pd.DataFrame:
    import yfinance as yf
    end   = datetime.now(timezone.utc)
    days  = _lookback_days(interval, n)
    start = end - timedelta(days=days)

    df = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval=interval,
        prepost=extended,
    )
    if df.empty:
        raise ValueError(f"yfinance: no data for {ticker}")

    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low":  "low",  "Close": "close", "Volume": "volume",
    })
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if not extended:
        df = _filter_regular_hours(df)
    return df.tail(n)


def _alpaca(ticker: str, interval: str, n: int, extended: bool) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key    = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or key == "your_key_here":
        raise ValueError("Alpaca keys not set")

    tf_map = {
        "1m":  TimeFrame.Minute,
        "2m":  TimeFrame(2,  TimeFrameUnit.Minute),
        "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h":  TimeFrame.Hour,
        "4h":  TimeFrame(4,  TimeFrameUnit.Hour),
        "1d":  TimeFrame.Day,
    }
    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols = ticker,
        timeframe         = tf_map.get(interval, TimeFrame(15, TimeFrameUnit.Minute)),
        start             = datetime.now(timezone.utc) - timedelta(days=_lookback_days(interval, n)),
        end               = datetime.now(timezone.utc),
        feed              = "iex",   # free: iex | paid: sip
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise ValueError(f"Alpaca: no data for {ticker}")
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level="symbol")
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars[["open", "high", "low", "close", "volume"]].dropna()
    if not extended:
        bars = _filter_regular_hours(bars)
    return bars.tail(n)


def _polygon(ticker: str, interval: str, n: int, extended: bool) -> pd.DataFrame:
    import httpx
    key = os.getenv("POLYGON_API_KEY", "")
    if not key:
        raise ValueError("Polygon key not set")

    mult    = {"1m":1,"2m":2,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}.get(interval, 15)
    span    = "minute" if mult < 1440 else "day"
    end_d   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_d = (datetime.now(timezone.utc) - timedelta(days=_lookback_days(interval, n))).strftime("%Y-%m-%d")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
        f"{mult}/{span}/{start_d}/{end_d}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={key}"
    )
    data = httpx.get(url, timeout=10).raise_for_status().json()
    if not data.get("resultsCount"):
        raise ValueError(f"Polygon: no data for {ticker}")

    rows = [
        {
            "open": r["o"], "high": r["h"], "low": r["l"],
            "close": r["c"], "volume": r["v"],
            "timestamp": pd.Timestamp(r["t"], unit="ms", tz="UTC"),
        }
        for r in data["results"]
    ]
    df = pd.DataFrame(rows).set_index("timestamp")
    if not extended:
        df = _filter_regular_hours(df)
    return df.tail(n)


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_candles(
    ticker:   str,
    interval: str  = "15m",
    lookback: int  = 50,
    extended: bool = True,
) -> pd.DataFrame:
    """
    Returns DataFrame(open, high, low, close, volume) with UTC DatetimeIndex.

    Extended hours are AUTO-ENABLED when the market is currently outside
    the regular session (pre-market or after-hours) so live data is never
    missed — regardless of the user's extended setting.

    Priority: Alpaca → Polygon → yfinance (always falls back).
    """
    # ── Auto-enable extended outside regular hours ────────────────────────
    use_extended = should_use_extended(extended)
    session_now  = current_session()

    if use_extended and not extended:
        logger.info(
            f"[fetcher] Market is {session_now} — auto-enabling extended hours"
        )

    sources = []
    if os.getenv("ALPACA_API_KEY", "") not in ("", "your_key_here"):
        sources.append(("Alpaca",   _alpaca))
    if os.getenv("POLYGON_API_KEY", ""):
        sources.append(("Polygon",  _polygon))
    sources.append(("yfinance", _yfinance))   # always available as final fallback

    last_err = None
    for name, fn in sources:
        try:
            df = fn(ticker, interval, lookback, use_extended)

            # Validate result
            if df is None or len(df) < 5:
                raise ValueError(f"Too few rows ({len(df) if df is not None else 0})")
            df = df[df["close"].apply(lambda x: isinstance(x, (int, float))
                                                and x == x
                                                and x not in (float("inf"), float("-inf")))]
            if len(df) < 2:
                raise ValueError("All close prices are invalid after filtering")

            session = _session_label(df)
            logger.info(
                f"[fetcher] {ticker} {interval} — {len(df)} candles via {name} "
                f"| last_candle={session} | clock={session_now} | extended={use_extended}"
            )
            df.attrs["session"]      = session
            df.attrs["session_now"]  = session_now   # current clock session
            df.attrs["extended"]     = use_extended
            return df

        except Exception as e:
            logger.warning(f"[fetcher] {name} failed: {e}")
            last_err = e

    raise RuntimeError(f"All data sources failed. Last error: {last_err}")


def get_latest_price(ticker: str) -> Optional[float]:
    """Always fetches with extended=True to get live price at any hour."""
    try:
        df = fetch_candles(ticker, "1m", 5, extended=True)
        return float(df["close"].iloc[-1])
    except Exception:
        return None
