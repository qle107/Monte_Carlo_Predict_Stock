"""OHLCV candle fetcher with caching."""

import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Prevents duplicate network round-trips when the scanner fetches the same
# ticker concurrently (e.g. scan pass + MC top-N pass both need AAPL/1d/60).
# Cache is keyed by (ticker, interval, lookback, extended) and expires after
# _CACHE_TTL seconds.  Thread-safe via a single RLock.

_CACHE_TTL = 30.0  # seconds before a cached DataFrame is evicted
_cache_lock = threading.RLock()
_cache: dict = {}  # key -> (df, expire_time)

# If two threads request the same (ticker, interval, lookback, extended) within
# milliseconds of each other (both miss the TTL cache) the second thread waits
# for the first (the "leader") to finish and then reads from the cache instead
# of launching its own network request.  This eliminates the thundering-herd
# duplicate fetch that happens when update_config restarts the poll loop while
# also calling _run_analysis() immediately.

_inflight_lock = threading.RLock()
_inflight: dict = {}  # key -> threading.Event


def _cache_get(key: tuple) -> pd.DataFrame | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        df, exp = entry
        if time.monotonic() > exp:
            del _cache[key]
            return None
        return df.copy()  # return a copy so callers can't mutate the cache


def _cache_put(key: tuple, df: pd.DataFrame) -> None:
    with _cache_lock:
        # Evict expired entries to prevent unbounded growth
        now = time.monotonic()
        expired = [k for k, (_, exp) in _cache.items() if now > exp]
        for k in expired:
            del _cache[k]
        _cache[key] = (df.copy(), now + _CACHE_TTL)


def clear_fetch_cache() -> None:
    """Flush the entire in-process fetch cache (useful in tests)."""
    with _cache_lock:
        _cache.clear()


_RETRY_ATTEMPTS = 3
_RETRY_BASE_S = 1.0  # first sleep = 1 s, then 2 s, then 4 s


def _with_retry(fn, *args, label: str = "fetch"):
    """
    Call fn(*args) up to _RETRY_ATTEMPTS times with exponential back-off.
    Raises the last exception if all attempts fail.
    """
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn(*args)
        except Exception as exc:
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                sleep_s = _RETRY_BASE_S * (2**attempt)
                logger.debug(
                    "[fetcher] %s attempt %d failed (%s) - retrying in %.1fs",
                    label,
                    attempt + 1,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)
    raise last_exc


# Pre-market:  04:00 - 09:29
# Regular:     09:30 - 15:59
# After-hours: 16:00 - 19:59
# Closed:      20:00 - 03:59

_REGULAR_START = (9, 30)  # hour, minute ET
_REGULAR_END = (16, 0)
_PRE_START = (4, 0)
_AFTER_END = (20, 0)

_YF_MAX_DAYS = {
    "1m": 7,  # yfinance hard limit for 1m
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "4h": 730,
    "1d": 3650,
}

_INTERVAL_MINUTES = {
    "1m": 1,
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _lookback_days(interval: str, n_candles: int, buffer: float = 1.6) -> int:
    """
    How many calendar days to request to reliably get n_candles.

    For daily ('1d') bars:
      1 trading day ≈ 1 calendar day, but weekends + holidays mean we need
      ~1.45× more calendar days. We also add a fixed 10-day cushion so that
      weekend/holiday runs never return 0 rows.

    For intraday bars:
      candles_per_trading_day = 390 min / interval_min
      calendar_days_needed    = (n_candles / cpd) * buffer + 3
    """
    if interval == "1d":
        # Each trading day = 1 bar; ~5/7 calendar days are trading days.
        needed = max(int(n_candles * 1.5) + 10, 14)  # always >=14 calendar days
        return min(needed, _YF_MAX_DAYS.get("1d", 3650))

    mins = _INTERVAL_MINUTES.get(interval, 15)
    cpd = 390.0 / mins  # candles per trading day
    needed = max(int((n_candles / cpd) * buffer) + 3, 5)
    return min(needed, _YF_MAX_DAYS.get(interval, 60))


def current_session() -> str:
    """Return the current US market session based on ET wall-clock time."""
    try:
        import zoneinfo

        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except ImportError:
        # fallback: UTC-5 (rough ET, ignores DST)
        et = datetime.now(timezone.utc) - timedelta(hours=5)

    h, m = et.hour, et.minute
    weekday = et.weekday()  # 0=Mon ... 6=Sun

    if weekday >= 5:
        return "closed"
    if (h > _REGULAR_START[0] or (h == _REGULAR_START[0] and m >= _REGULAR_START[1])) and (
        h < _REGULAR_END[0]
    ):
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
        return current_session()  # fall back to clock-based detection
    try:
        last_et = df.index[-1].tz_convert("America/New_York")
    except Exception:
        return current_session()

    h, m = last_et.hour, last_et.minute
    # Daily bars arrive as midnight UTC (= 7pm/8pm ET) - treat as regular
    if h == 0 and m == 0:
        return "regular"
    if (h > 9 or (h == 9 and m >= 30)) and h < 16:
        return "regular"
    if 4 <= h < 9 or (h == 9 and m < 30):
        return "pre-market"
    if 16 <= h < 20:
        return "after-hours"
    return "closed"


def _filter_regular_hours(df: pd.DataFrame, interval: str = "") -> pd.DataFrame:
    """
    Strip candles outside 9:30am-4:00pm ET.
    For daily bars the index is date-only (midnight UTC) - skip filtering,
    otherwise we'd remove every row.
    """
    # Daily bars: midnight timestamps have hour==0, filtering would wipe them.
    if interval == "1d" or interval == "4h":
        return df
    try:
        et = df.index.tz_convert("America/New_York")
        # If ALL bars are at midnight it's a daily feed - don't filter
        if (et.hour == 0).all():
            return df
        mask = ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (
            (et.hour < 16) | ((et.hour == 16) & (et.minute == 0))
        )
        filtered = df[mask]
        # Safety: if filtering removed everything, return original
        return filtered if len(filtered) > 0 else df
    except Exception:
        return df


def _yfinance(ticker: str, interval: str, n: int, extended: bool) -> pd.DataFrame:
    import yfinance as yf

    end = datetime.now(timezone.utc)
    days = _lookback_days(interval, n)
    start = end - timedelta(days=days)

    df = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval=interval,
        prepost=extended,
    )
    if df.empty:
        raise ValueError(f"yfinance: no data for {ticker}")

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if not extended:
        df = _filter_regular_hours(df, interval=interval)
    return df.tail(n)


def _alpaca(ticker: str, interval: str, n: int, extended: bool) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not key or key == "your_key_here":
        raise ValueError("Alpaca keys not set")

    tf_map = {
        "1m": TimeFrame.Minute,
        "2m": TimeFrame(2, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame.Hour,
        "4h": TimeFrame(4, TimeFrameUnit.Hour),
        "1d": TimeFrame.Day,
    }
    client = StockHistoricalDataClient(key, secret)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf_map.get(interval, TimeFrame(15, TimeFrameUnit.Minute)),
        start=datetime.now(timezone.utc) - timedelta(days=_lookback_days(interval, n)),
        end=datetime.now(timezone.utc),
        feed="iex",  # free: iex | paid: sip
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise ValueError(f"Alpaca: no data for {ticker}")
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level="symbol")
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars[["open", "high", "low", "close", "volume"]].dropna()
    if not extended:
        bars = _filter_regular_hours(bars, interval=interval)
    return bars.tail(n)


def _polygon(ticker: str, interval: str, n: int, extended: bool) -> pd.DataFrame:
    import httpx

    key = os.getenv("POLYGON_API_KEY", "")
    if not key:
        raise ValueError("Polygon key not set")

    mult = _INTERVAL_MINUTES.get(interval, 15)
    span = "minute" if mult < 1440 else "day"
    end_d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
            "open": r["o"],
            "high": r["h"],
            "low": r["l"],
            "close": r["c"],
            "volume": r["v"],
            "timestamp": pd.Timestamp(r["t"], unit="ms", tz="UTC"),
        }
        for r in data["results"]
    ]
    df = pd.DataFrame(rows).set_index("timestamp")
    if not extended:
        df = _filter_regular_hours(df, interval=interval)
    return df.tail(n)


def fetch_candles(
    ticker: str,
    interval: str = "15m",
    lookback: int = 50,
    extended: bool = True,
) -> pd.DataFrame:
    """
    Returns DataFrame(open, high, low, close, volume) with UTC DatetimeIndex.

    Extended hours are AUTO-ENABLED when the market is currently outside
    the regular session (pre-market or after-hours) so live data is never
    missed - regardless of the user's extended setting.

    Priority: Alpaca -> Polygon -> yfinance (always falls back).
    """

    use_extended = should_use_extended(extended)
    session_now = current_session()

    if use_extended and not extended:
        logger.debug(f"[fetcher] Market is {session_now} - auto-enabling extended hours")

    cache_key = (ticker.upper(), interval, lookback, use_extended)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("[fetcher] %s %s - cache hit (%d rows)", ticker, interval, len(cached))
        return cached

    # If another thread is already fetching the same key, wait for it to finish
    # and read from cache - avoids duplicate network calls on cache misses.
    with _inflight_lock:
        if cache_key in _inflight:
            event = _inflight[cache_key]
            is_leader = False
        else:
            event = threading.Event()
            _inflight[cache_key] = event
            is_leader = True

    if not is_leader:
        logger.debug("[fetcher] %s %s - waiting for in-flight fetch", ticker, interval)
        event.wait(timeout=35.0)  # never hang longer than the retry budget
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("[fetcher] %s %s - coalesced hit (%d rows)", ticker, interval, len(cached))
            return cached
        # Leader failed - fall through and try ourselves (different source may work)
        logger.debug("[fetcher] %s %s - leader failed, attempting own fetch", ticker, interval)
        # Become a new leader for a fresh attempt
        with _inflight_lock:
            event2 = threading.Event()
            _inflight[cache_key] = event2
            event = event2
            is_leader = True

    sources = []

    # Uncomment the block below to re-enable Alpaca (requires a valid key).
    # When the key is invalid (HTTP 401) _with_retry wastes ~3-4 s per call
    # retrying a permanent auth failure before falling back to yfinance.
    # if os.getenv("ALPACA_API_KEY", "") not in ("", "your_key_here"):
    #     sources.append(("Alpaca", _alpaca))
    if os.getenv("POLYGON_API_KEY", ""):
        sources.append(("Polygon", _polygon))
    sources.append(("yfinance", _yfinance))  # always available as final fallback

    last_err = None
    try:
        for name, fn in sources:
            try:
                df = _with_retry(fn, ticker, interval, lookback, use_extended, label=f"{name}/{ticker}")

                # Validate result
                if df is None or len(df) < 5:
                    raise ValueError(f"Too few rows ({len(df) if df is not None else 0})")
                df = df[np.isfinite(pd.to_numeric(df["close"], errors="coerce"))]
                if len(df) < 2:
                    raise ValueError("All close prices are invalid after filtering")

                session = _session_label(df)
                logger.info(
                    f"[fetcher] {ticker} {interval} - {len(df)} candles via {name} "
                    f"| last_candle={session} | clock={session_now} | extended={use_extended}"
                )
                df.attrs["session"] = session
                df.attrs["session_now"] = session_now  # current clock session
                df.attrs["extended"] = use_extended

                _cache_put(cache_key, df)
                return df

            except Exception as e:
                # Strip HTML bodies and long tracebacks - log a one-line summary only
                err_str = str(e)
                if "<html" in err_str.lower() or len(err_str) > 120:
                    # Extract HTTP status code if present (e.g. "401", "403", "429")
                    status = re.search(r"\b([45]\d{2})\b", err_str)
                    short = f"HTTP {status.group(1)}" if status else err_str[:80].replace("\n", " ").strip()
                    logger.warning(f"[fetcher] {name} unavailable - {short}")
                else:
                    logger.warning(f"[fetcher] {name} failed: {err_str}")
                last_err = e

        raise RuntimeError(f"All data sources failed. Last error: {last_err}")

    finally:
        # Always release the in-flight slot so waiting threads are unblocked,
        # whether the fetch succeeded or failed.
        if is_leader:
            with _inflight_lock:
                _inflight.pop(cache_key, None)
            event.set()  # wake any threads that were waiting on this key


def get_latest_price(ticker: str) -> float | None:
    """Always fetches with extended=True to get live price at any hour."""
    try:
        df = fetch_candles(ticker, "1m", 5, extended=True)
        return float(df["close"].iloc[-1])
    except Exception:
        return None
