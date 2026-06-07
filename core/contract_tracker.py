"""Single option contract premium and buy/sell volume tracker."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.db import sqlite_connect
from core.yf_client import is_rate_limit, safe_float, safe_int, yf_call

logger = logging.getLogger(__name__)

VALID_RANGES: dict[str, str] = {
    # range -> yahoo period
    "1d": "1d",
    "3d": "3d",
    "5d": "5d",
    "7d": "7d",
    "1mo": "1mo",
    "2mo": "60d",
}

VALID_BUCKETS: dict[str, int] = {
    # bucket -> seconds
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}

# Yahoo intraday limits: 1m -> 7d, 2m..30m -> 60d, 1h -> 730d.
_MAX_WATCHERS = 12


def build_occ_symbol(ticker: str, expiry: str, strike: float, option_type: str) -> str:
    """OCC option symbol: TICKER + YYMMDD + C/P + strike*1000 zero-padded to 8.

    build_occ_symbol("MU", "2026-06-12", 960, "put") -> "MU260612P00960000"
    """
    t = ticker.upper().strip().replace("-", "")
    d = datetime.strptime(expiry, "%Y-%m-%d")
    cp = "C" if option_type.lower().startswith("c") else "P"
    return f"{t}{d:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


# Chain lookups (for the contract picker UI)


def list_expiries(ticker: str) -> list[str]:
    """All listed option expiries for a ticker."""
    import yfinance as yf

    t = yf.Ticker(ticker.upper().strip())
    return list(yf_call(lambda: t.options) or [])


def chain_for_expiry(ticker: str, expiry: str) -> dict:
    """Strike rows (both sides) for one expiry: used to populate the picker."""
    import yfinance as yf

    t = yf.Ticker(ticker.upper().strip())
    chain = yf_call(t.option_chain, expiry)

    info = yf_call(lambda: t.fast_info)
    spot = safe_float(getattr(info, "last_price", None) or getattr(info, "regular_market_price", None))

    def _rows(df) -> list[dict]:
        out = []
        for _, row in df.iterrows():
            out.append(
                {
                    "strike": safe_float(row.get("strike")),
                    "bid": safe_float(row.get("bid")),
                    "ask": safe_float(row.get("ask")),
                    "last": safe_float(row.get("lastPrice")),
                    "volume": safe_int(row.get("volume")),
                    "open_interest": safe_int(row.get("openInterest")),
                    "implied_vol": round(safe_float(row.get("impliedVolatility")) * 100, 2),
                    "in_the_money": bool(row.get("inTheMoney", False)),
                }
            )
        out.sort(key=lambda r: r["strike"])
        return out

    return {
        "ticker": ticker.upper().strip(),
        "expiry": expiry,
        "spot": round(spot, 4),
        "calls": _rows(chain.calls),
        "puts": _rows(chain.puts),
    }


def contract_snapshot(ticker: str, expiry: str, strike: float, option_type: str) -> dict | None:
    """Live quote row for one contract from the chain (None if not found)."""
    chain = chain_for_expiry(ticker, expiry)
    side = chain["calls"] if option_type.lower().startswith("c") else chain["puts"]
    best = None
    for row in side:
        if abs(row["strike"] - strike) < 1e-6:
            best = row
            break
    if best is None:
        return None
    bid, ask = best["bid"], best["ask"]
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else best["last"]
    return {
        **best,
        "ticker": chain["ticker"],
        "expiry": expiry,
        "option_type": "call" if option_type.lower().startswith("c") else "put",
        "spot": chain["spot"],
        "mid": round(mid, 4),
        "total_premium": round(best["volume"] * mid * 100, 2),
        "occ_symbol": build_occ_symbol(ticker, expiry, strike, option_type),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


# Backfill: intraday premium bars for the OCC symbol


def _estimate_bar_split(close: float, prev_close: float | None, high: float, low: float) -> float:
    """Estimated buy fraction (0..1) of a bar's volume - tick-rule style.

    Blends two cues: direction of change vs previous bar close, and where the
    close sits inside the bar's high-low range.
    """
    # Position-in-range cue
    if high > low:
        pos = (close - low) / (high - low)
    else:
        pos = 0.5
    # Tick-rule cue
    if prev_close is None or prev_close <= 0:
        tick = 0.5
    elif close > prev_close:
        tick = 1.0
    elif close < prev_close:
        tick = 0.0
    else:
        tick = 0.5
    frac = 0.5 * pos + 0.5 * tick
    return min(max(frac, 0.0), 1.0)


def fetch_contract_bars(
    occ_symbol: str, range_: str = "7d", bucket: str = "30m"
) -> tuple[list[dict], str | None]:
    """Intraday premium bars for an option contract with estimated buy/sell split.

    Returns ``(bars, hint)`` where bars is a list of dicts (oldest first):
      {ts, open, high, low, close, volume, buy_vol, sell_vol, source: "est"}
    and hint is None on success, or one of "rate_limited" / "no_data" /
    "error: ..." when bars came back empty - so the UI can say WHY.
    """
    import yfinance as yf

    period = VALID_RANGES.get(range_, "7d")
    if bucket not in VALID_BUCKETS:
        bucket = "30m"

    t = yf.Ticker(occ_symbol)
    # Illiquid contracts may have zero prints in a short window - escalate the
    # lookback once so the chart shows whatever history exists instead of nothing.
    periods = [period]
    if period in ("1d", "3d", "5d", "7d"):
        periods.append("1mo")

    df = None
    hint: str | None = None
    for p in periods:
        try:
            # raise_errors=True: yfinance raises instead of logging a bogus
            # "possibly delisted" ERROR, letting us tell rate-limits apart
            # from genuinely traded-nothing contracts.
            df = yf_call(t.history, period=p, interval=bucket, prepost=False, raise_errors=True)
        except Exception as exc:
            if is_rate_limit(exc):
                logger.warning("[ContractTracker] %s: Yahoo rate limit (period=%s)", occ_symbol, p)
                return [], "rate_limited"
            msg = str(exc).lower()
            if "no price data" in msg or "possibly delisted" in msg or "no data found" in msg:
                hint = "no_data"
                continue  # try the wider period
            logger.warning("[ContractTracker] %s history failed (period=%s): %s", occ_symbol, p, exc)
            return [], "error: " + str(exc)[:160]
        if df is not None and not df.empty:
            hint = None
            if p != period:
                logger.debug("[ContractTracker] %s: no bars in %s, widened to %s", occ_symbol, period, p)
            break
        hint = "no_data"
    if df is None or df.empty:
        return [], hint or "no_data"

    bars: list[dict] = []
    prev_close: float | None = None
    for ts, row in df.iterrows():
        close = safe_float(row.get("Close"))
        high = safe_float(row.get("High"), close)
        low = safe_float(row.get("Low"), close)
        vol = safe_int(row.get("Volume"))
        if close <= 0:
            continue
        frac = _estimate_bar_split(close, prev_close, high, low)
        buy = int(round(vol * frac))
        opn = safe_float(row.get("Open"), close)
        # OHLC4 as the average fill estimate for the bucket
        avg_fill = (opn + high + low + close) / 4.0
        bars.append(
            {
                "ts": ts.isoformat(),
                "open": round(opn, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": vol,
                "buy_vol": buy,
                "sell_vol": vol - buy,
                "avg_fill": round(avg_fill, 4),
                "premium": round(vol * avg_fill * 100.0, 2),
                "source": "est",
            }
        )
        prev_close = close
    return bars, None


# Live tick store (SQLite)

_TICKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS contract_ticks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    occ         TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    last        REAL    NOT NULL,
    bid         REAL    NOT NULL,
    ask         REAL    NOT NULL,
    vol_delta   INTEGER NOT NULL,
    day_volume  INTEGER NOT NULL,
    oi          INTEGER NOT NULL,
    side        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contract_ticks_occ_ts ON contract_ticks(occ, ts);
"""


_ticks_write_lock = threading.Lock()
_ticks_schema_ready = False


def _ticks_db_path() -> str:
    from config import cfg

    return cfg.db_path


def _ticks_connect() -> sqlite3.Connection:
    return sqlite_connect(_ticks_db_path())


def _ensure_ticks_schema() -> None:
    global _ticks_schema_ready
    if _ticks_schema_ready:
        return
    with _ticks_write_lock, _ticks_connect() as conn:
        conn.executescript(_TICKS_SCHEMA)
    _ticks_schema_ready = True


def _record_tick(occ: str, tick: dict) -> None:
    _ensure_ticks_schema()
    with _ticks_write_lock, _ticks_connect() as conn:
        conn.execute(
            "INSERT INTO contract_ticks (occ, ts, last, bid, ask, vol_delta, day_volume, oi, side)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                occ,
                tick["ts"],
                tick["last"],
                tick["bid"],
                tick["ask"],
                tick["vol_delta"],
                tick["day_volume"],
                tick["oi"],
                tick["side"],
            ),
        )


def _tick_buckets(occ: str, bucket_seconds: int, since_iso: str | None = None) -> list[dict]:
    """Aggregate ticks into time buckets: live buy/sell/mid volume + last price."""
    _ensure_ticks_schema()
    q = (
        "SELECT ts, last, vol_delta, side FROM contract_ticks"
        " WHERE occ = ?" + (" AND ts >= ?" if since_iso else "") + " ORDER BY ts ASC"
    )
    args = (occ, since_iso) if since_iso else (occ,)
    with _ticks_connect() as conn:
        rows = conn.execute(q, args).fetchall()

    out: dict[int, dict] = {}
    for r in rows:
        with suppress(ValueError):
            ts = datetime.fromisoformat(r["ts"])
            epoch = int(ts.timestamp())
            key = epoch - (epoch % bucket_seconds)
            b = out.setdefault(
                key,
                {"buy_vol": 0, "sell_vol": 0, "mid_vol": 0, "close": 0.0, "ticks": 0, "_pv": 0.0},
            )
            vd = int(r["vol_delta"])
            side = r["side"]
            if side == "ask":
                b["buy_vol"] += vd
            elif side == "bid":
                b["sell_vol"] += vd
            else:
                b["mid_vol"] += vd
            b["close"] = float(r["last"])
            b["_pv"] += float(r["last"]) * vd  # for bucket VWAP (avg fill)
            b["ticks"] += 1

    result = []
    for k, v in sorted(out.items()):
        total = v["buy_vol"] + v["sell_vol"] + v["mid_vol"]
        avg_fill = (v.pop("_pv") / total) if total > 0 else v["close"]
        result.append(
            {
                "ts": datetime.fromtimestamp(k, tz=timezone.utc).isoformat(),
                "avg_fill": round(avg_fill, 4),
                **v,
            }
        )
    return result


# Live poller


def _classify_side(last: float, bid: float, ask: float) -> str:
    """Ask-side (buy) / bid-side (sell) / mid classification of a print."""
    if bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if last >= ask * 0.999:
            return "ask"
        if last <= bid * 1.001:
            return "bid"
        if last > mid:
            return "ask"
        if last < mid:
            return "bid"
    return "mid"


@dataclass
class _Watcher:
    ticker: str
    expiry: str
    strike: float
    option_type: str
    occ: str
    poll_seconds: int
    started_at: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    last_day_volume: int | None = None
    last_error: str | None = None
    tick_count: int = 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "expiry": self.expiry,
            "strike": self.strike,
            "option_type": self.option_type,
            "occ_symbol": self.occ,
            "poll_seconds": self.poll_seconds,
            "started_at": self.started_at,
            "tick_count": self.tick_count,
            "last_error": self.last_error,
        }


_watcher_lock = threading.Lock()
_watchers: dict[str, _Watcher] = {}


def watch(
    ticker: str, expiry: str, strike: float, option_type: str, poll_seconds: int = 30
) -> dict:
    occ = build_occ_symbol(ticker, expiry, strike, option_type)
    poll_seconds = max(10, min(poll_seconds, 600))
    with _watcher_lock:
        if occ in _watchers:
            return _watchers[occ].to_dict()
        if len(_watchers) >= _MAX_WATCHERS:
            raise RuntimeError(f"Max {_MAX_WATCHERS} watched contracts - unwatch one first")
        w = _Watcher(
            ticker=ticker.upper().strip(),
            expiry=expiry,
            strike=float(strike),
            option_type="call" if option_type.lower().startswith("c") else "put",
            occ=occ,
            poll_seconds=poll_seconds,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        w.thread = threading.Thread(target=_run_watcher, args=(w,), daemon=True, name=f"track-{occ}")
        _watchers[occ] = w
        w.thread.start()
    logger.info("[ContractTracker] watching %s every %ds", occ, poll_seconds)
    return w.to_dict()


def unwatch(ticker: str, expiry: str, strike: float, option_type: str) -> bool:
    occ = build_occ_symbol(ticker, expiry, strike, option_type)
    with _watcher_lock:
        w = _watchers.pop(occ, None)
    if w is None:
        return False
    w.stop_event.set()
    logger.info("[ContractTracker] unwatched %s", occ)
    return True


def is_watched(occ: str) -> bool:
    with _watcher_lock:
        return occ in _watchers


def watched() -> list[dict]:
    with _watcher_lock:
        return [w.to_dict() for w in _watchers.values()]


def live_buckets(occ: str, bucket_seconds: int, since_iso: str | None = None) -> list[dict]:
    return _tick_buckets(occ, bucket_seconds, since_iso)


def stop_all() -> None:
    with _watcher_lock:
        watchers = list(_watchers.values())
        _watchers.clear()
    for w in watchers:
        w.stop_event.set()


def _run_watcher(w: _Watcher) -> None:
    while not w.stop_event.is_set():
        try:
            snap = contract_snapshot(w.ticker, w.expiry, w.strike, w.option_type)
            if snap is None:
                w.last_error = "contract_not_in_chain"
            else:
                day_vol = safe_int(snap.get("volume"))
                last = safe_float(snap.get("last"))
                bid = safe_float(snap.get("bid"))
                ask = safe_float(snap.get("ask"))

                if w.last_day_volume is None or day_vol < w.last_day_volume:
                    # First poll, or Yahoo's session volume counter reset (new day).
                    delta = 0
                else:
                    delta = day_vol - w.last_day_volume
                w.last_day_volume = day_vol

                if delta > 0 and last > 0:
                    _record_tick(
                        w.occ,
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "last": last,
                            "bid": bid,
                            "ask": ask,
                            "vol_delta": delta,
                            "day_volume": day_vol,
                            "oi": safe_int(snap.get("open_interest")),
                            "side": _classify_side(last, bid, ask),
                        },
                    )
                    w.tick_count += 1
                w.last_error = None

            # Stop automatically once the contract has expired.
            if datetime.strptime(w.expiry, "%Y-%m-%d").date() < datetime.now(timezone.utc).date():
                logger.info("[ContractTracker] %s expired - stopping watcher", w.occ)
                break
        except Exception as exc:
            w.last_error = str(exc)[:200]
            logger.warning("[ContractTracker] %s poll failed: %s", w.occ, exc)

        w.stop_event.wait(w.poll_seconds)

    with _watcher_lock:
        _watchers.pop(w.occ, None)


# Combined view (backfill + live overlay)


def get_contract_view(
    ticker: str,
    expiry: str,
    strike: float,
    option_type: str,
    range_: str = "7d",
    bucket: str = "30m",
) -> dict:
    """Everything the tracker UI needs for one contract.

    Backfilled bars carry the tick-rule *estimated* buy/sell split; buckets that
    have live poller ticks get their split replaced by the observed ask/bid-side
    volume (source: "live").
    """
    occ = build_occ_symbol(ticker, expiry, strike, option_type)
    if bucket not in VALID_BUCKETS:
        bucket = "30m"
    if range_ not in VALID_RANGES:
        range_ = "7d"

    bars, bars_hint = fetch_contract_bars(occ, range_, bucket)

    # Overlay live-classified buckets where available
    bucket_secs = VALID_BUCKETS[bucket]
    since = bars[0]["ts"] if bars else None
    live = live_buckets(occ, bucket_secs, since)
    if live:
        by_key: dict[int, dict] = {}
        for b in bars:
            with suppress(ValueError):
                epoch = int(datetime.fromisoformat(b["ts"]).timestamp())
                by_key[epoch - (epoch % bucket_secs)] = b
        for lb in live:
            with suppress(ValueError):
                epoch = int(datetime.fromisoformat(lb["ts"]).timestamp())
                key = epoch - (epoch % bucket_secs)
                classified = lb["buy_vol"] + lb["sell_vol"] + lb["mid_vol"]
                if classified <= 0:
                    continue
                avg_fill = float(lb.get("avg_fill") or lb["close"])
                if key in by_key:
                    b = by_key[key]
                    b["buy_vol"] = lb["buy_vol"]
                    b["sell_vol"] = lb["sell_vol"]
                    b["mid_vol"] = lb["mid_vol"]
                    b["avg_fill"] = round(avg_fill, 4)
                    b["premium"] = round(classified * avg_fill * 100.0, 2)
                    b["source"] = "live"
                else:
                    bars.append(
                        {
                            "ts": datetime.fromtimestamp(key, tz=timezone.utc).isoformat(),
                            "open": lb["close"],
                            "high": lb["close"],
                            "low": lb["close"],
                            "close": lb["close"],
                            "volume": classified,
                            "buy_vol": lb["buy_vol"],
                            "sell_vol": lb["sell_vol"],
                            "mid_vol": lb["mid_vol"],
                            "avg_fill": round(avg_fill, 4),
                            "premium": round(classified * avg_fill * 100.0, 2),
                            "source": "live",
                        }
                    )
        bars.sort(key=lambda b: b["ts"])

    snapshot = None
    try:
        snapshot = contract_snapshot(ticker, expiry, strike, option_type)
    except Exception as exc:
        logger.warning("[ContractTracker] snapshot failed for %s: %s", occ, exc)

    return {
        "ticker": ticker.upper().strip(),
        "expiry": expiry,
        "strike": float(strike),
        "option_type": "call" if option_type.lower().startswith("c") else "put",
        "occ_symbol": occ,
        "range": range_,
        "bucket": bucket,
        "tracked": is_watched(occ),
        "snapshot": snapshot,
        "bars": bars,
        "bars_hint": None if bars else bars_hint,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
