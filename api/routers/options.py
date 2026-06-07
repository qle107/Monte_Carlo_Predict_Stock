"""Options endpoints: unusual flow, hot scan, GEX, and contract tracking."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request

from core.contract_tracker import (
    VALID_BUCKETS,
    VALID_RANGES,
    chain_for_expiry,
    get_contract_view,
    list_expiries,
    unwatch,
    watch,
    watched,
)
from core.expected_move import expected_move_for_ticker
from core.forecast import forecast_for_ticker
from core.options_flow import (
    cancel_unusual_scan,
    fetch_options_flow,
    scan_unusual_options,
    scan_volume_spikes,
)
from core.pop_scanner import scan as pop_scan
from core.scanner import get_watchlist

from ..deps import limiter, require_api_key
from ..models import ContractWatchRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["options"])

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_EXPIRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_contract(ticker: str, expiry: str, strike: float, option_type: str) -> tuple:
    t = (ticker or "").upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="invalid ticker")
    if not _EXPIRY_RE.match(expiry or ""):
        raise HTTPException(status_code=400, detail="expiry must be YYYY-MM-DD")
    if not (0 < strike < 1_000_000):
        raise HTTPException(status_code=400, detail="invalid strike")
    ot = (option_type or "").lower().strip()
    if ot not in ("call", "put", "c", "p"):
        raise HTTPException(status_code=400, detail="option_type must be call or put")
    return t, expiry, float(strike), "call" if ot.startswith("c") else "put"


@router.get("/api/options/unusual")
@limiter.limit("30/minute")
async def api_unusual_options(
    request: Request,
    tickers: str | None = None,
    watchlist: str | None = None,
    min_volume: int = 100,
    min_oi: int = 50,
    vol_oi_threshold: float = 3.0,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    max_dte: int = 60,
    top_n: int = 50,
    min_premium: float = 0.0,
    new_positions_only: bool = False,
    min_sweep_premium: float = 50_000.0,
    min_block_premium: float = 100_000.0,
    exclude_bid_side: bool = True,
    exclude_high_volume_etfs: bool = True,
):
    """Scan tickers for unusual options activity."""
    # Resolve ticker list
    # Default (no tickers, no watchlist) - full all_optionable universe
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif watchlist:
        ticker_list = get_watchlist(watchlist)
    else:
        ticker_list = get_watchlist("all_optionable")

    # Deduplicate while preserving order
    seen: set[str] = set()
    ticker_list = [t for t in ticker_list if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers resolved")
    # No hard cap - the all_optionable universe is ~500 tickers
    if len(ticker_list) > 600:
        raise HTTPException(status_code=400, detail="Max 600 tickers per scan")

    top_n = max(1, min(top_n, 200))
    # Scale concurrency with universe size.
    # yfinance caps connections regardless of thread count, so raising workers
    # beyond ~8 only adds thread overhead without increasing throughput.
    auto_concurrent = 4 if len(ticker_list) < 20 else (6 if len(ticker_list) < 100 else 8)

    logger.info(
        "Unusual options scan: %d tickers, max_dte=%d, vol_oi=%.1fx, workers=%d",
        len(ticker_list),
        max_dte,
        vol_oi_threshold,
        auto_concurrent,
    )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: scan_unusual_options(
                tickers=ticker_list,
                min_volume=min_volume,
                min_oi=min_oi,
                vol_oi_threshold=vol_oi_threshold,
                iv_spike_z=iv_spike_z,
                otm_pct=otm_pct,
                max_dte=max_dte,
                max_concurrent=auto_concurrent,
                top_n=top_n,
                min_premium=min_premium,
                new_positions_only=new_positions_only,
                min_sweep_premium=min_sweep_premium,
                min_block_premium=min_block_premium,
                exclude_bid_side=exclude_bid_side,
                exclude_high_volume_etfs=exclude_high_volume_etfs,
            ),
        )
        return result
    except Exception as e:
        logger.exception("Unusual options scan failed")
        raise HTTPException(status_code=500, detail=f"unusual options scan failed: {e}")  # noqa: B904


@router.post("/api/options/unusual/cancel")
@limiter.limit("30/minute")
async def api_unusual_options_cancel(request: Request):
    """Cancel any in-flight unusual-options scan (remaining tickers are skipped).

    The scan returns early with whatever hits it collected so far. Useful when
    yfinance starts rate-limiting and you want the workers to stop hammering it.
    """
    cancel_unusual_scan()
    logger.info("Unusual options scan: cancel requested")
    return {"status": "cancel_requested"}


@router.get("/api/options/hot")
@limiter.limit("3/minute")
async def api_options_hot(
    request: Request,
    min_vol_ratio: float = 2.0,
    max_dte: int = 60,
    min_volume: int = 50,
    min_oi: int = 25,
    vol_oi_threshold: float = 2.5,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    top_n: int = 60,
    min_premium: float = 50_000.0,
    max_vol_spikes: int = 40,
    lookback_days: int = 20,
):
    """Hot scanner: volume spikes, then unusual options on those tickers."""
    universe = get_watchlist("all_optionable")

    try:
        loop = asyncio.get_running_loop()

        # Stage 1: volume spike scan
        logger.info(
            "[hot_scan] Stage-1 volume scan: %d tickers, min_ratio=%.1fx",
            len(universe),
            min_vol_ratio,
        )
        vol_spikes: list[dict] = await loop.run_in_executor(
            None,
            lambda: scan_volume_spikes(
                tickers=universe,
                min_vol_ratio=min_vol_ratio,
                top_n=max_vol_spikes,
                lookback_days=lookback_days,
            ),
        )

        # Stage 2: unusual options on hot tickers
        hot_tickers = [s["ticker"] for s in vol_spikes]

        if not hot_tickers:
            logger.info("[hot_scan] No volume spikes found above %.1fx", min_vol_ratio)
            return {
                "hits": [],
                "volume_spikes": [],
                "summary": {
                    "universe_scanned": len(universe),
                    "vol_spike_found": 0,
                    "tickers_scanned": 0,
                    "tickers_with_hits": 0,
                    "total_hits": 0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "mixed_count": 0,
                },
                "scanned_at": datetime.now(timezone.utc).isoformat(),
            }

        logger.info(
            "[hot_scan] Stage-2 options scan: %d high-volume tickers",
            len(hot_tickers),
        )
        # Use a smaller concurrency cap - the tickers already have unusual
        # activity, so yfinance calls are fewer and we want to avoid hammering.
        auto_concurrent = min(6, len(hot_tickers))

        result: dict = await loop.run_in_executor(
            None,
            lambda: scan_unusual_options(
                tickers=hot_tickers,
                min_volume=min_volume,
                min_oi=min_oi,
                vol_oi_threshold=vol_oi_threshold,
                iv_spike_z=iv_spike_z,
                otm_pct=otm_pct,
                max_dte=max_dte,
                max_concurrent=auto_concurrent,
                top_n=top_n,
                min_premium=min_premium,
            ),
        )

        # Enrich each options hit with vol-spike metadata
        vol_spike_map = {s["ticker"]: s for s in vol_spikes}
        for hit in result.get("hits", []):
            tkr = hit.get("ticker", "")
            if tkr in vol_spike_map:
                hit["vol_spike"] = vol_spike_map[tkr]

        # Merge summary fields
        summary = result.get("summary", {})
        summary["universe_scanned"] = len(universe)
        summary["vol_spike_found"] = len(vol_spikes)
        result["summary"] = summary
        result["volume_spikes"] = vol_spikes

        logger.info(
            "[hot_scan] complete - %d vol spikes -> %d options hits",
            len(vol_spikes),
            len(result.get("hits", [])),
        )
        return result

    except Exception as exc:
        logger.exception("[hot_scan] failed")
        raise HTTPException(status_code=500, detail=f"hot options scan failed: {exc}")  # noqa: B904


@router.get("/api/options/gex")
@limiter.limit("20/minute")
async def api_options_gex(request: Request, ticker: str):
    """Gamma-exposure profile, max pain, and call/put walls for one ticker."""
    t = (ticker or "").upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")
    try:
        loop = asyncio.get_running_loop()
        flow = await loop.run_in_executor(None, fetch_options_flow, t)
        return flow.to_dict()
    except Exception as e:
        logger.exception("GEX fetch failed")
        raise HTTPException(status_code=500, detail=f"gex fetch failed: {e}")  # noqa: B904


@router.get("/api/options/expiries")
@limiter.limit("30/minute")
async def api_options_expiries(request: Request, ticker: str):
    """Listed option expiries for a ticker (for the contract picker)."""
    t = (ticker or "").upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="invalid ticker")
    try:
        loop = asyncio.get_running_loop()
        expiries = await loop.run_in_executor(None, list_expiries, t)
        return {"ticker": t, "expiries": expiries}
    except Exception as e:
        logger.exception("expiries fetch failed")
        raise HTTPException(status_code=500, detail=f"expiries fetch failed: {e}")  # noqa: B904


@router.get("/api/options/chain")
@limiter.limit("30/minute")
async def api_options_chain(request: Request, ticker: str, expiry: str):
    """Full chain (calls + puts) for one expiry - strikes, quotes, vol, OI."""
    t = (ticker or "").upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="invalid ticker")
    if not _EXPIRY_RE.match(expiry or ""):
        raise HTTPException(status_code=400, detail="expiry must be YYYY-MM-DD")
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, chain_for_expiry, t, expiry)
    except Exception as e:
        logger.exception("chain fetch failed")
        raise HTTPException(status_code=500, detail=f"chain fetch failed: {e}")  # noqa: B904


@router.get("/api/options/contract")
@limiter.limit("30/minute")
async def api_options_contract(
    request: Request,
    ticker: str,
    expiry: str,
    strike: float,
    option_type: str,
    range: str = "7d",  # noqa: A002 - matches query param name
    bucket: str = "30m",
):
    """Premium history + live snapshot for one contract.

    Bars carry a buy/sell volume split: estimated (tick rule) for backfilled
    buckets, observed ask/bid-side classification for buckets covered by the
    live poller (``source: "live"``).
    """
    t, exp, k, ot = _validate_contract(ticker, expiry, strike, option_type)
    if range not in VALID_RANGES:
        raise HTTPException(status_code=400, detail=f"range must be one of {sorted(VALID_RANGES)}")
    if bucket not in VALID_BUCKETS:
        raise HTTPException(status_code=400, detail=f"bucket must be one of {sorted(VALID_BUCKETS)}")
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: get_contract_view(t, exp, k, ot, range, bucket))
    except Exception as e:
        logger.exception("contract view failed")
        raise HTTPException(status_code=500, detail=f"contract view failed: {e}")  # noqa: B904


def _years_to_expiry(expiry: str) -> float:
    """Calendar time to expiry in years (>= ~1 trading day for 0-DTE)."""
    exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    days = (exp - today).days
    if days <= 0:
        return 1.0 / 365.0  # treat 0/expired-DTE as ~1 day to avoid div-by-zero
    return days / 365.0


@router.get("/api/options/pop_scan")
@limiter.limit("20/minute")
async def api_options_pop_scan(
    request: Request,
    ticker: str,
    expiry: str,
    target: float | None = None,      # MY-VIEW target price (absolute $)
    drift: float | None = None,       # MY-VIEW annualized drift, percent (e.g. 12)
    vol: float | None = None,         # MY-VIEW sigma override, annualized percent
    risk_free: float = 4.3,           # percent
    commission: float = 0.65,         # $ per contract per leg
    verticals: bool = True,
    show_bear: bool = False,          # hide bearish spreads by default
    top_n: int = 5,
):
    """Probability-of-profit / expected-return scanner for one ticker + expiry.

    Scores long calls, long puts and simple verticals under a lognormal model in
    BOTH modes (MARKET: mu=r, sigma=IV; MY-VIEW: sigma=RV-forecast or override,
    mu from drift or target). This is a model ESTIMATOR, not a prediction - every
    row carries the assumptions used.
    """
    t = (ticker or "").upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="invalid ticker")
    if not _EXPIRY_RE.match(expiry or ""):
        raise HTTPException(status_code=400, detail="expiry must be YYYY-MM-DD")
    top_n = max(1, min(int(top_n), 25))

    T = _years_to_expiry(expiry)
    try:
        loop = asyncio.get_running_loop()
        chain = await loop.run_in_executor(None, chain_for_expiry, t, expiry)
        if not chain.get("calls") and not chain.get("puts"):
            raise HTTPException(status_code=404, detail="no chain data for this expiry")

        # Realized-vol forecast (annualized fraction) from the IV/RV module.
        rv_frac = None
        em = await loop.run_in_executor(None, expected_move_for_ticker, t, chain.get("spot"))
        if em and em.get("annual_vol_pct"):
            rv_frac = float(em["annual_vol_pct"]) / 100.0

        result = await loop.run_in_executor(
            None,
            lambda: pop_scan(
                chain,
                T=T,
                spot=chain.get("spot"),
                r=risk_free / 100.0,
                rv_forecast=rv_frac,
                my_sigma=(vol / 100.0) if (vol and vol > 0) else None,
                my_drift=(drift / 100.0) if (drift is not None) else None,
                target_price=target,
                commission=commission,
                include_verticals=verticals,
                include_bear=show_bear,
                top_n=top_n,
            ),
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("pop_scan failed")
        raise HTTPException(status_code=500, detail=f"pop_scan failed: {e}")  # noqa: B904


def _dte_days(expiry: str) -> int:
    exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (exp - datetime.now(timezone.utc).date()).days


def _select_expiries(expiries: list[str], min_dte: int, max_dte: int, n: int) -> list[str]:
    """Pick up to `n` expiries within [min_dte, max_dte], spaced across the range."""
    elig = [(e, _dte_days(e)) for e in expiries if _EXPIRY_RE.match(e or "")]
    elig = [(e, d) for e, d in elig if min_dte <= d <= max_dte]
    elig.sort(key=lambda x: x[1])
    if len(elig) <= n:
        return [e for e, _ in elig]
    # Even spacing across the eligible list (always keep first and last).
    idxs = sorted({round(i * (len(elig) - 1) / (n - 1)) for i in range(n)})
    return [elig[i][0] for i in idxs]


async def _sentiment_score(ticker: str, timeout: float = 6.0) -> float | None:
    """Best-effort aggregate sentiment in [-1, 1]; None if unavailable/slow."""
    try:
        from core.sentiment import get_sentiment

        data = await asyncio.wait_for(get_sentiment(ticker), timeout=timeout)
        return float(data.get("aggregate", {}).get("score"))
    except Exception as e:
        logger.debug("[research] sentiment unavailable for %s: %s", ticker, e)
        return None


@router.get("/api/options/research_scan")
@limiter.limit("8/minute")
async def api_options_research_scan(
    request: Request,
    ticker: str,
    max_expiries: int = 6,
    min_dte: int = 5,
    max_dte: int = 120,
    risk_free: float = 4.3,
    commission: float = 0.65,
    use_sentiment: bool = True,
    top_n: int = 8,
):
    """Forecast-driven scan: estimate a research-based drift, then find which
    options and which expiry give the best expected return under that view.

    The drift blends momentum (Jegadeesh-Titman 1993), short-term reversal
    (Jegadeesh 1990 / Lehmann 1990), the high-volume premium (Gervais-Kaniel-
    Mingelgrin 2001) and sentiment (Tetlock 2007). It is a model ESTIMATOR, not
    a prediction; only single long calls/puts (what's buyable) are scored.
    """
    t = (ticker or "").upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="invalid ticker")
    max_expiries = max(1, min(int(max_expiries), 10))
    top_n = max(1, min(int(top_n), 25))

    loop = asyncio.get_running_loop()
    try:
        # Sentiment (best-effort) feeds the forecast.
        sent = await _sentiment_score(t) if use_sentiment else None

        forecast = await loop.run_in_executor(
            None, lambda: forecast_for_ticker(t, None, sentiment_score=sent))
        if not forecast:
            raise HTTPException(status_code=422, detail="not enough history to forecast this ticker")
        mu = float(forecast["mu_annual"])

        # Realized-vol forecast (shared sigma for MY-VIEW across expiries).
        rv_frac = None
        em = await loop.run_in_executor(None, expected_move_for_ticker, t, forecast.get("spot"))
        if em and em.get("annual_vol_pct"):
            rv_frac = float(em["annual_vol_pct"]) / 100.0

        all_exp = await loop.run_in_executor(None, list_expiries, t)
        chosen = _select_expiries(list(all_exp or []), min_dte, max_dte, max_expiries)
        if not chosen:
            raise HTTPException(status_code=404, detail="no expiries in the requested DTE window")

        sem = asyncio.Semaphore(4)

        async def _scan_one(expiry: str) -> dict | None:
            T = _years_to_expiry(expiry)
            dte = _dte_days(expiry)
            async with sem:
                try:
                    chain = await loop.run_in_executor(None, chain_for_expiry, t, expiry)
                    if not chain.get("calls") and not chain.get("puts"):
                        return None
                    res = await loop.run_in_executor(
                        None,
                        lambda: pop_scan(
                            chain, T=T, spot=chain.get("spot"), r=risk_free / 100.0,
                            rv_forecast=rv_frac, my_drift=mu, commission=commission,
                            include_verticals=False, include_bear=False, top_n=top_n,
                        ),
                    )
                except Exception as e:
                    logger.debug("[research] %s %s scan failed: %s", t, expiry, e)
                    return None
            for row in res.get("all", []):
                row["expiry"] = expiry
                row["dte"] = dte
            best = None
            valid = [r for r in res.get("all", []) if r.get("exp_return_pct") is not None]
            if valid:
                best = max(valid, key=lambda r: r["exp_return_pct"])
            return {"expiry": expiry, "dte": dte, "rows": res.get("all", []), "best": best}

        per_expiry = [r for r in await asyncio.gather(*[_scan_one(e) for e in chosen]) if r]
        if not per_expiry:
            raise HTTPException(status_code=404, detail="no scoreable option chains found")

        combined: list[dict] = []
        for pe in per_expiry:
            combined.extend(pe["rows"])

        with_ret = [r for r in combined if r.get("exp_return_pct") is not None]
        top_by_return = sorted(with_ret, key=lambda r: r["exp_return_pct"], reverse=True)[:top_n]
        top_by_kelly = sorted(
            combined, key=lambda r: r.get("kelly_pct") or 0.0, reverse=True)[:top_n]
        best_expiry = max(
            per_expiry,
            key=lambda pe: (pe["best"]["exp_return_pct"] if pe.get("best") else -1e9),
        )["expiry"] if any(pe.get("best") for pe in per_expiry) else None

        return {
            "ticker": t,
            "spot": forecast.get("spot"),
            "forecast": forecast,
            "sentiment_used": sent,
            "rv_forecast_pct": (None if rv_frac is None else round(rv_frac * 100, 2)),
            "expiries_scanned": [pe["expiry"] for pe in per_expiry],
            "best_expiry": best_expiry,
            "per_expiry": [
                {"expiry": pe["expiry"], "dte": pe["dte"], "best": pe["best"]}
                for pe in per_expiry
            ],
            "all": combined,
            "top_by_expected_return": top_by_return,
            "top_by_kelly": top_by_kelly,
            "scoring_note": (
                "Drift blends momentum, reversal, volume, and sentiment. "
                "Rankings are model estimates under lognormal assumptions."
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("research_scan failed")
        raise HTTPException(status_code=500, detail=f"research_scan failed: {e}")  # noqa: B904


@router.get("/api/options/contract/watched")
async def api_contract_watched():
    """Contracts currently followed by the live buy/sell poller."""
    return {"watched": watched()}


@router.post("/api/options/contract/watch")
@limiter.limit("20/minute")
async def api_contract_watch(
    request: Request, req: ContractWatchRequest, api_key: str | None = Header(None)
):
    """Start live tracking (volume-delta buy/sell classification) for a contract."""
    require_api_key(api_key)
    t, exp, k, ot = _validate_contract(req.ticker, req.expiry, req.strike, req.option_type)
    try:
        info = watch(t, exp, k, ot, poll_seconds=req.poll_seconds or 30)
        return {"status": "ok", "watcher": info}
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))  # noqa: B904


@router.post("/api/options/contract/unwatch")
@limiter.limit("20/minute")
async def api_contract_unwatch(
    request: Request, req: ContractWatchRequest, api_key: str | None = Header(None)
):
    """Stop live tracking for a contract."""
    require_api_key(api_key)
    t, exp, k, ot = _validate_contract(req.ticker, req.expiry, req.strike, req.option_type)
    removed = unwatch(t, exp, k, ot)
    return {"status": "ok" if removed else "not_watched"}
