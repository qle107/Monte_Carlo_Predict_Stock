"""Breakout and zone scanners + watchlist listing."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from config import cfg
from core.analysis.trade_setup import trade_setup_from_scan
from core.scanners.scanner import WATCHLISTS, get_watchlist, scan_tickers
from core.scanners.zone_scanner import zone_scan_tickers

from ..deps import limiter, require_api_key
from ..models import ScanRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scanner"])


@router.get("/api/scan/watchlists")
async def api_scan_watchlists():
    """Return available watchlist names and their tickers."""
    return dict(WATCHLISTS.items())


@router.post("/api/scan")
@limiter.limit("5/minute")
async def api_scan(request: Request, req: ScanRequest, api_key: str | None = Header(None)):
    """Scan tickers for breakouts and breakdowns."""
    require_api_key(api_key)
    if req.tickers:
        tickers = [t.upper().strip() for t in req.tickers if t.strip()]
    else:
        tickers = get_watchlist(req.watchlist or "sp500_large")

    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(tickers) > 200:
        raise HTTPException(status_code=400, detail="Max 200 tickers per scan")

    interval = req.interval or "1d"
    lookback = req.lookback or 60
    extended = req.extended if req.extended is not None else cfg.extended
    concurrent = min(req.max_concurrent or 8, 20)

    try:
        report = await scan_tickers(
            tickers=tickers,
            interval=interval,
            lookback=lookback,
            extended=extended,
            max_concurrent=concurrent,
            min_score_abs=req.min_score_abs or 0.0,
        )
    except Exception as e:
        logger.exception("Scan failed")
        raise HTTPException(status_code=500, detail=f"scan failed: {e}")  # noqa: B904

    # Attach trade setup to every scan result
    def _enrich(items: list) -> list:
        enriched = []
        for r in items:
            try:
                r["trade_setup"] = trade_setup_from_scan(r)
            except Exception as ex:
                r["trade_setup"] = {"valid": False, "side": "none", "reason": str(ex)}
            enriched.append(r)
        return enriched

    report["breakouts"] = _enrich(report.get("breakouts", []))
    report["breakdowns"] = _enrich(report.get("breakdowns", []))
    report["neutral"] = _enrich(report.get("neutral", []))
    report["all"] = _enrich(report.get("all", []))

    return report


@router.post("/api/zone-scan")
@limiter.limit("5/minute")
async def api_zone_scan(request: Request, req: ScanRequest, api_key: str | None = Header(None)):
    """Scan for demand/supply zone setups."""
    require_api_key(api_key)
    if req.tickers:
        tickers = [t.upper().strip() for t in req.tickers if t.strip()]
    else:
        tickers = get_watchlist(req.watchlist or "sp500_large")

    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(tickers) > 200:
        raise HTTPException(status_code=400, detail="Max 200 tickers per scan")

    interval = req.interval or "1d"
    lookback = req.lookback or 120  # need more bars for EMA200
    extended = req.extended if req.extended is not None else cfg.extended
    concurrent = min(req.max_concurrent or 8, 20)
    min_score = float(req.min_score_abs or 0.0)

    try:
        report = await zone_scan_tickers(
            tickers=tickers,
            interval=interval,
            lookback=lookback,
            extended=extended,
            max_concurrent=concurrent,
            min_score=min_score,
        )
    except Exception as e:
        logger.exception("Zone scan failed")
        raise HTTPException(status_code=500, detail=f"zone scan failed: {e}")  # noqa: B904

    return report
