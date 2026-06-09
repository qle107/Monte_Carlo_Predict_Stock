"""Signal analysis, config, health, backtest, and market structure."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request

from config import VALID_INTERVALS, VALID_MC_MODELS, cfg, cfg_lock
from core.analysis.backtest import walk_forward
from core.analysis.market_structure import analyse_market_structure
from core.data.fetcher import fetch_candles

from .. import state
from ..analysis import _poll_loop, _run_analysis, broadcast_analysis
from ..deps import limiter, require_api_key
from ..models import BacktestRequest, ConfigUpdate
logger = logging.getLogger(__name__)

router = APIRouter(tags=["signal"])


@router.get("/api/signal")
@limiter.limit("30/minute")
async def get_signal(request: Request):
    """Trigger fresh analysis with current config and return result."""
    result = await _run_analysis()
    if "error" not in result:
        await broadcast_analysis(state.clients, result, full_candles=True)
    return result


@router.get("/api/config")
async def get_config():
    return {
        **cfg.to_dict(),
        "valid_intervals": VALID_INTERVALS,
        "valid_mc_models": VALID_MC_MODELS,
    }


@router.post("/api/config")
@limiter.limit("10/minute")
async def update_config(request: Request, update: ConfigUpdate, api_key: str | None = Header(None)):
    """Apply config changes; restart poll loop; return fresh analysis."""
    require_api_key(api_key)

    changed: list[str] = []
    payload = update.model_dump(exclude_none=True)
    with cfg_lock:
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                changed.append(f"{key}={value}")
    if any(k in payload for k in ("ticker", "interval", "chart_bars", "lookback")):
        state.needs_full_candles = True

    if not changed:
        return {"status": "noop", "changed": [], "config": await get_config()}

    logger.info("Config updated: %s", ", ".join(changed))

    # Restart poll loop with new settings
    if state.poll_task and not state.poll_task.done():
        state.poll_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await state.poll_task

    # Run analysis before restarting poll loop so fetcher cache is warm.
    result = await _run_analysis()
    if "error" not in result:
        await broadcast_analysis(state.clients, result, full_candles=True)

    # Restart poll loop after analysis completes.
    state.poll_task = asyncio.create_task(_poll_loop())

    return {
        "status": "ok",
        "changed": changed,
        "config": await get_config(),
        "result": result if "error" not in result else None,
    }


@router.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ticker": cfg.ticker,
        "interval": cfg.interval,
        "mc_model": cfg.mc_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/backtest")
@limiter.limit("10/minute")
async def api_backtest(request: Request, req: BacktestRequest, api_key: str | None = Header(None)):
    """Walk-forward backtest."""
    require_api_key(api_key)
    ticker = (req.ticker or cfg.ticker).upper().strip()
    interval = req.interval or cfg.interval
    n_forward = req.n_forward or cfg.n_forward
    n_sim = req.n_sim or min(cfg.n_sim, 1000)  # cap for backtest speed
    mc_model = req.mc_model or cfg.mc_model
    history_bars = req.history_bars or 200

    try:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None, fetch_candles, ticker, interval, history_bars + 60, cfg.extended
        )
        report = await loop.run_in_executor(
            None,
            walk_forward,
            df,
            n_forward,
            n_sim,
            mc_model,
            50,
        )
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=400, detail=f"backtest failed: {e}")  # noqa: B904

    return {
        "ticker": ticker,
        "interval": interval,
        "mc_model": mc_model,
        "n_forward": n_forward,
        **report,
    }


@router.get("/api/market-structure")
@limiter.limit("10/minute")
async def api_market_structure(request: Request, ticker: str | None = None):
    """Volume profile, options flow, Hawkes, and HMM analysis."""
    symbol = (ticker or cfg.ticker).upper().strip()
    loop = asyncio.get_running_loop()

    try:
        # Wrap entire analysis with timeout to prevent hanging
        result = await asyncio.wait_for(
            analyse_market_structure(symbol, loop),
            timeout=40.0,
        )
        return result

    except asyncio.TimeoutError:
        logger.error("market-structure timeout after 40s for %s", symbol)
        raise HTTPException(status_code=504, detail="Market structure analysis timed out (>40s)")  # noqa: B904
    except Exception as exc:
        logger.exception("market-structure failed for %s", symbol)
        raise HTTPException(status_code=500, detail=f"market structure failed: {exc}")  # noqa: B904
