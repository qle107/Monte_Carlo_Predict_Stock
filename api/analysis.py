"""Analysis orchestration and poll loop."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import partial

from config import cfg
from core import _df_to_candles, analyse
from core.analysis.conformal import warm_start_from_history
from core.analysis.expected_move import expected_move_for_ticker
from core.analysis.hmm_regime import analyse_hmm
from core.analysis.htf import htf_confirmation
from core.analysis.trade_setup import trade_setup_from_analysis
from core.analysis.zones import detect_zones
from core.data.fetcher import fetch_candles

from . import state
from .websockets import broadcast_json

logger = logging.getLogger(__name__)


async def _fetch_and_broadcast_partial(loop) -> tuple:
    """Fetch candles and push a quick 'partial' frame so the chart updates early.

    Returns (df_full, df) where df is the lookback window used for analysis.
    """
    display_bars = max(cfg.lookback, cfg.chart_bars)
    df_full = await loop.run_in_executor(
        None, fetch_candles, cfg.ticker, cfg.interval, display_bars, cfg.extended
    )

    df = df_full.tail(cfg.lookback).copy()
    df.attrs = df_full.attrs
    try:
        await broadcast_json(
            state.clients,
            {
                "type": "partial",
                "ticker": cfg.ticker,
                "interval": cfg.interval,
                "current_price": round(float(df_full["close"].iloc[-1]), 4),
                "candles": _df_to_candles(df_full),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as _pe:
        logger.debug("partial broadcast failed: %s", _pe)
    return df_full, df


async def _calibrate_bands(loop, cal, df_full, spot_now: float) -> float:
    """Warm-start, settle, and return the conformal band alpha (default 0.20)."""
    band_alpha = 0.20
    if cal is None:
        return band_alpha
    try:
        cov0 = await loop.run_in_executor(None, cal.coverage, cfg.ticker, cfg.interval, cfg.n_forward)
        if not cov0.get("n_settled"):
            n_seed = await loop.run_in_executor(
                None,
                partial(
                    warm_start_from_history,
                    cal,
                    df_full,
                    cfg.ticker,
                    cfg.interval,
                    cfg.n_forward,
                    cfg.mc_model,
                ),
            )
            if n_seed:
                logger.info(
                    "conformal warm-start: scored %d historical forecasts for %s %s",
                    n_seed,
                    cfg.ticker,
                    cfg.interval,
                )
        await loop.run_in_executor(None, cal.settle, cfg.ticker, cfg.interval, spot_now)
        band_alpha = await loop.run_in_executor(
            None, cal.target_alpha, cfg.ticker, cfg.interval, cfg.n_forward
        )
    except Exception as e:
        logger.debug("conformal settle/alpha failed: %s", e)
    return band_alpha


async def _gather_analyses(loop, df, band_alpha: float, returns_for_hmm: list, spot_now: float) -> tuple:
    """Run MC analysis, zones, HTF, expected move (and HMM if enabled) concurrently.

    Returns (result, zones_data, hmm_data, htf_data, expected_move).
    Raises if the main MC analysis failed.
    """
    gather_tasks = [
        loop.run_in_executor(
            None,
            partial(analyse, df, cfg.n_sim, cfg.n_forward, cfg.mc_model, band_alpha=band_alpha),
        ),
        loop.run_in_executor(None, detect_zones, df),
        htf_confirmation(cfg.ticker, cfg.interval, cfg.extended, loop),
        loop.run_in_executor(None, expected_move_for_ticker, cfg.ticker, spot_now),
    ]
    if cfg.hmm_enabled:
        gather_tasks.insert(2, loop.run_in_executor(None, analyse_hmm, returns_for_hmm))

    raw = await asyncio.gather(*gather_tasks, return_exceptions=True)

    if cfg.hmm_enabled:
        result_raw, zone_raw, hmm_raw, htf_data, em_raw = raw
    else:
        result_raw, zone_raw, htf_data, em_raw = raw
        hmm_raw = None

    if isinstance(em_raw, BaseException):
        logger.debug("expected_move failed: %s", em_raw)
        expected_move = None
    else:
        expected_move = em_raw

    if isinstance(result_raw, BaseException):
        raise result_raw
    result = result_raw

    if isinstance(zone_raw, BaseException):
        logger.warning("zone detect failed: %s", zone_raw)
        zones_data = {
            "demand_zones": [],
            "supply_zones": [],
            "nearest_demand": None,
            "nearest_supply": None,
            "price_context": "unknown",
            "atr": 0.0,
        }
    else:
        zones_data = zone_raw.to_dict()

    if hmm_raw is None:
        hmm_data = None
    elif isinstance(hmm_raw, BaseException):
        logger.debug("hmm in _run_analysis failed: %s", hmm_raw)
        hmm_data = None
    else:
        hmm_data = hmm_raw.to_dict() if hmm_raw else None

    if isinstance(htf_data, BaseException):
        logger.debug("HTF confirmation failed: %s", htf_data)
        htf_data = {"available": False, "reason": str(htf_data)}

    return result, zones_data, hmm_data, htf_data, expected_move


def _assemble_result(result, df, df_full, zones_data, hmm_data, htf_data, expected_move) -> dict:
    """Attach trade setup, HTF alignment, and config metadata to the MC result."""
    vp_data = result.get("volume_profile")

    if len(df_full) > len(df):
        result["candles"] = _df_to_candles(df_full)

    mc_paths_full = result.pop("_mc_paths_full", None)
    try:
        trade_setup = trade_setup_from_analysis(
            cfg.ticker,
            result,
            interval=cfg.interval,
            df=df,
            mc_paths_full=mc_paths_full,
        )
    except Exception as e:
        logger.warning("trade_setup failed: %s", e)
        trade_setup = {"valid": False, "side": "none", "reason": str(e)}

    if htf_data.get("available"):
        base_comp = float(result.get("signal", {}).get("composite", 0.0))
        htf_comp = float(htf_data.get("composite", 0.0))
        if base_comp > 0.05 and htf_comp > 0.05:
            htf_data["alignment"] = "confirm_bullish"
        elif base_comp < -0.05 and htf_comp < -0.05:
            htf_data["alignment"] = "confirm_bearish"
        elif base_comp * htf_comp < 0:
            htf_data["alignment"] = "conflict"
        else:
            htf_data["alignment"] = "neutral"

    result.update(
        {
            "ticker": cfg.ticker,
            "interval": cfg.interval,
            "extended": cfg.extended,
            "mc_model": cfg.mc_model,
            "trade_setup": trade_setup,
            "expected_move": expected_move,
            "zones": zones_data,
            "volume_profile": vp_data,
            "hmm": hmm_data,
            "htf": htf_data,
            "config": {
                "n_sim": cfg.n_sim,
                "n_forward": cfg.n_forward,
                "lookback": cfg.lookback,
                "chart_bars": cfg.chart_bars,
                "poll_seconds": cfg.poll_seconds,
                "extended": cfg.extended,
                "mc_model": cfg.mc_model,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return result


async def _record_calibration(loop, cal, result) -> None:
    """Register this forecast's band with the calibrator and attach coverage stats."""
    if cal is None:
        return
    try:
        mc_d = result.get("mc", {})
        lo, hi = float(mc_d.get("p10_price", 0)), float(mc_d.get("p90_price", 0))
        await loop.run_in_executor(
            None,
            partial(
                cal.observe,
                cfg.ticker,
                cfg.interval,
                cfg.n_forward,
                result["updated_at"],
                result["current_price"],
                lo,
                hi,
            ),
        )
        result["band_calibration"] = await loop.run_in_executor(
            None, cal.coverage, cfg.ticker, cfg.interval, cfg.n_forward
        )
    except Exception as e:
        logger.debug("conformal observe failed: %s", e)


async def _run_analysis() -> dict:
    if state.analysis_lock is not None and state.analysis_lock.locked():
        if state.last_result:
            logger.debug("[server] _run_analysis skipped - already in flight, returning cached result")
            return state.last_result
        async with state.analysis_lock:
            return state.last_result

    lock_ctx = state.analysis_lock if state.analysis_lock is not None else asyncio.Lock()
    async with lock_ctx:
        try:
            loop = asyncio.get_running_loop()

            df_full, df = await _fetch_and_broadcast_partial(loop)

            returns_for_hmm = df["close"].pct_change().dropna().values.tolist()
            spot_now = float(df_full["close"].iloc[-1])

            cal = state.calibrator
            band_alpha = await _calibrate_bands(loop, cal, df_full, spot_now)

            result, zones_data, hmm_data, htf_data, expected_move = await _gather_analyses(
                loop, df, band_alpha, returns_for_hmm, spot_now
            )

            result = _assemble_result(result, df, df_full, zones_data, hmm_data, htf_data, expected_move)

            await _record_calibration(loop, cal, result)

            state.last_result = result

            if state.store is not None:
                try:
                    await loop.run_in_executor(None, lambda: state.store.record(result))
                except Exception as e:
                    logger.warning("store.record failed: %s", e)

            sig = result["signal"]
            reg = result.get("regime", {}) or {}
            warn_str = f"  Warning: {result['warnings'][0]}" if result.get("warnings") else ""
            logger.info(
                "%s %s [%s]  price=%.2f  regime=%s  pot up/dn/flat=%.0f/%.0f/%.0f  signal=%s (conf=%.0f%%)%s",
                cfg.ticker,
                cfg.interval,
                cfg.mc_model,
                result["current_price"],
                reg.get("regime", "?"),
                reg.get("potential_up", 0),
                reg.get("potential_down", 0),
                reg.get("potential_flat", 0),
                sig["label"],
                sig["confidence"] * 100,
                warn_str,
            )
            return result
        except Exception as e:
            logger.exception("Analysis failed")
            return {"error": str(e)}


async def _poll_loop():
    logger.info("Poll loop started: %s %s every %ds", cfg.ticker, cfg.interval, cfg.poll_seconds)
    try:
        await asyncio.sleep(1)
        while True:
            result = await _run_analysis()
            if "error" not in result:
                await broadcast_json(state.clients, result)
            await asyncio.sleep(cfg.poll_seconds)
    except asyncio.CancelledError:
        logger.info("Poll loop cancelled")
        raise
