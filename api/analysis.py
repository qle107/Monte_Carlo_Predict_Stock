"""Analysis orchestration and poll loop."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from config import cfg
from core import _df_to_candles, analyse
from core.fetcher import fetch_candles
from core.hmm_regime import analyse_hmm
from core.htf import _htf_confirmation
from core.trade_setup import trade_setup_from_analysis
from core.zones import detect_zones

from .state import state

logger = logging.getLogger(__name__)


async def _broadcast(data: dict):
    dead = set()
    for ws in state.clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    state.clients.difference_update(dead)


async def _run_analysis() -> dict:
    # Skip if another analysis is already running.
    if state.analysis_lock is not None and state.analysis_lock.locked():
        if state.last_result:
            logger.debug("[server] _run_analysis skipped - already in flight, returning cached result")
            return state.last_result
        # No cached result yet - wait for the in-flight call to finish then return
        async with state.analysis_lock:
            return state.last_result

    lock_ctx = state.analysis_lock if state.analysis_lock is not None else asyncio.Lock()
    async with lock_ctx:
        try:
            loop = asyncio.get_running_loop()

            # Fetch enough bars to cover both display history and MC analysis window.
            display_bars = max(cfg.lookback, cfg.chart_bars)
            df_full = await loop.run_in_executor(
                None, fetch_candles, cfg.ticker, cfg.interval, display_bars, cfg.extended
            )

            # Slice to the analysis window (lookback candles) for MC + indicators.
            df = df_full.tail(cfg.lookback).copy()
            df.attrs = df_full.attrs
            try:
                await _broadcast(
                    {
                        "type": "partial",
                        "ticker": cfg.ticker,
                        "interval": cfg.interval,
                        "current_price": round(float(df_full["close"].iloc[-1]), 4),
                        "candles": _df_to_candles(df_full),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception as _pe:
                logger.debug("partial broadcast failed: %s", _pe)

            _returns_for_hmm = df["close"].pct_change().dropna().values.tolist()
            _gather_tasks = [
                loop.run_in_executor(None, analyse, df, cfg.n_sim, cfg.n_forward, cfg.mc_model),
                loop.run_in_executor(None, detect_zones, df),
                _htf_confirmation(cfg.ticker, cfg.interval, cfg.extended, loop),
            ]
            if cfg.hmm_enabled:
                _gather_tasks.insert(2, loop.run_in_executor(None, analyse_hmm, _returns_for_hmm))

            raw = await asyncio.gather(*_gather_tasks, return_exceptions=True)

            if cfg.hmm_enabled:
                result_raw, zone_raw, hmm_raw, htf_data = raw
            else:
                result_raw, zone_raw, htf_data = raw
                hmm_raw = None

            # Unpack analyse result (must succeed)
            if isinstance(result_raw, BaseException):
                raise result_raw
            result = result_raw

            # Unpack zone result
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

            # Volume profile is already in analyse() result.
            _vp_inner = result.get("volume_profile") if isinstance(result, dict) else None
            if _vp_inner is None:
                vp_data = None
            elif isinstance(_vp_inner, dict):
                vp_data = _vp_inner
            elif hasattr(_vp_inner, "to_dict"):
                vp_data = _vp_inner.to_dict()
            else:
                vp_data = None

            # Unpack HMM result
            if hmm_raw is None:
                hmm_data = None
            elif isinstance(hmm_raw, BaseException):
                logger.debug("hmm in _run_analysis failed: %s", hmm_raw)
                hmm_data = None
            else:
                hmm_data = hmm_raw.to_dict() if hmm_raw else None

            # Unpack HTF confirmation
            if isinstance(htf_data, BaseException):
                logger.debug("HTF confirmation failed: %s", htf_data)
                htf_data = {"available": False, "reason": str(htf_data)}

            # Override candles with full display history
            if len(df_full) > len(df):
                result["candles"] = _df_to_candles(df_full)

            # Trade setup (runs after gather)
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

            # HTF alignment (needs both analyse + HTF results)
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
            state.last_result = result

            # Persist
            if state.store is not None:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: state.store.record(result))
                except Exception as e:
                    logger.warning("store.record failed: %s", e)

            sig = result["signal"]
            reg = result.get("regime", {}) or {}
            warn_str = f"  ⚠ {result['warnings'][0]}" if result.get("warnings") else ""
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
        await asyncio.sleep(1)  # let fetcher cache warm after config change
        while True:
            result = await _run_analysis()
            if "error" not in result:
                await _broadcast(result)
            await asyncio.sleep(cfg.poll_seconds)
    except asyncio.CancelledError:
        logger.info("Poll loop cancelled")
        raise
