"""Volume profile, options flow, HMM, and Hawkes analysis."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from config import cfg

from .fetcher import fetch_candles
from .hawkes import analyse_hawkes
from .hmm_regime import analyse_hmm, blend_zone_probability
from .options_flow import fetch_options_flow
from .volume_profile import compute_volume_profile
from .zones import detect_zones

logger = logging.getLogger(__name__)


async def analyse_market_structure(symbol: str, loop) -> dict:
    """Run volume profile, options flow, HMM, and Hawkes in parallel."""
    HMM_MIN_BARS = 40  # matches core.hmm_regime.MIN_BARS
    HAWKES_MIN_BARS = 20  # matches core.hawkes.analyse_hawkes
    try:
        logger.debug("[ms] data-fetch start  symbol=%s interval=%s", symbol, cfg.interval)
        df = await loop.run_in_executor(
            None,
            fetch_candles,
            symbol,
            cfg.interval,
            max(cfg.lookback, cfg.chart_bars),
            cfg.extended,
        )
        n_bars = len(df)
        spot = float(df["close"].iloc[-1]) if not df.empty else None
        logger.debug("[ms] data-fetch done   bars=%d spot=%s", n_bars, spot)

        log_returns = df["close"].pct_change().dropna().values.tolist()
        n_returns = len(log_returns)
        logger.debug("[ms] preprocess        n_returns=%d", n_returns)

        _ms_tasks = [
            loop.run_in_executor(None, compute_volume_profile, df),
            loop.run_in_executor(None, detect_zones, df),
            loop.run_in_executor(None, fetch_options_flow, symbol, spot),
            loop.run_in_executor(None, analyse_hmm, log_returns),
        ]
        vp_raw, zone_raw, of_raw, hmm_raw = await asyncio.gather(
            *_ms_tasks,
            return_exceptions=True,
        )
        logger.debug(
            "[ms] model-fit done   vp=%s zones=%s of=%s hmm=%s",
            type(vp_raw).__name__,
            type(zone_raw).__name__,
            type(of_raw).__name__,
            type(hmm_raw).__name__,
        )

        if isinstance(vp_raw, BaseException):
            vp_dict = {"state": "error", "error": str(vp_raw)[:120]}
        elif vp_raw is None:
            vp_dict = {"state": "error", "error": "vp_failed"}
        else:
            vp_dict = vp_raw.to_dict()
            vp_dict.setdefault("state", "ok")

        if isinstance(zone_raw, BaseException):
            logger.warning("[ms] zone detect: %s", zone_raw)
            zones_data = {
                "state": "error",
                "error": str(zone_raw)[:120],
                "demand_zones": [],
                "supply_zones": [],
            }
            zone_list = []
        else:
            zones_data = zone_raw.to_dict()
            zone_list = [
                {"level": z.level, "zone_type": "demand", "strength": z.strength}
                for z in zone_raw.demand_zones
            ] + [
                {"level": z.level, "zone_type": "supply", "strength": z.strength}
                for z in zone_raw.supply_zones
            ]
            if not zone_list:
                zones_data["state"] = "no_zones"
                # Zone detection needs at least zone_pivot_window*2+1 bars to
                # find any swing pivots. Surface that to the user.
                zones_data["min_bars_required"] = int(cfg.zone_pivot_window) * 2 + 1
                zones_data["bars_available"] = n_bars
            else:
                zones_data.setdefault("state", "ok")

        if isinstance(of_raw, BaseException):
            of_dict = {"state": "error", "error": str(of_raw)[:120]}
        else:
            of_dict = of_raw.to_dict()
            of_dict.setdefault("state", "error" if of_dict.get("error") else "ok")

        if isinstance(hmm_raw, BaseException):
            logger.warning("[ms] HMM raised: %s", hmm_raw)
            hmm_result = None
            hmm_dict = {"state": "error", "error": str(hmm_raw)[:120]}
        else:
            hmm_result = hmm_raw
            hmm_dict = hmm_raw.to_dict()
            # Surface the insufficient-data path with a clear contract for the UI.
            if hmm_raw.error == "too_few_bars" or hmm_raw.fit_method == "none":
                hmm_dict["state"] = "insufficient_data"
                hmm_dict["min_bars_required"] = HMM_MIN_BARS
                hmm_dict["bars_available"] = n_returns
                hmm_dict.setdefault(
                    "error_reason",
                    f"HMM needs at least {HMM_MIN_BARS} return bars; only {n_returns} available. "
                    f"Switch to a longer timeframe or wait for more candles.",
                )
            else:
                hmm_dict["state"] = "ok"

        if zone_list and n_returns >= HAWKES_MIN_BARS:
            try:
                hawkes_result = await loop.run_in_executor(
                    None,
                    analyse_hawkes,
                    log_returns,
                    zone_list,
                )
                hawkes_dict = hawkes_result.to_dict()
                hawkes_dict.setdefault("state", "ok")
            except Exception as exc:
                logger.warning("[ms] Hawkes raised: %s", exc)
                hawkes_result = None
                hawkes_dict = {"state": "error", "error": str(exc)[:120]}
        else:
            hawkes_result = None
            hawkes_dict = {
                "state": "insufficient_data" if n_returns < HAWKES_MIN_BARS else "no_zones",
                "min_bars_required": HAWKES_MIN_BARS,
                "bars_available": n_returns,
                "error_reason": (
                    f"Hawkes process needs at least {HAWKES_MIN_BARS} return bars; "
                    f"only {n_returns} available."
                    if n_returns < HAWKES_MIN_BARS
                    else "No demand/supply zones detected - Hawkes excitation requires them."
                ),
            }

        # Now also produces rows when HMM is missing - falls back to a 40/30/30
        # base so the table never goes empty just because Baum-Welch under-ran.
        blended_zones = []
        for z in zone_list:
            hk_probs = None
            if hawkes_result is not None:
                for hr in hawkes_result.zone_reactions:
                    if abs(hr.level - z["level"]) < 0.01:
                        hk_probs = {
                            "bounce": hr.bounce_prob,
                            "break": hr.break_prob,
                            "consolidate": hr.consolidate_prob,
                        }
                        break
            if hmm_result is not None:
                blended = blend_zone_probability(
                    hmm=hmm_result,
                    hawkes_probs=hk_probs,
                    zone_strength=z.get("strength", 0.5),
                )
            elif hk_probs is not None:
                blended = hk_probs
            else:
                # Neither HMM nor Hawkes - fall back to neutral priors so the
                # zone row at least renders something. Marked so the UI can
                # caveat the row.
                blended = {"bounce": 0.40, "break": 0.30, "consolidate": 0.30}
            blended_zones.append(
                {
                    "level": round(z["level"], 4),
                    "zone_type": z["zone_type"],
                    "strength": round(z.get("strength", 0.5), 3),
                    "bounce_prob": round(blended["bounce"], 4),
                    "break_prob": round(blended["break"], 4),
                    "consolidate_prob": round(blended["consolidate"], 4),
                    "blend_source": (
                        "hmm+hawkes"
                        if hmm_result is not None and hk_probs is not None
                        else "hmm"
                        if hmm_result is not None
                        else "hawkes"
                        if hk_probs is not None
                        else "fallback"
                    ),
                }
            )

        logger.debug(
            "[ms] render-ready     zones=%d hmm_state=%s hawkes_state=%s",
            len(blended_zones),
            hmm_dict.get("state"),
            hawkes_dict.get("state"),
        )

        return {
            "ticker": symbol,
            "interval": cfg.interval,
            "current_price": round(spot or 0.0, 4),
            "bars_available": n_bars,
            "volume_profile": vp_dict,
            "options_flow": of_dict,
            "hawkes": hawkes_dict,
            "hmm": hmm_dict,
            "blended_zones": blended_zones,
            "zones": zones_data,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except asyncio.TimeoutError:
        raise
    except Exception:
        logger.exception("analyse_market_structure failed for %s", symbol)
        raise
