"""Composite signal scoring with regime-aware weights."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from config import cfg

from .indicators import Indicators, _safe
from .regime import Regime

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    composite: float
    confidence: float
    drift_bias: float
    base_drift: float
    signal_adj: float
    vol_adj: float
    label: str
    reasoning: str
    gap_warning: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)


def _score_rsi(rsi: float) -> float:
    """Mean-reversion read on RSI. Uses cfg thresholds. Capped ±0.6."""
    ob = cfg.rsi_overbought  # default 82
    os_ = cfg.rsi_oversold  # default 18

    if rsi < os_:
        return 0.60
    elif rsi < os_ + 10:
        return 0.40
    elif rsi < os_ + 20:
        return 0.15
    elif rsi < ob - 20:
        return 0.00
    elif rsi < ob - 10:
        return -0.15
    elif rsi < ob:
        return -0.40
    else:
        return -0.60


def _score_slope(slope_pct: float) -> float:
    return float(np.clip(slope_pct * 15, -0.8, 0.8))


def _score_momentum(mom_pct: float) -> float:
    return float(np.clip(mom_pct * 5, -0.6, 0.6))


def _score_ema(cross: str) -> float:
    return {"bullish": 0.4, "bearish": -0.4, "neutral": 0.0}.get(cross, 0.0)


def _score_macd(hist_pct: float) -> float:
    return float(np.clip(hist_pct * 25, -0.6, 0.6))


def _score_bollinger(bb_pos: float) -> float:
    """Mean-reversion: at upper band → bearish, at lower → bullish."""
    return float(np.clip(-bb_pos * 0.4, -0.6, 0.6))


def _score_adx(adx: float, slope_pct: float) -> float:
    if adx < 20:
        return 0.0
    strength = float(np.clip((adx - 20) / 30, 0.0, 1.0))
    direction = 1.0 if slope_pct > 0 else -1.0 if slope_pct < 0 else 0.0
    return float(np.clip(strength * direction * 0.5, -0.5, 0.5))


def _score_obv(obv_slope: float) -> float:
    return float(np.clip(obv_slope * 0.3, -0.4, 0.4))


def _score_vwap(vwap_dist_pct: float) -> float:
    return float(np.clip(vwap_dist_pct * 0.05, -0.3, 0.3))


def _score_skewness(skew: float) -> float:
    return float(np.clip(skew * 0.15, -0.3, 0.3))


def _score_trend_bias(bias: float) -> float:
    return float(np.clip((bias - 0.5) * 2.0, -0.4, 0.4))


def _score_rsi_divergence(div: float) -> float:
    """Divergence is a strong mean-reversion signal. ±0.5 max contribution."""
    return float(np.clip(div * 0.5, -0.5, 0.5))


def _score_ema200_dist(dist_pct: float) -> float:
    """
    Price far above EMA200 → mildly bearish (overbought long-term).
    Price far below EMA200 → mildly bullish (oversold long-term).
    Capped at ±0.25 - this is a slow signal.
    """
    return float(np.clip(-dist_pct * 0.012, -0.25, 0.25))


# Each row sums to 1.0. Trend regimes lean on slope/MACD/ADX; range regimes
# lean on RSI/Bollinger; breakout regimes lean on momentum + ADX.
# v3: rsi_div and ema200 added as small but useful signals.
_BASE_WEIGHTS = {
    "rsi": 0.09,
    "slope": 0.15,
    "momentum": 0.11,
    "ema": 0.09,
    "macd": 0.11,
    "bollinger": 0.07,
    "adx": 0.07,
    "obv": 0.07,
    "vwap": 0.05,
    "skew": 0.04,
    "trend_bias": 0.06,
    "rsi_div": 0.05,  # RSI divergence - mean reversion
    "ema200": 0.04,  # long-term trend anchor
}

# Strong trend weights are identical for up and down - direction is encoded in
# the sub-scores themselves, not in the weights.
_WEIGHTS_STRONG_TREND: dict[str, float] = {
    "rsi": 0.03,
    "slope": 0.20,
    "momentum": 0.15,
    "ema": 0.13,
    "macd": 0.15,
    "bollinger": 0.02,
    "adx": 0.11,
    "obv": 0.07,
    "vwap": 0.03,
    "skew": 0.01,
    "trend_bias": 0.02,
    "rsi_div": 0.04,
    "ema200": 0.04,
}
_WEIGHTS_WEAK_TREND: dict[str, float] = {
    "rsi": 0.05,
    "slope": 0.18,
    "momentum": 0.13,
    "ema": 0.11,
    "macd": 0.13,
    "bollinger": 0.04,
    "adx": 0.09,
    "obv": 0.09,
    "vwap": 0.05,
    "skew": 0.02,
    "trend_bias": 0.03,
    "rsi_div": 0.05,
    "ema200": 0.03,
}
_WEIGHTS_BREAKOUT: dict[str, float] = {
    "rsi": 0.02,
    "slope": 0.19,
    "momentum": 0.19,
    "ema": 0.09,
    "macd": 0.17,
    "bollinger": 0.02,
    "adx": 0.15,
    "obv": 0.08,
    "vwap": 0.02,
    "skew": 0.01,
    "trend_bias": 0.01,
    "rsi_div": 0.03,
    "ema200": 0.02,
}

_REGIME_WEIGHTS = {
    "strong_uptrend": _WEIGHTS_STRONG_TREND,
    "strong_downtrend": _WEIGHTS_STRONG_TREND,
    "weak_uptrend": _WEIGHTS_WEAK_TREND,
    "weak_downtrend": _WEIGHTS_WEAK_TREND,
    "breakout_up": _WEIGHTS_BREAKOUT,
    "breakout_down": _WEIGHTS_BREAKOUT,
    "range_bound": {
        "rsi": 0.20,
        "slope": 0.03,
        "momentum": 0.03,
        "ema": 0.02,
        "macd": 0.05,
        "bollinger": 0.27,
        "adx": 0.02,
        "obv": 0.05,
        "vwap": 0.09,
        "skew": 0.05,
        "trend_bias": 0.07,
        "rsi_div": 0.08,
        "ema200": 0.04,  # divergence is very useful in range
    },
    "choppy": _BASE_WEIGHTS,
}

# Module-level cache: re-parsed only when the raw config string changes.
_weights_cache_key: str = ""
_weights_cache_value: dict[str, float] | None = None


def _load_custom_base_weights() -> dict[str, float] | None:
    """
    Load custom base weights from cfg.signal_base_weights if set.
    Format: comma-separated floats in the fixed order:
      rsi, slope, momentum, ema, macd, bollinger, adx, obv, vwap, skew, trend_bias, rsi_div, ema200
    Returns None if not set or if parsing fails.

    Result is cached at module level and only re-parsed when the raw string changes,
    avoiding repeated string-splitting on every compute_signal() call.
    """
    global _weights_cache_key, _weights_cache_value

    raw = (cfg.signal_base_weights or "").strip()
    if raw == _weights_cache_key:  # fast path - nothing changed
        return _weights_cache_value

    _weights_cache_key = raw
    if not raw:
        _weights_cache_value = None
        return None

    keys = [
        "rsi",
        "slope",
        "momentum",
        "ema",
        "macd",
        "bollinger",
        "adx",
        "obv",
        "vwap",
        "skew",
        "trend_bias",
        "rsi_div",
        "ema200",
    ]
    try:
        vals = [float(v.strip()) for v in raw.split(",")]
        if len(vals) != len(keys):
            logger.warning(
                "signal: SIGNAL_BASE_WEIGHTS has %d values, expected %d - using defaults",
                len(vals),
                len(keys),
            )
            _weights_cache_value = None
            return None
        total = sum(vals)
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "signal: SIGNAL_BASE_WEIGHTS sums to %.3f (expected 1.0) - normalising",
                total,
            )
            vals = [v / total for v in vals]
        _weights_cache_value = dict(zip(keys, vals, strict=False))
        return _weights_cache_value
    except Exception:
        logger.warning("signal: failed to parse SIGNAL_BASE_WEIGHTS=%r - using defaults", raw)
        _weights_cache_value = None
        return None


def _weights_for(regime: Regime | None) -> dict[str, float]:
    # Allow runtime override of base weights via config
    custom_base = _load_custom_base_weights()
    base = custom_base if custom_base is not None else _BASE_WEIGHTS
    if regime is None:
        return base
    # For choppy, always use base (don't use the regime map)
    if regime.regime == "choppy":
        return base
    return _REGIME_WEIGHTS.get(regime.regime, base)


def _confidence(scores: dict[str, float], regime: Regime | None) -> float:
    if not scores:
        return 0.0

    vals = list(scores.values())
    active = [v for v in vals if abs(v) > 0.05]
    if not active:
        return 0.0

    pos = sum(1 for v in active if v > 0)
    neg = sum(1 for v in active if v < 0)
    n = len(active)

    agreement = max(pos, neg) / n

    total = len(vals)
    zero = total - n
    p = np.array([pos, neg, zero], dtype=float) / total
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    Hmax = math.log(3)
    H_norm = H / Hmax if Hmax else 0.0
    entropy_factor = 1.0 - H_norm * 0.5

    mag = float(np.mean([abs(v) for v in active]))
    mag_factor = float(np.clip(mag / 0.5, 0.0, 1.0))

    raw = agreement * entropy_factor * (0.5 + 0.5 * mag_factor)

    # Regime amplifier: clean regimes add confidence, choppy subtracts.
    if regime is not None:
        if regime.regime in ("strong_uptrend", "strong_downtrend", "breakout_up", "breakout_down"):
            raw *= 1.20
        elif regime.regime in ("weak_uptrend", "weak_downtrend"):
            raw *= 1.05
        elif regime.regime == "choppy":
            raw *= 0.55
        # range_bound left at 1.0 - RSI/BB are doing real work

    return round(float(np.clip(raw, 0.0, 1.0)), 3)


def compute_signal(ind: Indicators, regime: Regime | None = None) -> Signal:

    sub_scores: dict[str, float] = {
        "rsi": _score_rsi(ind.rsi),
        "slope": _score_slope(ind.slope),
        "momentum": _score_momentum(ind.momentum),
        "ema": _score_ema(ind.ema_cross),
        "macd": _score_macd(ind.macd_hist),
        "bollinger": _score_bollinger(ind.bb_position),
        "adx": _score_adx(ind.adx, ind.slope),
        "obv": _score_obv(ind.obv_slope),
        "vwap": _score_vwap(ind.vwap_dist),
        "skew": _score_skewness(ind.skewness),
        "trend_bias": _score_trend_bias(ind.trend_bias),
        # v3
        "rsi_div": _score_rsi_divergence(ind.rsi_divergence),
        "ema200": _score_ema200_dist(ind.ema200_dist),
    }

    weights = _weights_for(regime)

    composite = float(np.clip(sum(sub_scores[k] * weights[k] for k in weights), -1.0, 1.0))

    # If we have a regime view, blend the regime's directional read into the
    # composite. This is the key change: the regime is the dominant voice.
    if regime is not None:
        # regime.trend_score is signed in [-1, 1]
        composite = float(np.clip(0.55 * regime.trend_score + 0.45 * composite, -1.0, 1.0))

    confidence = _confidence(sub_scores, regime)

    std_dec = ind.std_return / 100.0
    mean_dec = ind.mean_return / 100.0

    base_drift = mean_dec
    signal_adj = composite * confidence * (std_dec * 0.5)
    drift_bias = float(np.clip(base_drift + signal_adj, -2.0 * std_dec, 2.0 * std_dec))

    base_vol = ind.atr_pct / 100.0
    vol_scale = {"low": 0.75, "normal": 1.0, "high": 1.40}.get(ind.vol_regime, 1.0)
    vol_adj = float(np.clip(base_vol * vol_scale, 0.003, 0.06))

    # In range-bound regimes: tighten vol (less directional movement expected)
    # In breakout regimes: widen vol (volatility expansion is the whole point)
    if regime is not None:
        if regime.regime == "range_bound":
            vol_adj = float(np.clip(vol_adj * 0.85, 0.003, 0.06))
        elif regime.regime in ("breakout_up", "breakout_down"):
            vol_adj = float(np.clip(vol_adj * 1.25, 0.003, 0.06))

    # On gap days: suppress BOTH the signal adjustment AND the base drift.
    # The base drift (historical mean) is irrelevant when a gap event occurred -
    # post-gap price action is driven by news, not the historical drift regime.
    # We zero out drift entirely and inflate vol to represent the uncertainty.
    gap_warning = ""
    if ind.is_gap_up or ind.is_gap_down:
        direction = "UP" if ind.is_gap_up else "DOWN"
        gap_warning = (
            f"GAP {direction} {ind.gap_pct:+.1f}% detected - likely news-driven. "
            f"Drift bias zeroed (historical drift unreliable post-gap). "
            f"MC volatility inflated to reflect elevated uncertainty."
        )
        drift_bias = 0.0  # zero drift - gap day mean reversion is unpredictable
        signal_adj = 0.0
        vol_adj = float(np.clip(vol_adj * 1.6, 0.003, 0.06))

    eff = composite * confidence
    if regime is not None and regime.regime in ("breakout_up", "breakout_down"):
        # Breakouts: lower bar to call it
        if eff > 0.25:
            label = "Strong buy"
        elif eff > 0.10:
            label = "Buy"
        elif eff < -0.25:
            label = "Strong sell"
        elif eff < -0.10:
            label = "Sell"
        else:
            label = "Neutral"
    elif regime is not None and regime.regime == "range_bound":
        # In a range we want bigger conviction before calling a side
        if eff > 0.30:
            label = "Buy"
        elif eff < -0.30:
            label = "Sell"
        else:
            label = "Neutral"
    else:
        if eff > 0.40:
            label = "Strong buy"
        elif eff > 0.15:
            label = "Buy"
        elif eff < -0.40:
            label = "Strong sell"
        elif eff < -0.15:
            label = "Sell"
        else:
            label = "Neutral"

    rsi_desc = (
        f"RSI {ind.rsi:.0f} oversold"
        if ind.rsi < 35
        else f"RSI {ind.rsi:.0f} overbought"
        if ind.rsi > 65
        else f"RSI {ind.rsi:.0f} neutral"
    )
    macd_dir = "↑" if ind.macd_hist > 0 else "↓" if ind.macd_hist < 0 else "·"
    drift_pct = drift_bias * 100
    regime_part = f"regime: {regime.regime} · " if regime is not None else ""
    reasoning = (
        f"{regime_part}{rsi_desc} · EMA {ind.ema_cross} · "
        f"MACD {macd_dir} · ADX {ind.adx:.0f} · "
        f"drift {drift_pct:+.3f}%/c · "
        f"conf {confidence:.0%} · "
        f"vol {ind.vol_regime}"
    )

    # NaN guards
    drift_bias = _safe(drift_bias, 0.0)
    vol_adj = _safe(vol_adj, 0.01)
    base_drift = _safe(base_drift, 0.0)
    signal_adj = _safe(signal_adj, 0.0)
    composite = _safe(composite, 0.0)
    confidence = _safe(confidence, 0.0)

    return Signal(
        composite=round(composite, 4),
        confidence=round(confidence, 3),
        drift_bias=round(drift_bias, 6),
        base_drift=round(base_drift, 6),
        signal_adj=round(signal_adj, 6),
        vol_adj=round(vol_adj, 6),
        label=label,
        reasoning=reasoning,
        gap_warning=gap_warning,
        sub_scores={k: round(float(v), 3) for k, v in sub_scores.items()},
        weights={k: round(float(v), 3) for k, v in weights.items()},
    )
