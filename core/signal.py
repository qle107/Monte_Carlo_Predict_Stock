"""
core/signal.py
Composite AI signal + auto-derived MC drift/vol parameters.

Drift bias
──────────
  base_drift        = stock's actual mean return per candle (from history)
  signal_adjustment = composite × confidence × (std_return × 0.5)
  drift_bias        = base_drift + signal_adjustment
  Hard cap          = ±2 × std_return  (prevents the "99.9% certain" trap)

Volatility (input to MC engine)
───────────────────────────────
  base_vol  = ATR%
  scaled by vol_regime  ({low: 0.75, normal: 1.0, high: 1.4})
  inflated 1.6× when a price gap is detected

New in v2
─────────
  • MACD histogram, Bollinger band position, ADX, OBV slope, VWAP distance,
    and trend-bias all feed the composite score with calibrated weights.
  • Confidence is now an *entropy-aware* alignment metric — it is high
    only when most active signals point the same way AND there are
    enough active signals to matter.
  • All sub-scores returned in `sub_scores` so the dashboard / backtest
    can introspect what drove the call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .indicators import Indicators, _safe


# ─── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class Signal:
    composite:   float
    confidence:  float
    drift_bias:  float
    base_drift:  float
    signal_adj:  float
    vol_adj:     float
    label:       str
    reasoning:   str
    gap_warning: str
    sub_scores:  Dict[str, float] = field(default_factory=dict)


# ─── Sub-signal scorers ─────────────────────────────────────────────────────

def _score_rsi(rsi: float) -> float:
    """Mean-reversion read on RSI. Capped ±0.6."""
    if   rsi < 20: return  0.60
    elif rsi < 30: return  0.40
    elif rsi < 40: return  0.15
    elif rsi < 60: return  0.00
    elif rsi < 70: return -0.15
    elif rsi < 80: return -0.40
    else:          return -0.60


def _score_slope(slope_pct: float) -> float:
    return float(np.clip(slope_pct * 15, -0.8, 0.8))


def _score_momentum(mom_pct: float) -> float:
    return float(np.clip(mom_pct * 5, -0.6, 0.6))


def _score_ema(cross: str) -> float:
    return {"bullish": 0.4, "bearish": -0.4, "neutral": 0.0}.get(cross, 0.0)


def _score_macd(hist_pct: float) -> float:
    """MACD histogram (% of price). Strong signal when hist is large."""
    return float(np.clip(hist_pct * 25, -0.6, 0.6))


def _score_bollinger(bb_pos: float) -> float:
    """
    Mean-reversion: at upper band → bearish, at lower → bullish.
    bb_pos ∈ [-3, 3]; ±1 = at the band, ±2+ = punched through.
    """
    return float(np.clip(-bb_pos * 0.4, -0.6, 0.6))


def _score_adx(adx: float, slope_pct: float) -> float:
    """
    ADX is direction-agnostic, so we pair it with slope sign:
       strong trend  + up slope   → bullish
       strong trend  + down slope → bearish
       weak trend                 → 0
    """
    if adx < 20:
        return 0.0
    strength = float(np.clip((adx - 20) / 30, 0.0, 1.0))   # 20→0, 50+→1
    direction = 1.0 if slope_pct > 0 else -1.0 if slope_pct < 0 else 0.0
    return float(np.clip(strength * direction * 0.5, -0.5, 0.5))


def _score_obv(obv_slope: float) -> float:
    """Volume-confirmed momentum."""
    return float(np.clip(obv_slope * 0.3, -0.4, 0.4))


def _score_vwap(vwap_dist_pct: float) -> float:
    """
    Trend-following: above VWAP → bullish bias.
    Capped — too far above is overextension territory.
    """
    return float(np.clip(vwap_dist_pct * 0.05, -0.3, 0.3))


def _score_skewness(skew: float) -> float:
    return float(np.clip(skew * 0.15, -0.3, 0.3))


def _score_trend_bias(bias: float) -> float:
    return float(np.clip((bias - 0.5) * 2.0, -0.4, 0.4))


# ─── Confidence (entropy-aware) ─────────────────────────────────────────────

def _confidence(scores: Dict[str, float]) -> float:
    """
    Confidence ∈ [0,1]. High when:
      (1) most active scores agree on direction, AND
      (2) there are enough active scores (entropy penalty for sparse signals).
    """
    if not scores:
        return 0.0

    vals = list(scores.values())
    active = [v for v in vals if abs(v) > 0.05]
    if not active:
        return 0.0

    pos = sum(1 for v in active if v > 0)
    neg = sum(1 for v in active if v < 0)
    n   = len(active)

    # Directional agreement: 1.0 when unanimous
    agreement = max(pos, neg) / n

    # Entropy of distribution over { +, -, 0 } across all scores including zeros
    total = len(vals)
    zero  = total - n
    p     = np.array([pos, neg, zero], dtype=float) / total
    p     = p[p > 0]
    H     = float(-(p * np.log(p)).sum())             # nats
    Hmax  = math.log(3)
    H_norm = H / Hmax if Hmax else 0.0
    # Lower entropy → higher confidence (signals concentrated in one bucket)
    entropy_factor = 1.0 - H_norm * 0.5               # max penalty 0.5

    # Magnitude bonus: average |score| of active signals
    mag = float(np.mean([abs(v) for v in active]))
    mag_factor = float(np.clip(mag / 0.5, 0.0, 1.0))   # full credit at avg |s|=0.5

    raw = agreement * entropy_factor * (0.5 + 0.5 * mag_factor)
    return round(float(np.clip(raw, 0.0, 1.0)), 3)


# ─── Main ───────────────────────────────────────────────────────────────────

def compute_signal(ind: Indicators) -> Signal:
    # 1. Sub-scores ──────────────────────────────────────────────────────────
    sub_scores: Dict[str, float] = {
        "rsi":        _score_rsi(ind.rsi),
        "slope":      _score_slope(ind.slope),
        "momentum":   _score_momentum(ind.momentum),
        "ema":        _score_ema(ind.ema_cross),
        "macd":       _score_macd(ind.macd_hist),
        "bollinger":  _score_bollinger(ind.bb_position),
        "adx":        _score_adx(ind.adx, ind.slope),
        "obv":        _score_obv(ind.obv_slope),
        "vwap":       _score_vwap(ind.vwap_dist),
        "skew":       _score_skewness(ind.skewness),
        "trend_bias": _score_trend_bias(ind.trend_bias),
    }

    # Calibrated weights (sum to 1.0)
    weights = {
        "rsi":        0.10,
        "slope":      0.16,
        "momentum":   0.12,
        "ema":        0.10,
        "macd":       0.12,
        "bollinger":  0.08,
        "adx":        0.08,
        "obv":        0.08,
        "vwap":       0.05,
        "skew":       0.05,
        "trend_bias": 0.06,
    }

    composite = float(np.clip(
        sum(sub_scores[k] * weights[k] for k in weights), -1.0, 1.0
    ))
    confidence = _confidence(sub_scores)

    # 2. Drift bias from stock's own history ────────────────────────────────
    std_dec  = ind.std_return / 100.0
    mean_dec = ind.mean_return / 100.0

    base_drift = mean_dec
    signal_adj = composite * confidence * (std_dec * 0.5)
    drift_bias = float(np.clip(base_drift + signal_adj, -2.0 * std_dec, 2.0 * std_dec))

    # 3. Simulation volatility ──────────────────────────────────────────────
    base_vol  = ind.atr_pct / 100.0
    vol_scale = {"low": 0.75, "normal": 1.0, "high": 1.40}.get(ind.vol_regime, 1.0)
    vol_adj   = float(np.clip(base_vol * vol_scale, 0.003, 0.06))

    # 4. Gap / news override ────────────────────────────────────────────────
    gap_warning = ""
    if ind.is_gap_up or ind.is_gap_down:
        direction = "UP" if ind.is_gap_up else "DOWN"
        gap_warning = (
            f"GAP {direction} {ind.gap_pct:+.1f}% detected — likely news-driven. "
            f"Signal adjustment suppressed; base historical drift preserved. "
            f"MC volatility inflated to reflect elevated uncertainty."
        )
        drift_bias = float(np.clip(base_drift, -2.0 * std_dec, 2.0 * std_dec))
        signal_adj = 0.0
        vol_adj    = float(np.clip(vol_adj * 1.6, 0.003, 0.06))

    # 5. Label ──────────────────────────────────────────────────────────────
    eff = composite * confidence
    if   eff >  0.40: label = "Strong buy"
    elif eff >  0.15: label = "Buy"
    elif eff < -0.40: label = "Strong sell"
    elif eff < -0.15: label = "Sell"
    else:             label = "Neutral"

    # 6. Reasoning string ───────────────────────────────────────────────────
    rsi_desc = (
        f"RSI {ind.rsi:.0f} oversold"   if ind.rsi < 35 else
        f"RSI {ind.rsi:.0f} overbought" if ind.rsi > 65 else
        f"RSI {ind.rsi:.0f} neutral"
    )
    drift_pct = drift_bias * 100
    macd_dir  = "↑" if ind.macd_hist > 0 else "↓" if ind.macd_hist < 0 else "·"
    reasoning = (
        f"{rsi_desc} · EMA {ind.ema_cross} · "
        f"slope {'↑' if ind.slope > 0 else '↓' if ind.slope < 0 else '·'} · "
        f"MACD {macd_dir} · ADX {ind.adx:.0f} · "
        f"hist drift {ind.mean_return:+.3f}%/c · "
        f"adj {drift_pct:+.3f}%/c · "
        f"conf {confidence:.0%} · "
        f"vol {ind.vol_regime}"
    )

    # NaN guards
    drift_bias = _safe(drift_bias, 0.0)
    vol_adj    = _safe(vol_adj,    0.01)
    base_drift = _safe(base_drift, 0.0)
    signal_adj = _safe(signal_adj, 0.0)
    composite  = _safe(composite,  0.0)
    confidence = _safe(confidence, 0.0)

    return Signal(
        composite   = round(composite, 4),
        confidence  = round(confidence, 3),
        drift_bias  = round(drift_bias, 6),
        base_drift  = round(base_drift, 6),
        signal_adj  = round(signal_adj, 6),
        vol_adj     = round(vol_adj, 6),
        label       = label,
        reasoning   = reasoning,
        gap_warning = gap_warning,
        sub_scores  = {k: round(float(v), 3) for k, v in sub_scores.items()},
    )
