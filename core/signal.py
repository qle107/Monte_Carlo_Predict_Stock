"""
core/signal.py
Computes the AI signal and — critically — derives drift bias and
simulation volatility automatically from the stock's own history.

How drift bias is calculated:
──────────────────────────────
  base_drift        = stock's actual mean return per candle (from history)
  signal_adjustment = composite × confidence × (std_return × 0.5)
  drift_bias        = base_drift + signal_adjustment

  Cap: drift_bias is clamped to ±2 × std_return so it can never exceed
       two standard deviations — prevents the 99.9% trap.

Why this is better than a fixed constant:
  - PLTR (4% daily vol) gets a much wider MC cone than SPY (0.5% vol)
  - A stock that historically drifts up 0.03%/candle keeps that bias
  - The AI signal can only shift the drift by up to ½ a std dev
  - Gaps suppress the signal portion but preserve the historical base

Vol regime adjustment:
  - "high" regime → inflate simulation vol by 1.4×  (wider paths)
  - "low"  regime → deflate by 0.75×                 (tighter paths)
  - "normal"      → use std_return as-is
"""

import numpy as np
from dataclasses import dataclass
from .indicators import Indicators, _safe


@dataclass
class Signal:
    # scores
    composite:        float   # raw weighted score  −1 … +1
    confidence:       float   # 0 … 1 (signal alignment)

    # MC parameters (auto-derived from stock history)
    drift_bias:       float   # per-candle drift injected into MC (%)
    base_drift:       float   # historical mean return component
    signal_adj:       float   # signal adjustment component
    vol_adj:          float   # per-candle volatility for MC (decimal)

    # metadata
    label:            str
    reasoning:        str
    gap_warning:      str


# ── Sub-signal scorers ───────────────────────────────────────────────────────

def _score_rsi(rsi: float) -> float:
    """
    Graduated RSI score. Max ±0.6 to avoid over-dominance.
    Treats RSI as a mean-reversion hint, not a momentum signal.
    """
    if   rsi < 20: return  0.60
    elif rsi < 30: return  0.40
    elif rsi < 40: return  0.15
    elif rsi < 60: return  0.00
    elif rsi < 70: return -0.15
    elif rsi < 80: return -0.40
    else:          return -0.60


def _score_slope(slope: float) -> float:
    """Regression slope (%/candle). Capped at ±0.8."""
    return float(np.clip(slope * 15, -0.8, 0.8))


def _score_momentum(mom: float) -> float:
    """5-candle momentum. Capped at ±0.6."""
    return float(np.clip(mom * 5, -0.6, 0.6))


def _score_ema(cross: str) -> float:
    return {"bullish": 0.4, "bearish": -0.4, "neutral": 0.0}[cross]


def _score_skewness(skew: float) -> float:
    """
    Positive skew (more upside tails) → mild bullish bonus.
    Negative skew (crash-prone) → mild bearish penalty.
    Capped at ±0.3 so it never dominates.
    """
    return float(np.clip(skew * 0.15, -0.3, 0.3))


def _score_trend_bias(bias: float) -> float:
    """
    trend_bias > 0.55 means the stock closes up more than 55% of candles
    → mild bullish. Centred at 0.5 (random walk baseline).
    """
    return float(np.clip((bias - 0.5) * 2.0, -0.4, 0.4))


# ── Confidence ───────────────────────────────────────────────────────────────

def _confidence(scores: list[float]) -> float:
    """
    Measures signal alignment: 1.0 = all signals agree, 0.0 = all disagree.
    Always returns a value in [0.0, 1.0].
    """
    if not scores:
        return 0.0
    signs    = [1 if s > 0.05 else -1 if s < -0.05 else 0 for s in scores]
    majority = max(signs.count(1), signs.count(-1))
    # clamp to [0, 1]: majority/total gives 0..1, then clip just in case
    raw = majority / len(scores)
    return round(float(max(0.0, min(1.0, raw))), 3)


# ── Main ─────────────────────────────────────────────────────────────────────

def compute_signal(ind: Indicators) -> Signal:
    # ── 1. Composite score ───────────────────────────────────────────────
    rsi_s   = _score_rsi(ind.rsi)
    slope_s = _score_slope(ind.slope)
    mom_s   = _score_momentum(ind.momentum)
    ema_s   = _score_ema(ind.ema_cross)
    skew_s  = _score_skewness(ind.skewness)
    tbias_s = _score_trend_bias(ind.trend_bias)

    all_scores = [rsi_s, slope_s, mom_s, ema_s, skew_s, tbias_s]

    composite = (
        rsi_s   * 0.20 +
        slope_s * 0.25 +   # slope most important on short timeframes
        mom_s   * 0.20 +
        ema_s   * 0.15 +
        skew_s  * 0.10 +   # skewness: stock character
        tbias_s * 0.10     # trend bias: is this stock a persistent drifter?
    )
    composite  = float(np.clip(composite, -1.0, 1.0))
    confidence = _confidence(all_scores)

    # ── 2. Drift bias — auto-derived from stock's own history ────────────
    #
    # std_return is in % (e.g. 1.5 for a 1.5%-per-candle volatile stock).
    # Convert to decimal for drift calculation.
    std_dec  = ind.std_return / 100.0   # e.g. 0.015
    mean_dec = ind.mean_return / 100.0  # e.g. 0.0002

    # Base: the stock's actual historical drift per candle
    base_drift = mean_dec

    # Signal adjustment: composite × confidence × half a std dev
    # This means even a perfect signal can only shift drift by 0.5σ
    signal_adj = composite * confidence * (std_dec * 0.5)

    # Combined drift
    drift_bias = base_drift + signal_adj

    # Hard cap: drift cannot exceed ±2 standard deviations
    # This is the fix for the 99.9% bug — statistically, 2σ is the limit
    drift_bias = float(np.clip(drift_bias, -2.0 * std_dec, 2.0 * std_dec))

    # ── 3. Simulation volatility — from ATR + vol regime ─────────────────
    #
    # Use ATR-based vol as the primary estimate (more stable than std_return
    # on short lookbacks), then scale by vol regime.
    base_vol = ind.atr_pct / 100.0   # ATR as decimal

    vol_scale = {"low": 0.75, "normal": 1.0, "high": 1.40}[ind.vol_regime]
    vol_adj   = float(np.clip(base_vol * vol_scale, 0.003, 0.06))

    # ── 4. Gap / news override ────────────────────────────────────────────
    gap_warning = ""
    if ind.is_gap_up or ind.is_gap_down:
        direction = "UP" if ind.is_gap_up else "DOWN"
        gap_warning = (
            f"GAP {direction} {ind.gap_pct:+.1f}% detected — likely news-driven. "
            f"Signal adjustment suppressed; base historical drift preserved. "
            f"MC volatility inflated to reflect elevated uncertainty."
        )
        # Keep base_drift (historical) but zero the signal adjustment
        drift_bias = float(np.clip(base_drift, -2.0 * std_dec, 2.0 * std_dec))
        signal_adj = 0.0
        # Inflate vol for gap sessions (news = higher uncertainty)
        vol_adj    = float(np.clip(vol_adj * 1.6, 0.003, 0.06))

    # ── 5. Label ──────────────────────────────────────────────────────────
    eff = composite * confidence
    if   eff >  0.40: label = "Strong buy"
    elif eff >  0.15: label = "Buy"
    elif eff < -0.40: label = "Strong sell"
    elif eff < -0.15: label = "Sell"
    else:             label = "Neutral"

    # ── 6. Reasoning string ───────────────────────────────────────────────
    rsi_desc = (
        f"RSI {ind.rsi:.0f} oversold" if ind.rsi < 35 else
        f"RSI {ind.rsi:.0f} overbought" if ind.rsi > 65 else
        f"RSI {ind.rsi:.0f} neutral"
    )
    drift_pct = drift_bias * 100
    reasoning = (
        f"{rsi_desc} · EMA {ind.ema_cross} · "
        f"slope {'↑' if ind.slope > 0 else '↓'} · "
        f"hist drift {ind.mean_return:+.3f}%/c · "
        f"adj {drift_pct:+.3f}%/c · "
        f"conf {confidence:.0%} · "
        f"vol regime {ind.vol_regime}"
    )

    # Final NaN guard — ensure nothing escapes into MC
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
    )
