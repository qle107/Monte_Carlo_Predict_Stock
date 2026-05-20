"""
core/trade_setup.py
───────────────────
Computes Entry / Stop-Loss / Take-Profit / Risk-Reward for a ticker
based on the existing indicator + regime + Monte Carlo pipeline output.

Entry logic
───────────
Valid entry requires ALL of:
  1. |score| >= 0.28  (meaningful directional signal)
  2. ADX >= 20        (trend present, not pure chop)
  3. MC prob in the signal direction >= 0.45
  4. Confidence >= 0.40
  5. RSI 20–80 (not wildly over/oversold in the wrong direction)
  6. Regime not in ("choppy", "range_bound") for breakout plays

Stop Loss (two flavours, both shown)
─────────
  ATR-based : SL = entry ∓ (ATR × multiplier)
               multiplier: 1.5× breakout, 2.0× trend, 2.5× other
  Fixed-pct : SL = entry ∓ pct
               pct: 2.0% breakout/trending, 3.0% other

Take Profit (MC percentiles)
────────────────────────────
  TP1 = P75  (first target — partial exit)
  TP2 = P90  (runner target — rest of position)

  For SHORT setups (breakdown):
  TP1 = P25 (below), TP2 = P10

Risk:Reward
───────────
  RR_atr = (TP1 - entry) / (entry - SL_atr)   for longs
  RR_pct = (TP1 - entry) / (entry - SL_pct)

Probability of hitting TP before SL
────────────────────────────────────
  Uses the MC final-price distribution:
    p_tp1 = fraction of simulations where final >= TP1 (longs)
    p_tp2 = fraction of simulations where final >= TP2

  For shorts the inequalities reverse.

Entry price
───────────
  Estimated as current_price * (1 + small_slippage_factor).
  For breakouts: +0.1% (buy slightly above last close to confirm break).
  For breakdowns: -0.1% (sell slightly below).
  For trending: at current close (already confirmed).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List
import numpy as np
import pandas as pd

from .zones import detect_zones, ZoneResult
from config import cfg


# ─── Thresholds ──────────────────────────────────────────────────────────────
# These are read from cfg at call time so live config changes take effect.
# The module-level constants below are kept as documentation / fallbacks only.

def _min_score()    -> float: return cfg.min_score
def _min_adx()      -> float: return cfg.min_adx
def _min_conf()     -> float: return cfg.min_conf
def _min_mc_prob()  -> float: return cfg.min_mc_prob
def _min_rr()       -> float: return cfg.min_rr
def _rsi_overbought() -> float: return cfg.rsi_overbought
def _rsi_oversold()   -> float: return cfg.rsi_oversold
def _min_score_choppy() -> float: return cfg.min_score_choppy
def _fixed_max_pct()  -> float: return cfg.sl_max_pct

# ATR multipliers per regime type.
# Also covers scanner direction strings ("bullish", "bearish", "trending_up", etc.)
# so the lookup never falls through to a mismatched default.
_ATR_MULT = {
    # Regime keys (from detect_regime)
    "breakout_up":      1.5,
    "breakout_down":    1.5,
    "strong_uptrend":   2.0,
    "strong_downtrend": 2.0,
    "weak_uptrend":     2.5,
    "weak_downtrend":   2.5,
    "range_bound":      2.5,
    "choppy":           3.0,
    # Scanner direction keys (from _classify_direction)
    "bullish":          2.0,   # treated like weak uptrend
    "bearish":          2.0,
    "trending_up":      2.0,
    "trending_down":    2.0,
    "breakdown":        1.5,
    "neutral":          3.0,
}

# Fixed % SL — BASE values per regime/direction (before volatility scaling).
# These are the minimum acceptable distances.
# Final SL = max(base, ATR-scaled suggestion) capped at _FIXED_MAX_PCT.
# Scaling: if ATR% is large, SL widens up to the max so we don't get
# stopped out by normal candle noise on volatile stocks.
_FIXED_BASE_PCT = {
    # Regime keys
    "breakout_up":      0.030,   # 3.0% base — breakouts need room after the gap
    "breakout_down":    0.030,
    "strong_uptrend":   0.030,   # 3% — confirmed trend, tight-ish is ok
    "strong_downtrend": 0.030,
    "weak_uptrend":     0.040,   # 4% — less conviction, more room
    "weak_downtrend":   0.040,
    "range_bound":      0.040,   # 4% — range can whip both ways
    "choppy":           0.050,   # 5% — maximum; choppy = wide or skip
    # Scanner direction keys
    "bullish":          0.040,
    "bearish":          0.040,
    "trending_up":      0.035,
    "trending_down":    0.035,
    "breakdown":        0.030,
    "neutral":          0.050,
}
_FIXED_ATR_MULT = 1.5    # fixed-% SL = max(base, ATR × 1.5) — covers 1.5 ATR of noise

# Maximum ATR-SL distance as a fraction of price, per timeframe.
# If the raw ATR × mult distance exceeds this cap, we clamp it.
# This prevents gap-day ATR from producing 80%+ SL distances on intraday charts.
_MAX_ATR_SL_PCT = {
    "1m":  0.03,   # 3%  max intraday scalp
    "2m":  0.03,
    "5m":  0.04,
    "15m": 0.05,   # 5%  max — INTC 15m fix
    "30m": 0.06,
    "1h":  0.07,
    "2h":  0.08,
    "4h":  0.10,
    "1d":  0.12,   # 12% max daily
    "1wk": 0.20,
    "1mo": 0.30,
}


# ─── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class TradeSetup:
    ticker:       str
    side:         str       # "long" | "short" | "none"
    valid:        bool      # True if entry criteria met
    reason:       str       # human-readable reason (for "none" or context)

    entry:        Optional[float] = None
    sl_atr:       Optional[float] = None   # ATR-based stop loss
    sl_pct:       Optional[float] = None   # Fixed-% stop loss
    sl_atr_dist:  Optional[float] = None   # distance entry → sl_atr in %
    sl_pct_dist:  Optional[float] = None   # distance entry → sl_pct in %

    tp1:          Optional[float] = None   # first target  (P75 long / P25 short)
    tp2:          Optional[float] = None   # runner target (P90 long / P10 short)
    tp1_dist:     Optional[float] = None   # % distance entry → tp1
    tp2_dist:     Optional[float] = None   # % distance entry → tp2

    rr_atr:       Optional[float] = None   # RR using ATR stop
    rr_pct:       Optional[float] = None   # RR using fixed-% stop

    prob_tp1:     Optional[float] = None   # MC probability of reaching TP1
    prob_tp2:     Optional[float] = None   # MC probability of reaching TP2
    prob_sl_atr:  Optional[float] = None   # MC probability of hitting ATR stop
    prob_sl_pct:  Optional[float] = None   # MC probability of hitting fixed stop

    atr_mult:     Optional[float] = None   # which multiplier was used
    sl_pct_used:  Optional[float] = None   # which fixed % was used
    atr_capped:   Optional[bool]  = None   # True if ATR SL was capped to max %
    sl_recommended: Optional[str] = None   # "atr" | "pct" — which stop has better R:R

    # ── Zone-based TP / SL (demand / supply zones) ───────────────────
    # These are shown alongside MC P75/P90 — user picks which to use.
    zone_tp1:       Optional[float] = None   # nearest supply zone above (long TP)
    zone_tp2:       Optional[float] = None   # second supply zone (runner target)
    zone_sl:        Optional[float] = None   # zone-based SL: edge of nearest demand zone
    zone_tp1_dist:  Optional[float] = None   # % distance entry → zone_tp1
    zone_tp2_dist:  Optional[float] = None   # % from entry
    zone_sl_dist:   Optional[float] = None   # % risk to zone_sl
    zone_rr:        Optional[float] = None   # R:R using zone_tp1 / zone_sl
    zone_type:      Optional[str]   = None   # "demand_bounce" | "demand_break" |
                                             #  "supply_bounce" | "supply_break"
    zone_strength:  Optional[float] = None   # 0–1 strength of the triggering zone
    zone_context:   Optional[str]   = None   # "at_demand"|"at_supply"|"between"|"unknown"

    direction:    Optional[str]   = None   # breakout_up / trending_up / etc.
    regime:       Optional[str]   = None
    score:        Optional[float] = None
    confidence:   Optional[float] = None

    # ── Position sizing ──────────────────────────────────────────────────
    # Expressed as % of account equity to risk on this trade.
    # kelly_fraction       : raw Kelly % (may be aggressive — use half-Kelly)
    # kelly_half           : half-Kelly (recommended sizing)
    # fixed_frac           : fixed-fractional sizing (risk 1% of equity)
    # Both require knowing the per-trade win probability and avg W/L.
    kelly_fraction:  Optional[float] = None   # raw Kelly f  (%)
    kelly_half:      Optional[float] = None   # half-Kelly   (%)
    fixed_frac:      Optional[float] = None   # fixed-frac 1% equity (%)


def to_dict(ts: TradeSetup) -> dict:
    return asdict(ts)


# ─── Core computation ─────────────────────────────────────────────────────────

def compute_trade_setup(
    ticker:        str,
    price:         float,
    score:         float,
    direction:     str,
    regime:        str,
    confidence:    float,
    rsi:           float,
    adx:           float,
    atr_pct:       float,        # ATR as fraction of price (e.g. 0.015 = 1.5%)
    prob_up:       float,
    prob_down:     float,
    mc_p10:        Optional[float] = None,
    mc_p25:        Optional[float] = None,
    mc_p75:        Optional[float] = None,
    mc_p90:        Optional[float] = None,
    mc_final_prices: Optional[List[float]] = None,  # raw MC simulation finals
    interval:      str = "15m",  # timeframe — used to cap ATR SL distance
    df:            Optional[pd.DataFrame] = None,   # OHLCV — for zone detection
) -> TradeSetup:
    """
    Compute a full trade setup for one ticker.

    Parameters come directly from the scanner result or the main analyse() output.
    mc_final_prices (optional): the raw final prices from MC simulations — used
    for precise probability estimates of hitting TP/SL levels.
    """

    is_long  = score >= 0
    is_short = score < 0
    side     = "long" if is_long else "short"

    # ── Gate 1: Score magnitude ──────────────────────────────────────
    if abs(score) < _min_score():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"Score {score:+.2f} below threshold ±{_min_score()} — no clear directional edge",
        )

    # ── Gate 2: ADX (trend strength) ────────────────────────────────
    if adx < _min_adx():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"ADX {adx:.1f} < {_min_adx()} — trend too weak, avoid choppy conditions",
        )

    # ── Gate 3: Signal confidence ────────────────────────────────────
    if confidence < _min_conf():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"Confidence {confidence:.0%} below {_min_conf():.0%} — signals conflicting",
        )

    # ── Gate 4: MC directional probability ──────────────────────────
    # Only apply this gate when MC was actually run.
    # prob_up/prob_down may be:
    #   • percentages 0–100 (from main analyse() — MCResult)
    #   • fractions   0–1   (from scanner with mc_data)
    #   • 0.0               (scanner without MC — skip gate entirely)
    #
    # Normalise to fraction for comparison.
    _prob_u = prob_up   / 100.0 if prob_up   > 1.0 else prob_up
    _prob_d = prob_down / 100.0 if prob_down > 1.0 else prob_down
    mc_available = (_prob_u + _prob_d) > 0.50   # sum > 50% → real MC data present
    mc_dir_prob  = _prob_u if is_long else _prob_d
    if mc_available and mc_dir_prob < _min_mc_prob():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"MC {side} probability {mc_dir_prob:.0%} < {_min_mc_prob():.0%} — MC disagrees",
        )

    # ── Gate 5: RSI extremes ─────────────────────────────────────────
    if is_long and rsi > _rsi_overbought():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"RSI {rsi:.1f} > {_rsi_overbought():.0f} — severely overbought, wait for pullback",
        )
    if is_short and rsi < _rsi_oversold():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"RSI {rsi:.1f} < {_rsi_oversold():.0f} — severely oversold, wait for bounce first",
        )

    # ── Gate 6: Regime sanity ────────────────────────────────────────
    # Don't go long in a choppy or range-bound market unless score is very strong
    if regime in ("choppy", "range_bound") and abs(score) < _min_score_choppy():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"Regime '{regime}' — need score ≥ {_min_score_choppy()} for non-trending market, got {score:+.2f}",
        )

    # ── All gates passed: compute levels ────────────────────────────

    # Entry price = current close price.
    # If a breakout already happened (price moved past ideal entry), we still
    # accept the trade — the user can enter at market. No slippage penalty added
    # because the user decides when to actually execute.
    entry = round(price, 4)

    # ── Demand / Supply zone detection ──────────────────────────────
    # Run zone detection if OHLCV data is available.
    zones: ZoneResult = detect_zones(df) if df is not None and len(df) >= 10 else None

    # ATR in dollar terms
    atr_dollar = price * atr_pct if atr_pct > 0 else price * 0.015
    # Look up direction first (scanner sets direction, not regime key);
    # fall back to regime key, then to a safe default of 2.0.
    atr_mult   = _ATR_MULT.get(direction, _ATR_MULT.get(regime, 2.0))

    # ── ATR-based SL distance (capped per timeframe) ─────────────────
    raw_atr_dist = atr_dollar * atr_mult
    max_atr_pct  = _MAX_ATR_SL_PCT.get(interval, 0.07)
    max_atr_dist = entry * max_atr_pct
    atr_sl_dist_capped = min(raw_atr_dist, max_atr_dist)
    atr_was_capped = raw_atr_dist > max_atr_dist

    # ── Dynamic fixed-% SL: scales with volatility, 3%–5% ─────────
    # Logic:
    #   • Start with direction/regime base (3–5%)
    #   • Widen if ATR is large so we cover at least 1.5× ATR of noise
    #   • Hard cap at 5% — beyond that the trade risk/reward collapses
    base_pct    = _FIXED_BASE_PCT.get(direction, _FIXED_BASE_PCT.get(regime, 0.030))
    atr_suggest = atr_pct * _FIXED_ATR_MULT        # e.g. ATR=1.5% → suggest 2.25%
    sl_pct_val  = min(_fixed_max_pct(), max(base_pct, atr_suggest))
    # Also floor at base to avoid tiny SL on very stable stocks
    sl_pct_val  = max(sl_pct_val, base_pct)

    if is_long:
        sl_atr = round(entry - atr_sl_dist_capped, 4)
        sl_pct = round(entry * (1 - sl_pct_val), 4)
    else:  # short
        sl_atr = round(entry + atr_sl_dist_capped, 4)
        sl_pct = round(entry * (1 + sl_pct_val), 4)

    # Ensure SL doesn't cross entry due to tiny ATR
    if is_long and sl_atr >= entry:
        sl_atr = round(entry * 0.98, 4)
    if is_short and sl_atr <= entry:
        sl_atr = round(entry * 1.02, 4)

    # Take Profit levels — prefer MC percentiles but fall back to an
    # ATR-based projection when the MC percentile sits on the *wrong* side
    # of entry (which happens on tight, low-vol setups). The previous code
    # blindly inflated TP1 to entry × 1.015, producing a fake 1.5 % target
    # that didn't reflect realistic price action for the timeframe.
    atr_tp1_long  = entry + atr_dollar * atr_mult * 1.5
    atr_tp2_long  = entry + atr_dollar * atr_mult * 2.5
    atr_tp1_short = entry - atr_dollar * atr_mult * 1.5
    atr_tp2_short = entry - atr_dollar * atr_mult * 2.5

    have_mc = all(v is not None for v in (mc_p10, mc_p25, mc_p75, mc_p90))
    if is_long:
        mc_tp1 = mc_p75 if have_mc else None
        mc_tp2 = mc_p90 if have_mc else None
        tp1 = float(mc_tp1) if (mc_tp1 is not None and mc_tp1 > entry) else atr_tp1_long
        tp2 = float(mc_tp2) if (mc_tp2 is not None and mc_tp2 > tp1)   else max(atr_tp2_long, tp1 + atr_dollar)
    else:
        mc_tp1 = mc_p25 if have_mc else None
        mc_tp2 = mc_p10 if have_mc else None
        tp1 = float(mc_tp1) if (mc_tp1 is not None and mc_tp1 < entry) else atr_tp1_short
        tp2 = float(mc_tp2) if (mc_tp2 is not None and mc_tp2 < tp1)   else min(atr_tp2_short, tp1 - atr_dollar)

    tp1 = round(float(tp1), 4)
    tp2 = round(float(tp2), 4)

    # ── Zone-based TP / SL ───────────────────────────────────────────
    # All four scenarios supported:
    #   LONG  scenarios: demand_bounce (price holds demand), supply_break (price breaks above supply)
    #   SHORT scenarios: supply_bounce (price fails at supply), demand_break (price breaks below demand)
    #
    # Zone TP = next zone in the direction of the trade.
    # Zone SL = edge of the triggering zone on the wrong side.
    zone_tp1 = zone_tp2 = zone_sl = None
    zone_tp1_dist = zone_tp2_dist = zone_sl_dist = zone_rr = None
    zone_type_str  = None
    zone_strength_val = None
    zone_context_str  = zones.price_context if zones else "unknown"

    if zones:
        nd = zones.nearest_demand   # demand zone closest below price
        ns = zones.nearest_supply   # supply zone closest above price
        ctx = zones.price_context

        if is_long:
            if ctx == "at_demand" and nd:
                # Scenario 1: Bounce off demand — buy the bounce
                # TP  = nearest supply zone above (if any), else second supply zone
                # SL  = just below the demand zone's lower edge
                zone_type_str = "demand_bounce"
                zone_strength_val = nd.strength
                zone_sl = round(nd.low - atr_dollar * 0.2, 4)    # just below zone low
                # Pick supply zone TP targets
                sup_above = sorted(
                    [z for z in zones.supply_zones if z.level > entry],
                    key=lambda z: z.level,
                )
                zone_tp1 = round(sup_above[0].level, 4) if len(sup_above) > 0 else None
                zone_tp2 = round(sup_above[1].level, 4) if len(sup_above) > 1 else None

            elif ctx in ("between", "at_supply") and ns and ns.level <= entry * 1.02:
                # Scenario 4: Break above supply — buy the breakout above supply zone
                # TP  = next supply zone higher up
                # SL  = back below the broken supply zone (now demand)
                zone_type_str = "supply_break"
                zone_strength_val = ns.strength
                zone_sl = round(ns.low - atr_dollar * 0.2, 4)
                sup_above = sorted(
                    [z for z in zones.supply_zones if z.level > ns.level + atr_dollar],
                    key=lambda z: z.level,
                )
                zone_tp1 = round(sup_above[0].level, 4) if len(sup_above) > 0 else None
                zone_tp2 = round(sup_above[1].level, 4) if len(sup_above) > 1 else None

        else:  # short
            if ctx == "at_supply" and ns:
                # Scenario 3: Bounce off supply — sell the rejection
                # TP  = nearest demand zone below
                # SL  = just above the supply zone's upper edge
                zone_type_str = "supply_bounce"
                zone_strength_val = ns.strength
                zone_sl = round(ns.high + atr_dollar * 0.2, 4)
                dem_below = sorted(
                    [z for z in zones.demand_zones if z.level < entry],
                    key=lambda z: z.level, reverse=True,
                )
                zone_tp1 = round(dem_below[0].level, 4) if len(dem_below) > 0 else None
                zone_tp2 = round(dem_below[1].level, 4) if len(dem_below) > 1 else None

            elif ctx in ("between", "at_demand") and nd and nd.level >= entry * 0.98:
                # Scenario 2: Break below demand — sell the breakdown through demand
                # TP  = next demand zone lower down
                # SL  = just above the broken demand zone (now resistance)
                zone_type_str = "demand_break"
                zone_strength_val = nd.strength
                zone_sl = round(nd.high + atr_dollar * 0.2, 4)
                dem_below = sorted(
                    [z for z in zones.demand_zones if z.level < nd.level - atr_dollar],
                    key=lambda z: z.level, reverse=True,
                )
                zone_tp1 = round(dem_below[0].level, 4) if len(dem_below) > 0 else None
                zone_tp2 = round(dem_below[1].level, 4) if len(dem_below) > 1 else None

        # Compute zone distances and R:R
        if zone_tp1 is not None and zone_sl is not None:
            if is_long:
                zone_tp1_dist = round((zone_tp1 - entry) / entry * 100, 2)
                zone_tp2_dist = round((zone_tp2 - entry) / entry * 100, 2) if zone_tp2 else None
                zone_sl_dist  = round((entry - zone_sl) / entry * 100, 2)
            else:
                zone_tp1_dist = round((entry - zone_tp1) / entry * 100, 2)
                zone_tp2_dist = round((entry - zone_tp2) / entry * 100, 2) if zone_tp2 else None
                zone_sl_dist  = round((zone_sl - entry) / entry * 100, 2)

            zone_risk   = abs(entry - zone_sl)
            zone_reward = abs(zone_tp1 - entry)
            zone_rr = round(zone_reward / zone_risk, 2) if zone_risk > 0 else None

    # Distances (MC-based TP)
    if is_long:
        sl_atr_dist = round((entry - sl_atr) / entry * 100, 2)
        sl_pct_dist = round((entry - sl_pct) / entry * 100, 2)
        tp1_dist    = round((tp1 - entry) / entry * 100, 2)
        tp2_dist    = round((tp2 - entry) / entry * 100, 2)
    else:
        sl_atr_dist = round((sl_atr - entry) / entry * 100, 2)
        sl_pct_dist = round((sl_pct - entry) / entry * 100, 2)
        tp1_dist    = round((entry - tp1) / entry * 100, 2)
        tp2_dist    = round((entry - tp2) / entry * 100, 2)

    # Risk:Reward
    # Use the TIGHTER (smaller risk = better R:R) stop as the primary R:R.
    # The tighter stop is whichever gives a higher price for longs (or lower for shorts).
    risk_atr = abs(entry - sl_atr)
    risk_pct = abs(entry - sl_pct)
    reward1  = abs(tp1 - entry)

    rr_atr = round(reward1 / risk_atr, 2) if risk_atr > 0 else 0.0
    rr_pct = round(reward1 / risk_pct, 2) if risk_pct > 0 else 0.0

    # Best R:R = the better of the two (tighter stop wins)
    best_rr = max(rr_atr, rr_pct)

    # ── Gate 7: Minimum R:R ──────────────────────────────────────────
    # Gate uses best_rr — if at least one SL method gives acceptable R:R, allow entry.
    if best_rr < _min_rr():
        return TradeSetup(
            ticker=ticker, side="none", valid=False, direction=direction, regime=regime,
            score=round(score, 4), confidence=round(confidence, 3),
            reason=f"R:R too low — ATR:{rr_atr:.1f}R, Fixed:{rr_pct:.1f}R (need ≥{_min_rr()}R). TP targets not far enough.",
        )

    # ── Probabilities of hitting TP / SL (path-aware) ───────────────
    # We use mc_paths (full price paths) when available so we can detect
    # stop-hunts: cases where SL is touched BEFORE price reaches TP.
    # Using only final prices would under-estimate the SL-hit rate on
    # volatile intraday setups.
    mc_paths_arr = mc_final_prices  # may be list-of-paths or list-of-finals
    have_paths   = mc_paths_arr is not None and len(mc_paths_arr) >= 10

    if have_paths:
        arr = np.asarray(mc_paths_arr, dtype=float)
        # Detect shape: if 2-D → full paths; if 1-D → just final prices
        if arr.ndim == 2:
            # arr shape: (n_sim, n_steps)
            # For each path check: does price touch SL before TP?
            if is_long:
                touched_tp1    = np.any(arr >= tp1,    axis=1)
                touched_tp2    = np.any(arr >= tp2,    axis=1)
                touched_sl_atr = np.any(arr <= sl_atr, axis=1)
                touched_sl_pct = np.any(arr <= sl_pct, axis=1)
                # Stop-hunted: SL hit AND TP not hit first
                # (approximate: if both triggered, assume the one hit first was whichever
                # occurred at the earlier step index)
                sl_atr_first = np.argmax(arr <= sl_atr, axis=1)  # first step hitting SL
                tp1_first    = np.argmax(arr >= tp1,    axis=1)
                # hits SL before TP1 (or never hits TP1 at all)
                sh_atr = touched_sl_atr & (~touched_tp1 | (sl_atr_first < tp1_first))
                sl_atr_first2 = np.argmax(arr <= sl_pct, axis=1)
                sh_pct = touched_sl_pct & (~touched_tp1 | (sl_atr_first2 < tp1_first))
            else:
                touched_tp1    = np.any(arr <= tp1,    axis=1)
                touched_tp2    = np.any(arr <= tp2,    axis=1)
                touched_sl_atr = np.any(arr >= sl_atr, axis=1)
                touched_sl_pct = np.any(arr >= sl_pct, axis=1)
                sl_atr_first = np.argmax(arr >= sl_atr, axis=1)
                tp1_first    = np.argmax(arr <= tp1,    axis=1)
                sh_atr = touched_sl_atr & (~touched_tp1 | (sl_atr_first < tp1_first))
                sl_atr_first2 = np.argmax(arr >= sl_pct, axis=1)
                sh_pct = touched_sl_pct & (~touched_tp1 | (sl_atr_first2 < tp1_first))

            prob_tp1    = round(float(np.mean(touched_tp1)), 3)
            prob_tp2    = round(float(np.mean(touched_tp2)), 3)
            prob_sl_atr = round(float(np.mean(sh_atr)),      3)   # stop-hunted before TP
            prob_sl_pct = round(float(np.mean(sh_pct)),      3)
        else:
            # Only final prices — use them directly
            finals = arr
            if is_long:
                prob_tp1    = round(float(np.mean(finals >= tp1)),    3)
                prob_tp2    = round(float(np.mean(finals >= tp2)),    3)
                prob_sl_atr = round(float(np.mean(finals <= sl_atr)), 3)
                prob_sl_pct = round(float(np.mean(finals <= sl_pct)), 3)
            else:
                prob_tp1    = round(float(np.mean(finals <= tp1)),    3)
                prob_tp2    = round(float(np.mean(finals <= tp2)),    3)
                prob_sl_atr = round(float(np.mean(finals >= sl_atr)), 3)
                prob_sl_pct = round(float(np.mean(finals >= sl_pct)), 3)
    else:
        # No MC data — estimate from score and confidence
        _dir_prob = mc_dir_prob if mc_available else min(0.65, 0.40 + abs(score) * 0.5)
        prob_tp1    = round(_dir_prob * 0.75, 3)
        prob_tp2    = round(_dir_prob * 0.45, 3)
        prob_sl_atr = round((1 - _dir_prob) * 0.55, 3)
        prob_sl_pct = round((1 - _dir_prob) * 0.40, 3)

    # Which SL has better R:R? (tighter stop = better R:R)
    sl_recommended = "pct" if rr_pct >= rr_atr else "atr"

    # ── Position sizing ──────────────────────────────────────────────
    # Kelly criterion: f = (p·b − q) / b
    #   p = probability of winning (prob_tp1 from MC)
    #   q = 1 - p
    #   b = win/loss ratio: reward / risk  (use best R:R)
    # Fixed-fractional: risk 1% of equity per trade, size = 1% / SL%
    kelly_fraction = kelly_half = fixed_frac = None
    p_win = prob_tp1 if prob_tp1 is not None else (mc_dir_prob if mc_available else 0.5)
    p_win = float(np.clip(p_win, 0.01, 0.99))
    p_lose = 1.0 - p_win
    best_rr_for_kelly = best_rr if best_rr > 0 else 1.0

    raw_kelly = (p_win * best_rr_for_kelly - p_lose) / best_rr_for_kelly
    if raw_kelly > 0:
        kelly_fraction = round(float(np.clip(raw_kelly, 0.0, 1.0)) * 100, 2)
        kelly_half     = round(kelly_fraction / 2.0, 2)
    else:
        kelly_fraction = 0.0
        kelly_half     = 0.0

    # Fixed-fractional: risk exactly 1% of account equity
    # position_size_pct = risk_fraction / stop_loss_fraction
    # (if SL is 2%, and we risk 1% equity → we put 50% of equity in)
    sl_dist_for_sizing = min(sl_atr_dist, sl_pct_dist) / 100.0  # use tighter stop
    if sl_dist_for_sizing > 0:
        fixed_frac = round(min(0.01 / sl_dist_for_sizing, 1.0) * 100, 2)  # as % of equity
    else:
        fixed_frac = None

    # ── Build reason string ──────────────────────────────────────────
    direction_label = (direction or "unknown").replace("_", " ").title()
    cap_note  = f" [ATR capped at {max_atr_pct*100:.0f}%]" if atr_was_capped else ""
    zone_note = (
        f" · Zone: {zone_type_str.replace('_', ' ')} (str={zone_strength_val:.2f})"
        if zone_type_str is not None and zone_strength_val is not None else ""
    )
    reason_parts = [
        f"{direction_label} confirmed",
        f"ADX {adx:.0f}",
        f"RSI {rsi:.0f}",
        f"Regime: {regime.replace('_',' ')}",
        f"MC {side} prob {mc_dir_prob:.0%}",
        f"Best R:R {best_rr:.1f}R ({sl_recommended.upper()} stop){cap_note}",
    ]
    reason = " · ".join(reason_parts) + zone_note

    return TradeSetup(
        ticker       = ticker,
        side         = side,
        valid        = True,
        direction    = direction,
        regime       = regime,
        score        = round(score,      4),
        confidence   = round(confidence, 3),
        reason       = reason,

        entry        = entry,
        sl_atr       = sl_atr,
        sl_pct       = sl_pct,
        sl_atr_dist  = sl_atr_dist,
        sl_pct_dist  = sl_pct_dist,

        tp1          = tp1,
        tp2          = tp2,
        tp1_dist     = tp1_dist,
        tp2_dist     = tp2_dist,

        rr_atr       = rr_atr,
        rr_pct       = rr_pct,

        prob_tp1     = prob_tp1,
        prob_tp2     = prob_tp2,
        prob_sl_atr  = prob_sl_atr,
        prob_sl_pct  = prob_sl_pct,

        atr_mult       = atr_mult,
        sl_pct_used    = round(sl_pct_val * 100, 2),
        atr_capped     = atr_was_capped,
        sl_recommended = sl_recommended,

        # Zone-based levels (shown alongside MC targets)
        zone_tp1       = zone_tp1,
        zone_tp2       = zone_tp2,
        zone_sl        = zone_sl,
        zone_tp1_dist  = zone_tp1_dist,
        zone_tp2_dist  = zone_tp2_dist,
        zone_sl_dist   = zone_sl_dist,
        zone_rr        = zone_rr,
        zone_type      = zone_type_str,
        zone_strength  = zone_strength_val,
        zone_context   = zone_context_str,

        # Position sizing
        kelly_fraction = kelly_fraction,
        kelly_half     = kelly_half,
        fixed_frac     = fixed_frac,
    )


# ─── Convenience wrapper for the main analyse() output ───────────────────────

def trade_setup_from_analysis(
    ticker:         str,
    result:         dict,
    interval:       str = "15m",
    df:             Optional[pd.DataFrame] = None,   # OHLCV for zone detection
    mc_paths_full:  Optional[object] = None,         # numpy array (n_sim, n_steps+1)
) -> dict:
    """
    Build a TradeSetup directly from the dict returned by core.analyse().
    Pass df (the same DataFrame used for analysis) to enable zone-based TP/SL.
    Returns a JSON-serialisable dict.
    """
    price      = result.get("current_price", 0.0)
    indicators = result.get("indicators", {})
    regime_d   = result.get("regime",     {})
    signal_d   = result.get("signal",     {})
    mc_d       = result.get("mc",         {})

    score      = float(signal_d.get("composite",  0.0))
    confidence = float(signal_d.get("confidence", 0.0))
    rsi        = float(indicators.get("rsi",      50.0))
    adx        = float(indicators.get("adx",      0.0))
    atr_pct    = float(indicators.get("atr_pct",  0.015))
    regime     = str(regime_d.get("regime",       "unknown"))
    direction  = str(signal_d.get("label",        "neutral")).lower().replace(" ", "_")

    # MC probs come back as percentages (0–100) from analyse()
    prob_up    = float(mc_d.get("prob_up",   0.0))
    prob_down  = float(mc_d.get("prob_down", 0.0))
    mc_p10     = mc_d.get("p10_price") or mc_d.get("p10")
    mc_p25     = mc_d.get("p25_price") or mc_d.get("p25")
    mc_p75     = mc_d.get("p75_price") or mc_d.get("p75")
    mc_p90     = mc_d.get("p90_price") or mc_d.get("p90")
    # Prefer the full numpy paths array (all simulations) for accurate probabilities.
    # Fall back to the chart-sample list only if full paths not provided.
    mc_paths   = mc_paths_full if mc_paths_full is not None else mc_d.get("paths") or None

    label_lower = signal_d.get("label", "").lower()
    if "strong buy" in label_lower or "buy" in label_lower:
        direction = "trending_up" if "trend" in regime else "breakout_up"
    elif "strong sell" in label_lower or "sell" in label_lower:
        direction = "trending_down" if "trend" in regime else "breakout_down"
    elif "breakout" in regime:
        direction = regime
    else:
        direction = "neutral"

    if "down" in direction or "sell" in label_lower:
        score = -abs(score)
    else:
        score = abs(score) if abs(score) > 0.01 else score

    ts = compute_trade_setup(
        ticker          = ticker,
        price           = price,
        score           = score,
        direction       = direction,
        regime          = regime,
        confidence      = confidence,
        rsi             = rsi,
        adx             = adx,
        atr_pct         = atr_pct,
        prob_up         = prob_up,
        prob_down       = prob_down,
        mc_p10          = mc_p10,
        mc_p25          = mc_p25,
        mc_p75          = mc_p75,
        mc_p90          = mc_p90,
        mc_final_prices = mc_paths,
        interval        = interval,
        df              = df,
    )
    out = to_dict(ts)

    # ── Structural price targets (Fib + S/R confluence) ─────────────
    # Computed independently of whether a trade is valid, so the
    # dashboard always has max_high / max_downside / fib to show.
    if df is not None and len(df) >= 25 and price > 0:
        try:
            from .levels import compute_price_targets
            zones_for_levels = detect_zones(df) if df is not None else None
            targets = compute_price_targets(
                df            = df,
                current_price = price,
                atr_pct       = atr_pct,
                zones         = zones_for_levels,
            )
            out.update(targets)   # adds max_high, max_downside, fib keys
        except Exception as _lvl_err:
            import logging
            logging.getLogger(__name__).warning("levels compute failed: %s", _lvl_err)

    return out


# ─── Convenience wrapper for scanner ScanResult ───────────────────────────────

def trade_setup_from_scan(
    scan_result: dict,
    mc_data:    Optional[dict] = None,
    interval:   str = "1d",
    df:         Optional[pd.DataFrame] = None,   # OHLCV for zone detection
) -> dict:
    """
    Build a TradeSetup from a scanner result dict (from scan_tickers()).
    mc_data is optional — if provided it should have p10/p25/p75/p90 keys.
    Without it, TP is estimated from ATR-based projections.
    Pass df to enable demand/supply zone-based TP/SL targets.
    """
    price      = float(scan_result.get("price",       0.0))
    score      = float(scan_result.get("score",       0.0))
    direction  = str(scan_result.get("direction",     "neutral"))
    regime     = str(scan_result.get("regime",        "unknown"))
    confidence = float(scan_result.get("confidence",  0.0))
    rsi        = float(scan_result.get("rsi",         50.0))
    adx        = float(scan_result.get("adx",         0.0))
    atr_pct    = float(scan_result.get("atr_pct",     0.015))
    interval   = str(scan_result.get("interval",      interval))
    # prob_up/prob_down are None in scanner results (no MC run).
    # Pass 0.0 so the mc_available gate in compute_trade_setup is skipped.
    raw_prob_up   = scan_result.get("prob_up")
    raw_prob_down = scan_result.get("prob_down")
    prob_up   = float(raw_prob_up)   if raw_prob_up   is not None else 0.0
    prob_down = float(raw_prob_down) if raw_prob_down is not None else 0.0
    ticker     = scan_result.get("ticker", "")

    # Per-timeframe maximum TP distance from entry.
    # These are realistic MC-horizon caps: on a 15m chart the MC runs N candles
    # forward (e.g. 10 bars = 150 min), so a 100% up-move is impossible.
    # Without these caps ATR×2.5 math produces $199 TP for a $99 stock on 15m.
    _MAX_TP1_PCT = {
        "1m":  0.04,   # 4%  TP1 / 8%  TP2
        "2m":  0.05,
        "5m":  0.06,
        "15m": 0.08,   # 8%  TP1 / 14% TP2 on 15m — ~10 bars horizon
        "30m": 0.10,
        "1h":  0.12,
        "2h":  0.15,
        "4h":  0.18,
        "1d":  0.25,   # 25% TP1 / 40% TP2 on daily
        "1wk": 0.40,
        "1mo": 0.60,
    }
    _tp1_cap = _MAX_TP1_PCT.get(interval, 0.10)
    _tp2_cap = _tp1_cap * 1.75   # TP2 allowed ~1.75× TP1 max

    mc_p10 = mc_p25 = mc_p75 = mc_p90 = None
    if mc_data:
        mc_p10 = mc_data.get("p10")
        mc_p25 = mc_data.get("p25")
        mc_p75 = mc_data.get("p75")
        mc_p90 = mc_data.get("p90")
    else:
        # Estimate TP from ATR-based distance — realistic, not from raw potential%.
        # potential_up is a relative score (0-100), NOT a price-target %; we ignore it
        # for TP estimation to avoid huge numbers. Instead we use ATR × modest mult,
        # then hard-cap at the per-timeframe max percentage above.
        atr_d = price * atr_pct if atr_pct > 0 else price * 0.015

        # Typical realistic TP distance: 1.5× ATR for TP1, 2.5× ATR for TP2.
        # (ATR already captures the "normal move" for that candle size.)
        raw_tp1_up   = atr_d * 1.5
        raw_tp2_up   = atr_d * 2.5
        raw_tp1_down = atr_d * 1.5
        raw_tp2_down = atr_d * 2.5

        # Cap so TP1 ≤ tp1_cap% above entry, TP2 ≤ tp2_cap%
        max_tp1_dist = price * _tp1_cap
        max_tp2_dist = price * _tp2_cap
        tp1_up   = min(raw_tp1_up,   max_tp1_dist)
        tp2_up   = min(raw_tp2_up,   max_tp2_dist)
        tp1_down = min(raw_tp1_down, max_tp1_dist)
        tp2_down = min(raw_tp2_down, max_tp2_dist)

        mc_p75 = round(price + tp1_up,   4)
        mc_p90 = round(price + tp2_up,   4)
        mc_p25 = round(price - tp1_down, 4)
        mc_p10 = round(price - tp2_down, 4)

    ts = compute_trade_setup(
        ticker    = ticker,
        price     = price,
        score     = score,
        direction = direction,
        regime    = regime,
        confidence= confidence,
        rsi       = rsi,
        adx       = adx,
        atr_pct   = atr_pct,
        prob_up   = prob_up,
        prob_down = prob_down,
        mc_p10    = mc_p10,
        mc_p25    = mc_p25,
        mc_p75    = mc_p75,
        mc_p90    = mc_p90,
        interval  = interval,
        df        = df,
    )
    return to_dict(ts)
