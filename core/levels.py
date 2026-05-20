"""
core/levels.py
══════════════
Fibonacci retracement / extension analysis and structural price targets.

Given an OHLCV DataFrame this module:
  1.  Identifies the most significant recent swing high and low using a
      pivot-window scan (same style as zones.py).
  2.  Computes the standard Fibonacci retracement levels within that range:
      23.6 %, 38.2 %, 50 %, 61.8 %, 78.6 %.
  3.  Computes Fibonacci extension levels beyond the swing endpoints:
      127.2 %, 161.8 %, 200 %, 261.8 %.
  4.  Scores every candidate level for confluence:
        +2.0  at a demand/supply zone (within 0.5 % of price)
        +1.5  key Fibonacci ratio (38.2, 50, 61.8, 78.6 %)
        +0.5  near an ATR-multiple from current price
        +0.25 standard Fibonacci ratio (23.6 %)
  5.  Returns:
        max_high     – highest-confluence resistance level above price
        max_downside – highest-confluence support level below price
        fib          – all computed Fibonacci levels for chart overlay

Research basis (see README § Estimating max downside / max high):
  • Fibonacci retracements alone are statistically similar to random.
    Confluence with S/R zones improves timing accuracy to ~70 %.
  • The 61.8 % (Golden Ratio) level carries the highest empirical weight.
  • High-volume S/R zones add 10–16 pp of precision over classical methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

_RETRACEMENTS: List[float] = [0.236, 0.382, 0.500, 0.618, 0.786]
_EXTENSIONS:   List[float] = [1.000, 1.272, 1.618, 2.000, 2.618]

# Ratios with the highest empirical confluence weight
_KEY_FIBS: frozenset = frozenset({0.382, 0.500, 0.618, 0.786})

# A level is "at" a zone if it is within this fraction of current price
_ZONE_PROX_PCT: float = 0.005   # 0.5 %

# Labels shown in the UI
_FIB_LABELS = {
    0.236: "23.6%",
    0.382: "38.2%",
    0.500: "50.0%",
    0.618: "61.8% ✦",   # golden ratio highlight
    0.786: "78.6%",
    1.000: "100%",
    1.272: "127.2%",
    1.618: "161.8% ✦",
    2.000: "200%",
    2.618: "261.8%",
}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FibLevels:
    """All Fibonacci levels derived from the most recent significant swing."""
    swing_high: float
    swing_low:  float
    # Retracements (count down from swing_high)
    r_236:  float
    r_382:  float
    r_500:  float
    r_618:  float
    r_786:  float
    # Extensions above swing_high (upside targets)
    e_1000: float
    e_1272: float
    e_1618: float
    e_2000: float
    e_2618: float
    # Extensions below swing_low (downside targets beyond the range)
    d_1272: float
    d_1618: float
    d_2618: float


@dataclass
class StructuralTarget:
    """A single max-high or max-downside price target with confluence metadata."""
    price:          float
    pct_from_price: float           # positive = above current price
    method:         str             # "fib_zone" | "fib" | "zone" | "atr"
    confluence:     str             # human-readable label for the UI
    fib_ratio:      Optional[float] = None
    zone_price:     Optional[float] = None
    score:          float           = 0.0


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _find_swing(df: pd.DataFrame, window: int = 8) -> Tuple[float, float]:
    """
    Return (swing_high, swing_low) from the most significant pivot pair in df.

    Algorithm:
      • Scan the series for local pivot highs and lows using a ±window bar
        look-around.
      • Of the pivot highs found, pick the highest; of the lows, pick the
        lowest.  This gives the widest swing in the lookback window —
        the reference range Fibonacci traders actually use.
      • Fall back to the raw series max/min when fewer than 3× window bars
        are available.
    """
    n = len(df)
    if n == 0:
        return 0.0, 0.0

    highs = df["high"].to_numpy(dtype=float)
    lows  = df["low"].to_numpy(dtype=float)

    if n < window * 3:
        return float(np.nanmax(highs)), float(np.nanmin(lows))

    pivot_highs: List[float] = []
    pivot_lows:  List[float] = []

    for i in range(window, n - window):
        lo_idx = max(0, i - window)
        hi_idx = i + window + 1
        if highs[i] >= np.max(highs[lo_idx:hi_idx]):
            pivot_highs.append(float(highs[i]))
        if lows[i] <= np.min(lows[lo_idx:hi_idx]):
            pivot_lows.append(float(lows[i]))

    sh = max(pivot_highs) if pivot_highs else float(np.nanmax(highs))
    sl = min(pivot_lows)  if pivot_lows  else float(np.nanmin(lows))
    return sh, sl


def _build_fib_levels(swing_high: float, swing_low: float) -> FibLevels:
    rng = swing_high - swing_low
    if rng <= 0:
        rng = max(abs(swing_high) * 0.01, 1e-6)

    def retrace(r: float) -> float:
        return round(swing_high - rng * r, 4)

    def ext_up(r: float) -> float:
        return round(swing_low + rng * r, 4)

    def ext_dn(r: float) -> float:
        return round(swing_high - rng * r, 4)

    return FibLevels(
        swing_high = round(swing_high, 4),
        swing_low  = round(swing_low,  4),
        r_236  = retrace(0.236),
        r_382  = retrace(0.382),
        r_500  = retrace(0.500),
        r_618  = retrace(0.618),
        r_786  = retrace(0.786),
        e_1000 = ext_up(1.000),
        e_1272 = ext_up(1.272),
        e_1618 = ext_up(1.618),
        e_2000 = ext_up(2.000),
        e_2618 = ext_up(2.618),
        d_1272 = ext_dn(1.272),
        d_1618 = ext_dn(1.618),
        d_2618 = ext_dn(2.618),
    )


def _score_level(
    level:         float,
    current_price: float,
    atr_dollar:    float,
    fib_ratio:     Optional[float],
    zone_prices:   List[float],
) -> float:
    """Confluence score for one candidate price level."""
    score = 0.0

    # Zone proximity (+2.0 per zone hit)
    prox = _ZONE_PROX_PCT * current_price
    for zp in zone_prices:
        if abs(level - zp) <= prox:
            score += 2.0
            break

    # Fibonacci quality
    if fib_ratio is not None:
        score += 1.5 if fib_ratio in _KEY_FIBS else 0.25

    # ATR-multiple proximity (+0.5 when level sits within 25 % of one ATR
    # of a round ATR multiple from current price)
    if atr_dollar > 0:
        tolerance = atr_dollar * 0.25
        for mult in (1.0, 1.5, 2.0, 2.5, 3.0):
            for base in (current_price + mult * atr_dollar,
                         current_price - mult * atr_dollar):
                if abs(level - base) <= tolerance:
                    score += 0.5
                    break

    return round(score, 2)


def _confluence_label(
    fib_ratio:  Optional[float],
    zone_price: Optional[float],
    method:     str,
) -> str:
    parts: List[str] = []
    if fib_ratio is not None:
        parts.append(_FIB_LABELS.get(fib_ratio, f"{fib_ratio*100:.1f}%") + " Fib")
    if zone_price is not None:
        parts.append("S/R zone")
    if not parts:
        parts.append("2× ATR" if method == "atr" else "S/R zone")
    return " + ".join(parts)


def _best_target(
    candidates:    List[Tuple[float, Optional[float]]],  # (price, fib_ratio)
    zone_pool:     List[float],
    current_price: float,
    atr_dollar:    float,
    is_above:      bool,
) -> StructuralTarget:
    """
    Pick the highest-confluence candidate.  Falls back to:
      1. Nearest zone in the correct direction.
      2. 2× ATR from current price.
    """
    scored: List[Tuple[float, float, Optional[float]]] = []

    for lvl, fib_r in candidates:
        if not np.isfinite(lvl) or lvl <= 0:
            continue
        s = _score_level(lvl, current_price, atr_dollar, fib_r, zone_pool)
        scored.append((s, lvl, fib_r))

    # Sort: highest score first; break ties by proximity to current price
    scored.sort(key=lambda x: (-x[0], abs(x[1] - current_price)))

    if scored:
        best_score, best_lvl, best_fib = scored[0]
        prox = _ZONE_PROX_PCT * current_price
        zone_hit = next(
            (z for z in zone_pool if abs(best_lvl - z) <= prox), None
        )
        method = (
            "fib_zone" if best_fib is not None and zone_hit is not None
            else "fib"  if best_fib is not None
            else "zone"
        )
        return StructuralTarget(
            price          = round(best_lvl, 4),
            pct_from_price = round((best_lvl - current_price) / current_price * 100, 2),
            method         = method,
            confluence     = _confluence_label(best_fib, zone_hit, method),
            fib_ratio      = best_fib,
            zone_price     = zone_hit,
            score          = best_score,
        )

    # Fallback 1: nearest zone
    valid_zones = [z for z in zone_pool
                   if (z > current_price if is_above else z < current_price)]
    if valid_zones:
        zp = min(valid_zones, key=lambda z: abs(z - current_price))
        return StructuralTarget(
            price          = round(zp, 4),
            pct_from_price = round((zp - current_price) / current_price * 100, 2),
            method         = "zone",
            confluence     = "S/R zone",
            score          = 1.0,
        )

    # Fallback 2: 2× ATR
    sign    = 1 if is_above else -1
    fallback = round(current_price + sign * 2 * atr_dollar, 4)
    return StructuralTarget(
        price          = fallback,
        pct_from_price = round((fallback - current_price) / current_price * 100, 2),
        method         = "atr",
        confluence     = "2× ATR",
        score          = 0.0,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_price_targets(
    df:            pd.DataFrame,
    current_price: float,
    atr_pct:       float,
    zones          = None,       # ZoneResult | None
    swing_window:  int = 8,
) -> dict:
    """
    Compute max_downside and max_high structural price targets.

    Parameters
    ──────────
    df            : OHLCV DataFrame (the same one passed to analyse())
    current_price : latest close price
    atr_pct       : ATR as a fraction of price (e.g. 0.015 = 1.5 %)
    zones         : ZoneResult from detect_zones() — optional but improves accuracy
    swing_window  : pivot look-around bars for swing detection

    Returns
    ───────
    dict with three keys, all JSON-serialisable:
      "max_high"     : StructuralTarget fields
      "max_downside" : StructuralTarget fields
      "fib"          : FibLevels fields (all level prices + swing anchors)
    """
    if len(df) < swing_window * 3 or current_price <= 0:
        return {}

    atr_dollar = current_price * max(atr_pct, 0.005)

    # ── 1. Swing detection ───────────────────────────────────────────
    swing_high, swing_low = _find_swing(df, window=swing_window)

    # Clamp so levels stay useful even after a breakout / breakdown
    swing_high = max(swing_high, current_price * 1.001)
    swing_low  = min(swing_low,  current_price * 0.999)

    fib = _build_fib_levels(swing_high, swing_low)
    rng = swing_high - swing_low

    # ── 2. Collect zone prices ────────────────────────────────────────
    supply_prices: List[float] = []
    demand_prices: List[float] = []
    if zones is not None:
        supply_prices = [
            z.level for z in (zones.supply_zones or [])
            if z.level > current_price * 0.97
        ]
        demand_prices = [
            z.level for z in (zones.demand_zones or [])
            if z.level < current_price * 1.03
        ]

    # ── 3. Candidate levels above current price (max high) ────────────
    above: List[Tuple[float, Optional[float]]] = []

    # Retracement levels that happen to sit above current price
    # (can occur when price pulled back below the midpoint)
    for ratio in _RETRACEMENTS:
        lvl = round(swing_high - rng * ratio, 4)
        if lvl > current_price:
            above.append((lvl, ratio))

    # Extension levels above swing_high (the natural upside targets)
    for ratio in _EXTENSIONS:
        lvl = round(swing_low + rng * ratio, 4)
        if lvl > current_price:
            above.append((lvl, ratio))

    # Pure zone candidates above price
    for zp in supply_prices:
        if zp > current_price:
            above.append((zp, None))

    # ── 4. Candidate levels below current price (max downside) ────────
    below: List[Tuple[float, Optional[float]]] = []

    # Retracement levels below current price (the natural pullback targets)
    for ratio in _RETRACEMENTS:
        lvl = round(swing_high - rng * ratio, 4)
        if lvl < current_price:
            below.append((lvl, ratio))

    # Extension levels below swing_low (deeper downside targets)
    for ratio in _EXTENSIONS:
        lvl = round(swing_high - rng * ratio, 4)
        if lvl < current_price:
            below.append((lvl, ratio))

    # Pure zone candidates below price
    for zp in demand_prices:
        if zp < current_price:
            below.append((zp, None))

    # ── 5. Score and pick ─────────────────────────────────────────────
    max_high     = _best_target(above, supply_prices, current_price, atr_dollar, is_above=True)
    max_downside = _best_target(below, demand_prices, current_price, atr_dollar, is_above=False)

    return {
        "max_high":     asdict(max_high),
        "max_downside": asdict(max_downside),
        "fib":          asdict(fib),
    }
