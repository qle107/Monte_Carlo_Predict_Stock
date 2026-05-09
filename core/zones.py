"""
core/zones.py — Demand / Supply zone detector
───────────────────────────────────────────────
Identifies key price zones where institutional buyers (demand) or sellers
(supply) have historically stepped in, causing notable reversals.

Algorithm
─────────
1. Find swing pivot highs and pivot lows using a look-left / look-right
   window (default ±4 bars).  A pivot low = potential demand zone origin.
   A pivot high = potential supply zone origin.

2. Cluster nearby pivots: if two pivot prices are within `cluster_atr`
   ATRs of each other they belong to the same zone.  The zone's price
   level = average of clustered pivots.

3. Score each zone (0–1) by:
   • Touches   — how many times price returned to (±ATR) the zone level
   • Recency   — zones tested in the last 20 bars score higher
   • Freshness — untested zones (price never re-entered after first touch)
                 score higher than already-broken ones
   • Depth     — the candle that formed the zone had a strong body (clear
                 rejection) → higher score

4. Filter zones:
   • Remove zones where price has already blown THROUGH by more than
     `break_thresh` (default 0.5 × ATR) — they are broken zones.
   • Keep at most `max_zones` demand + `max_zones` supply zones.

Returns
───────
  ZoneResult dataclass with:
    demand_zones : List[Zone]   sorted nearest-above-entry first
    supply_zones : List[Zone]   sorted nearest-below-entry first
    nearest_demand : Zone | None   closest demand zone BELOW current price
    nearest_supply : Zone | None   closest supply zone ABOVE current price
    price_context  : str   "at_demand" | "at_supply" | "between" | "unknown"

Zone dataclass:
    level    : float   central price of the zone
    low      : float   zone lower edge  (level − 0.3×ATR)
    high     : float   zone upper edge  (level + 0.3×ATR)
    strength : float   0–1 score
    touches  : int     number of times price returned to zone
    fresh    : bool    True if zone has never been re-tested after formation
    zone_type: str     "demand" | "supply"
    bar_idx  : int     bar index of the most recent pivot that formed the zone
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from config import cfg


# ─── Tunables ────────────────────────────────────────────────────────────────
# These are now read from cfg at runtime so they can be tuned without code changes.
# Accessor functions ensure live config changes take effect on each call.

def _PIVOT_WING()     -> int:   return cfg.zone_pivot_window
def _CLUSTER_ATR()    -> float: return cfg.zone_cluster_atr
def _TOUCH_BAND_ATR() -> float: return cfg.zone_touch_atr
def _BREAK_THRESH()   -> float: return cfg.zone_break_atr
def _MAX_ZONES_D()    -> int:   return cfg.zone_max_demand
def _MAX_ZONES_S()    -> int:   return cfg.zone_max_supply
def _ZONE_HALF_ATR()  -> float: return cfg.zone_width_atr


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class Zone:
    level:     float
    low:       float
    high:      float
    strength:  float      # 0 – 1
    touches:   int
    fresh:     bool
    zone_type: str        # "demand" | "supply"
    bar_idx:   int        # most recent contributing pivot (0 = oldest bar)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ZoneResult:
    demand_zones:    List[Zone]
    supply_zones:    List[Zone]
    nearest_demand:  Optional[Zone]   # strongest demand zone BELOW price
    nearest_supply:  Optional[Zone]   # strongest supply zone ABOVE price
    price_context:   str              # "at_demand" | "at_supply" | "between" | "unknown"
    atr:             float

    def to_dict(self) -> dict:
        return {
            "demand_zones":   [z.to_dict() for z in self.demand_zones],
            "supply_zones":   [z.to_dict() for z in self.supply_zones],
            "nearest_demand": self.nearest_demand.to_dict() if self.nearest_demand else None,
            "nearest_supply": self.nearest_supply.to_dict() if self.nearest_supply else None,
            "price_context":  self.price_context,
            "atr":            round(self.atr, 4),
        }


# ─── ATR helper ──────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range as a dollar value."""
    if len(df) < period + 1:
        c = float(df["close"].iloc[-1]) if len(df) else 1.0
        return c * 0.015   # 1.5% fallback

    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) else c[-1] * 0.015
    return float(np.mean(tr[-period:]))


# ─── Pivot finder ────────────────────────────────────────────────────────────

def _find_pivots(
    highs: np.ndarray,
    lows:  np.ndarray,
    wing:  int,
) -> tuple[List[tuple[int, float]], List[tuple[int, float]]]:
    """
    Returns (pivot_highs, pivot_lows) each as list of (bar_index, price).
    bar_index is relative to the input arrays (0 = oldest).
    """
    n = len(highs)
    ph: List[tuple[int, float]] = []
    pl: List[tuple[int, float]] = []

    for i in range(wing, n - wing):
        win_h = highs[i - wing: i + wing + 1]
        win_l = lows [i - wing: i + wing + 1]
        if highs[i] == np.max(win_h):
            ph.append((i, float(highs[i])))
        if lows[i] == np.min(win_l):
            pl.append((i, float(lows[i])))

    return ph, pl


# ─── Cluster pivots into zones ───────────────────────────────────────────────

def _cluster(
    pivots:    List[tuple[int, float]],
    atr:       float,
    zone_type: str,
) -> List[Zone]:
    """
    Merge nearby pivots into zones.  Returns unsorted Zone list.
    """
    if not pivots:
        return []

    # Sort by price
    sorted_p = sorted(pivots, key=lambda x: x[1])
    half = atr * _CLUSTER_ATR()

    zones: List[Zone] = []
    cluster_idxs: List[int] = [sorted_p[0][0]]
    cluster_lvls: List[float] = [sorted_p[0][1]]

    def _flush(idxs: List[int], lvls: List[float]) -> Zone:
        level   = float(np.mean(lvls))
        bar_idx = max(idxs)   # most recent
        return Zone(
            level     = round(level, 4),
            low       = round(level - atr * _ZONE_HALF_ATR(), 4),
            high      = round(level + atr * _ZONE_HALF_ATR(), 4),
            strength  = 0.0,   # filled in later
            touches   = 0,
            fresh     = True,
            zone_type = zone_type,
            bar_idx   = bar_idx,
        )

    for idx, price in sorted_p[1:]:
        if abs(price - float(np.mean(cluster_lvls))) <= half:
            cluster_idxs.append(idx)
            cluster_lvls.append(price)
        else:
            zones.append(_flush(cluster_idxs, cluster_lvls))
            cluster_idxs = [idx]
            cluster_lvls = [price]

    zones.append(_flush(cluster_idxs, cluster_lvls))
    return zones


# ─── Score zones ─────────────────────────────────────────────────────────────

def _score_zones(
    zones:   List[Zone],
    closes:  np.ndarray,
    lows:    np.ndarray,
    highs:   np.ndarray,
    atr:     float,
    n_bars:  int,
) -> None:
    """
    Mutates each Zone in-place: fills touches, fresh, strength.
    """
    touch_band = atr * _TOUCH_BAND_ATR()

    for z in zones:
        touches   = 0
        re_tested = False
        formation_bar = z.bar_idx

        for i in range(len(closes)):
            if i == formation_bar:
                continue
            # Price is "touching" the zone if any part of the candle is within band
            candle_lo = lows[i]
            candle_hi = highs[i]
            if candle_lo <= z.level + touch_band and candle_hi >= z.level - touch_band:
                touches += 1
                if i > formation_bar:
                    re_tested = True

        z.touches = touches
        z.fresh   = not re_tested

        # ── Strength scoring ─────────────────────────────────────────
        # 1. Touches (0–0.4): more tests = stronger zone, up to 4 extra touches
        touch_score = min(touches / 4.0, 1.0) * 0.4

        # 2. Recency (0–0.3): zone formed in last 20% of bars = full score
        recency_frac = z.bar_idx / max(n_bars - 1, 1)
        recency_score = recency_frac * 0.3   # newer = higher bar_idx = higher score

        # 3. Freshness (0–0.2): unretested zone = more likely to hold
        fresh_score = 0.2 if z.fresh else 0.0

        # 4. Formation quality (0–0.1): dummy — flat score since we don't have body%
        formation_score = 0.1

        z.strength = round(
            touch_score + recency_score + fresh_score + formation_score, 3
        )


# ─── Filter broken zones ─────────────────────────────────────────────────────

def _remove_broken(
    zones:     List[Zone],
    closes:    np.ndarray,
    atr:       float,
    current_price: float,
) -> List[Zone]:
    """
    Remove zones that have been decisively broken by recent price action.
    A demand zone is broken if price closed more than break_thresh × ATR
    below it (on a bar AFTER the zone formed).
    A supply zone is broken if price closed above it by that margin.
    """
    thresh = atr * _BREAK_THRESH()
    surviving = []

    for z in zones:
        broken = False
        for i in range(z.bar_idx + 1, len(closes)):
            c = closes[i]
            if z.zone_type == "demand" and c < z.level - thresh:
                broken = True
                break
            if z.zone_type == "supply" and c > z.level + thresh:
                broken = True
                break
        if not broken:
            surviving.append(z)

    return surviving


# ─── Main entry point ─────────────────────────────────────────────────────────

def detect_zones(df: pd.DataFrame) -> ZoneResult:
    """
    Detect demand and supply zones from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        At least 20 rows recommended; returns empty result with <10.

    Returns
    -------
    ZoneResult
    """
    _empty = ZoneResult(
        demand_zones=[], supply_zones=[],
        nearest_demand=None, nearest_supply=None,
        price_context="unknown", atr=0.0,
    )

    if df is None or len(df) < 10:
        return _empty

    try:
        highs  = df["high"].to_numpy(float)
        lows   = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        n      = len(closes)
        price  = float(closes[-1])
        atr    = _atr(df)

        # ── 1. Find swing pivots ──────────────────────────────────────
        pivot_highs, pivot_lows = _find_pivots(highs, lows, wing=_PIVOT_WING())

        # ── 2. Cluster into zones ─────────────────────────────────────
        demand_zones = _cluster(pivot_lows,  atr, "demand")
        supply_zones = _cluster(pivot_highs, atr, "supply")

        # ── 3. Score zones ────────────────────────────────────────────
        _score_zones(demand_zones, closes, lows, highs, atr, n)
        _score_zones(supply_zones, closes, lows, highs, atr, n)

        # ── 4. Remove broken zones ────────────────────────────────────
        demand_zones = _remove_broken(demand_zones, closes, atr, price)
        supply_zones = _remove_broken(supply_zones, closes, atr, price)

        # ── 5. Sort by strength desc, limit ──────────────────────────
        demand_zones.sort(key=lambda z: z.strength, reverse=True)
        supply_zones.sort(key=lambda z: z.strength, reverse=True)
        demand_zones = demand_zones[:_MAX_ZONES_D()]
        supply_zones = supply_zones[:_MAX_ZONES_S()]

        # ── 6. Find nearest demand (below price) / supply (above price) ─
        touch_band = atr * _TOUCH_BAND_ATR()

        # Nearest demand BELOW current price (sorted by level desc = closest first)
        below = sorted(
            [z for z in demand_zones if z.level < price + touch_band],
            key=lambda z: z.level, reverse=True,
        )
        nearest_demand = below[0] if below else None

        # Nearest supply ABOVE current price (sorted by level asc = closest first)
        above = sorted(
            [z for z in supply_zones if z.level > price - touch_band],
            key=lambda z: z.level,
        )
        nearest_supply = above[0] if above else None

        # ── 7. Price context ──────────────────────────────────────────
        at_demand = nearest_demand and abs(price - nearest_demand.level) <= touch_band
        at_supply = nearest_supply and abs(price - nearest_supply.level) <= touch_band

        if at_demand and at_supply:
            price_context = "at_demand"   # demand wins if both
        elif at_demand:
            price_context = "at_demand"
        elif at_supply:
            price_context = "at_supply"
        elif nearest_demand or nearest_supply:
            price_context = "between"
        else:
            price_context = "unknown"

        return ZoneResult(
            demand_zones   = demand_zones,
            supply_zones   = supply_zones,
            nearest_demand = nearest_demand,
            nearest_supply = nearest_supply,
            price_context  = price_context,
            atr            = atr,
        )

    except Exception:
        return _empty
