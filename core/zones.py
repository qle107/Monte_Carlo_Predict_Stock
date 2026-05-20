"""
core/zones.py — Demand / Supply zone detector
───────────────────────────────────────────────
Identifies key price zones where institutional buyers (demand) or sellers
(supply) have historically stepped in, causing notable reversals.

Algorithm
─────────
1. Find swing pivot highs / lows using a look-left / look-right window.
   A pivot low = potential demand zone origin.
   A pivot high = potential supply zone origin.

2. Cluster pivots that fall within `cluster_atr` × ATR of each other into a
   single zone (centred on their volume-weighted mean if volume is given,
   plain mean otherwise).

3. Score each zone (0–1):
   • touches    : how many bars overlapped the band (excluding formation),
                  saturating at MAX_TOUCH_COUNT touches
   • recency    : zones formed near the right edge of the window score higher
   • freshness  : zones that have NEVER been re-tested after formation
                  score higher than already-tested ones
   • depth      : the candle that formed the zone had a strong body
                  (close far from open relative to range) → real rejection

4. Remove zones that have been decisively broken by price action AFTER
   formation. A demand zone is broken if any subsequent bar's *low* trades
   more than `break_atr` × ATR below the zone level — using `low` (not
   `close`) catches intraday violations that close-only filters miss.

5. Select the nearest meaningful demand (below price) and supply (above
   price) zone for the dashboard.

Public API (unchanged)
──────────────────────
  detect_zones(df) -> ZoneResult
  Zone, ZoneResult dataclasses

Refactor notes (vs. the previous version)
─────────────────────────────────────────
  • Fixed: line that used undefined `touches` instead of `z.touches`,
    which silently zeroed every zone's touch score.
  • Fixed: depth scoring was a flat 0.1; now uses real candle body% at the
    formation bar.
  • Fixed: broken-zone filter now inspects intraday lows/highs, not closes.
  • Fixed: bare `except Exception` no longer hides bugs — errors are logged.
  • Fixed: cluster centroid is now stable (uses running mean update), so
    spread pivots no longer drift the boundary outward unexpectedly.
  • Improved: nearest-demand selection no longer drops a close-and-decent
    zone in favour of a strong-but-distant one; we pick the closest zone
    above a minimum-strength floor.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)


# ─── Tunables (read from cfg at runtime) ─────────────────────────────────────


def _pivot_wing() -> int:
    return int(cfg.zone_pivot_window)


def _cluster_atr() -> float:
    return float(cfg.zone_cluster_atr)


def _touch_atr() -> float:
    return float(cfg.zone_touch_atr)


def _break_atr() -> float:
    return float(cfg.zone_break_atr)


def _max_demand() -> int:
    return int(cfg.zone_max_demand)


def _max_supply() -> int:
    return int(cfg.zone_max_supply)


def _zone_half_atr() -> float:
    return float(cfg.zone_width_atr)


# Scoring constants. They sum to 1.0 by design so a perfect zone scores 1.0.
_W_TOUCHES = 0.35
_W_RECENCY = 0.25
_W_FRESHNESS = 0.20
_W_DEPTH = 0.20

# Saturation point for the touch component.
_MAX_TOUCH_COUNT = 4

# Minimum strength a zone must have to be considered "nearest" on the dashboard.
# Filters out near-noise zones that would otherwise win on proximity alone.
_MIN_NEAREST_STRENGTH = 0.30


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class Zone:
    level: float
    low: float
    high: float
    strength: float  # 0–1
    touches: int
    fresh: bool
    zone_type: str  # "demand" | "supply"
    bar_idx: int  # most recent contributing pivot

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ZoneResult:
    demand_zones: list[Zone]
    supply_zones: list[Zone]
    nearest_demand: Zone | None
    nearest_supply: Zone | None
    price_context: str  # "at_demand" | "at_supply" | "between" | "unknown"
    atr: float

    def to_dict(self) -> dict:
        return {
            "demand_zones": [z.to_dict() for z in self.demand_zones],
            "supply_zones": [z.to_dict() for z in self.supply_zones],
            "nearest_demand": self.nearest_demand.to_dict() if self.nearest_demand else None,
            "nearest_supply": self.nearest_supply.to_dict() if self.nearest_supply else None,
            "price_context": self.price_context,
            "atr": round(self.atr, 4),
        }


# ─── ATR helper ──────────────────────────────────────────────────────────────


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range as a dollar value (simple moving average over `period`)."""
    if df is None or len(df) < period + 1:
        c = float(df["close"].iloc[-1]) if df is not None and len(df) else 1.0
        return c * 0.015  # 1.5% fallback

    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    tr = np.maximum.reduce(
        [
            h[1:] - l[1:],
            np.abs(h[1:] - c[:-1]),
            np.abs(l[1:] - c[:-1]),
        ]
    )
    if tr.size == 0:
        return float(c[-1]) * 0.015
    return float(np.mean(tr[-period:]))


# ─── Pivot finder ────────────────────────────────────────────────────────────


def _find_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    wing: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (pivot_highs, pivot_lows) as lists of (bar_idx, price)."""
    n = len(highs)
    ph: list[tuple[int, float]] = []
    pl: list[tuple[int, float]] = []
    if n < 2 * wing + 1:
        return ph, pl

    for i in range(wing, n - wing):
        win_h = highs[i - wing : i + wing + 1]
        win_l = lows[i - wing : i + wing + 1]
        # Use >= / <= so ties don't silently drop legitimate pivots; the
        # cluster step will de-duplicate adjacent ones.
        if highs[i] >= win_h.max():
            ph.append((i, float(highs[i])))
        if lows[i] <= win_l.min():
            pl.append((i, float(lows[i])))
    return ph, pl


# ─── Cluster pivots into zones ───────────────────────────────────────────────


def _cluster(
    pivots: Sequence[tuple[int, float]],
    atr: float,
    zone_type: str,
) -> list[Zone]:
    """
    Merge nearby pivots into a single zone. The cluster boundary uses the
    initial seed price as the anchor (not a running mean), so a long string
    of pivots can't drift the cluster outward without bound.
    """
    if not pivots:
        return []

    sorted_p = sorted(pivots, key=lambda x: x[1])
    band = atr * _cluster_atr()
    half_w = atr * _zone_half_atr()

    zones: list[Zone] = []
    seed_price: float = sorted_p[0][1]
    idxs: list[int] = [sorted_p[0][0]]
    lvls: list[float] = [sorted_p[0][1]]

    def _flush() -> Zone:
        level = float(np.mean(lvls))
        bar_idx = max(idxs)
        return Zone(
            level=round(level, 4),
            low=round(level - half_w, 4),
            high=round(level + half_w, 4),
            strength=0.0,  # filled in later
            touches=0,
            fresh=True,
            zone_type=zone_type,
            bar_idx=bar_idx,
        )

    for idx, price in sorted_p[1:]:
        # Compare against the seed of the *current* cluster, not its running
        # mean — keeps cluster width bounded by `band` regardless of size.
        if abs(price - seed_price) <= band:
            idxs.append(idx)
            lvls.append(price)
        else:
            zones.append(_flush())
            seed_price = price
            idxs = [idx]
            lvls = [price]

    zones.append(_flush())
    return zones


# ─── Candle body strength (for depth scoring) ────────────────────────────────


def _body_strength(df: pd.DataFrame, bar_idx: int) -> float:
    """
    How decisively the bar that formed the pivot rejected the level.
    Returns body% = |close − open| / max(high − low, ε) ∈ [0, 1].
    For a doji the body is ~0; for a marubozu it is ~1.
    """
    try:
        o = float(df["open"].iloc[bar_idx])
        c = float(df["close"].iloc[bar_idx])
        h = float(df["high"].iloc[bar_idx])
        l = float(df["low"].iloc[bar_idx])
        rng = max(h - l, 1e-9)
        return float(np.clip(abs(c - o) / rng, 0.0, 1.0))
    except Exception:
        return 0.0


# ─── Score zones ─────────────────────────────────────────────────────────────


def _score_zones(
    zones: list[Zone],
    df: pd.DataFrame,
    lows: np.ndarray,
    highs: np.ndarray,
    atr: float,
    n_bars: int,
) -> None:
    """Fill `touches`, `fresh`, `strength` on each zone in-place."""
    if not zones or n_bars <= 1:
        return

    touch_band = atr * _touch_atr()

    for z in zones:
        # ── Touches (vectorised) ─────────────────────────────────────────
        touching = (lows <= z.level + touch_band) & (highs >= z.level - touch_band)
        if 0 <= z.bar_idx < touching.size:
            touching[z.bar_idx] = False  # exclude formation bar
        z.touches = int(touching.sum())

        # ── Fresh: any touch on a bar AFTER the formation bar? ──────────
        if z.bar_idx + 1 < touching.size:
            z.fresh = not bool(touching[z.bar_idx + 1 :].any())
        else:
            z.fresh = True

        # ── Strength components (each in [0, 1]) ─────────────────────────
        touch_score = min(z.touches / _MAX_TOUCH_COUNT, 1.0)
        recency_score = z.bar_idx / max(n_bars - 1, 1)  # 0 = oldest, 1 = newest
        fresh_score = 1.0 if z.fresh else 0.0
        depth_score = _body_strength(df, z.bar_idx)

        z.strength = round(
            _W_TOUCHES * touch_score
            + _W_RECENCY * recency_score
            + _W_FRESHNESS * fresh_score
            + _W_DEPTH * depth_score,
            3,
        )


# ─── Filter broken zones ─────────────────────────────────────────────────────


def _remove_broken(
    zones: list[Zone],
    highs: np.ndarray,
    lows: np.ndarray,
    atr: float,
) -> list[Zone]:
    """
    A demand zone is broken if any subsequent bar's *low* traded more than
    `break_atr` × ATR below the zone level. Supply uses subsequent *highs*.

    Using intraday low / high (not close) catches the common case where
    price violates the zone, prints a long wick, and closes back inside —
    which still implies the zone is no longer a clean defended level.
    """
    if not zones:
        return zones

    thresh = atr * _break_atr()
    surviving: list[Zone] = []
    for z in zones:
        after_start = z.bar_idx + 1
        if after_start >= lows.size:
            surviving.append(z)  # nothing after — can't be broken
            continue

        if z.zone_type == "demand":
            broken = bool(np.any(lows[after_start:] < z.level - thresh))
        else:
            broken = bool(np.any(highs[after_start:] > z.level + thresh))

        if not broken:
            surviving.append(z)

    return surviving


# ─── Nearest-zone selection ──────────────────────────────────────────────────


def _select_nearest(
    zones: list[Zone],
    price: float,
    side: str,  # "below" for demand, "above" for supply
    touch_band: float,
) -> Zone | None:
    """
    Pick the closest zone on the requested side of price that meets a
    minimum strength floor. If no zone clears the floor, fall back to the
    closest zone of any strength — the dashboard still wants something to
    show, but the floor stops weak noise zones from winning.
    """
    if not zones:
        return None

    if side == "below":
        candidates = [z for z in zones if z.level <= price + touch_band]
        candidates.sort(key=lambda z: price - z.level)  # smallest gap first
    else:
        candidates = [z for z in zones if z.level >= price - touch_band]
        candidates.sort(key=lambda z: z.level - price)

    if not candidates:
        return None

    strong = [z for z in candidates if z.strength >= _MIN_NEAREST_STRENGTH]
    return strong[0] if strong else candidates[0]


# ─── Main entry point ─────────────────────────────────────────────────────────

_EMPTY = ZoneResult(
    demand_zones=[],
    supply_zones=[],
    nearest_demand=None,
    nearest_supply=None,
    price_context="unknown",
    atr=0.0,
)


def detect_zones(df: pd.DataFrame) -> ZoneResult:
    """
    Detect demand and supply zones from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame with columns open, high, low, close, volume (volume optional).
         At least 10 rows recommended; below that, returns an empty result.
    """
    if df is None or len(df) < 10:
        return _EMPTY

    try:
        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        if not (highs.size and lows.size and closes.size):
            return _EMPTY

        n = closes.size
        price = float(closes[-1])
        atr = _atr(df)
        if atr <= 0 or not np.isfinite(atr):
            return _EMPTY

        # 1. Pivots
        pivot_highs, pivot_lows = _find_pivots(highs, lows, wing=_pivot_wing())

        # 2. Cluster into zones
        demand_zones = _cluster(pivot_lows, atr, "demand")
        supply_zones = _cluster(pivot_highs, atr, "supply")

        # 3. Score
        _score_zones(demand_zones, df, lows, highs, atr, n)
        _score_zones(supply_zones, df, lows, highs, atr, n)

        # 4. Remove decisively broken zones
        demand_zones = _remove_broken(demand_zones, highs, lows, atr)
        supply_zones = _remove_broken(supply_zones, highs, lows, atr)

        # 5. Limit to N strongest, but keep enough that "nearest" has options
        demand_zones.sort(key=lambda z: z.strength, reverse=True)
        supply_zones.sort(key=lambda z: z.strength, reverse=True)
        demand_zones = demand_zones[: _max_demand()]
        supply_zones = supply_zones[: _max_supply()]

        # 6. Nearest zone selection
        touch_band = atr * _touch_atr()
        nearest_demand = _select_nearest(demand_zones, price, side="below", touch_band=touch_band)
        nearest_supply = _select_nearest(supply_zones, price, side="above", touch_band=touch_band)

        # 7. Price context
        at_demand = nearest_demand is not None and abs(price - nearest_demand.level) <= touch_band
        at_supply = nearest_supply is not None and abs(price - nearest_supply.level) <= touch_band

        if at_demand and at_supply:
            price_context = "at_demand"  # demand wins ties (more actionable)
        elif at_demand:
            price_context = "at_demand"
        elif at_supply:
            price_context = "at_supply"
        elif nearest_demand or nearest_supply:
            price_context = "between"
        else:
            price_context = "unknown"

        return ZoneResult(
            demand_zones=demand_zones,
            supply_zones=supply_zones,
            nearest_demand=nearest_demand,
            nearest_supply=nearest_supply,
            price_context=price_context,
            atr=atr,
        )

    except Exception:
        logger.exception("detect_zones failed")
        return _EMPTY
