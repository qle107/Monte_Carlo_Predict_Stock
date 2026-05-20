"""
core/volume_profile.py — Volume Profile (Market Profile) analysis.

Computes the volume-at-price distribution from OHLCV candle data.
No external data required — pure OHLCV.

Key outputs
-----------
  poc             : float    Point of Control — highest-volume price level
  value_area_high : float    Top of Value Area (70 % of total volume)
  value_area_low  : float    Bottom of Value Area
  hvn             : list     High Volume Nodes — local peaks → support / resistance
  lvn             : list     Low Volume Nodes  — local troughs → fast-move gaps
  bins            : list     Full histogram [{"price", "volume", "pct", "type"}]
  current_zone    : str      Where is price relative to the VP? "above_va" | "in_va" | "below_va"
  poc_distance_pct: float    How far current price is from POC (% of price)

Theory
------
  HVN: Price spent a lot of time here — buyers/sellers agreed on this level.
       When price revisits → expect BOUNCE or CONSOLIDATION (market memory).

  LVN: Little volume traded here — price passed through quickly.
       When price enters an LVN → expect a FAST directional move (low resistance).

  POC: The "fairest" price per the market — acts as a mean-reversion magnet.

  Value Area: The 70 % of volume band. Price outside VA tends to return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

_N_BINS = 100  # number of price-level buckets
_VALUE_AREA = 0.70  # fraction of volume that defines the Value Area
_HVN_PERCENTILE = 70  # bins above this percentile = HVN
_LVN_PERCENTILE = 30  # bins below this percentile = LVN
_SMOOTH_WINDOW = 3  # Gaussian smooth the histogram before peak-finding


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class VPBin:
    price: float  # centre of the price bucket
    volume: float  # total volume traded in this bucket
    pct: float  # fraction of total volume (0–1)
    type: str  # "hvn" | "lvn" | "poc" | "normal"


@dataclass
class VolumeProfile:
    poc: float
    value_area_high: float
    value_area_low: float
    hvn: list[float]  # price levels
    lvn: list[float]  # price levels
    bins: list[VPBin]
    current_zone: str  # "above_va" | "in_va" | "below_va" | "unknown"
    poc_distance_pct: float  # (price - poc) / poc * 100
    current_price: float

    def to_dict(self) -> dict:
        return {
            "poc": round(self.poc, 4),
            "value_area_high": round(self.value_area_high, 4),
            "value_area_low": round(self.value_area_low, 4),
            "hvn": [round(v, 4) for v in self.hvn],
            "lvn": [round(v, 4) for v in self.lvn],
            "current_zone": self.current_zone,
            "poc_distance_pct": round(self.poc_distance_pct, 3),
            "current_price": round(self.current_price, 4),
            "bins": [
                {
                    "price": round(b.price, 4),
                    "volume": round(b.volume, 0),
                    "pct": round(b.pct, 5),
                    "type": b.type,
                }
                for b in self.bins
            ],
        }


# ─── Core computation ─────────────────────────────────────────────────────────


def compute_volume_profile(
    df: pd.DataFrame,
    n_bins: int = _N_BINS,
) -> VolumeProfile | None:
    """
    Compute Volume Profile from OHLCV DataFrame.

    Volume is distributed uniformly across the high-low range of each candle
    (TPO-style approximation — fine for bars ≥ 1m).

    Returns None if df is too small or malformed.
    """
    try:
        return _compute(df, n_bins)
    except Exception as exc:
        logger.warning("[VP] compute_volume_profile failed: %s", exc)
        return None


def _compute(df: pd.DataFrame, n_bins: int) -> VolumeProfile:
    if df is None or len(df) < 10:
        raise ValueError("Too few bars")

    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    price_lo = float(np.nanmin(lows))
    price_hi = float(np.nanmax(highs))
    if price_hi <= price_lo:
        raise ValueError("Zero price range")

    # Build bin edges and centres
    edges = np.linspace(price_lo, price_hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    bucket = np.zeros(n_bins, dtype=float)
    bin_w = edges[1] - edges[0]

    # Distribute each candle's volume across the price bins it covers
    for h, lo, vol in zip(highs, lows, volumes, strict=False):
        if not np.isfinite(vol) or vol <= 0:
            continue
        lo_c = max(lo, price_lo)
        hi_c = min(h, price_hi)
        if hi_c < lo_c:
            continue
        candle_range = hi_c - lo_c
        for i, c in enumerate(centres):
            # Overlap of bin [c-bw/2, c+bw/2] with candle [lo_c, hi_c]
            overlap = min(c + bin_w / 2, hi_c) - max(c - bin_w / 2, lo_c)
            if overlap <= 0:
                continue
            frac = overlap / (candle_range + 1e-12) if candle_range > 0 else 1.0 / n_bins
            bucket[i] += vol * frac

    total_vol = bucket.sum()
    if total_vol == 0:
        raise ValueError("All-zero volume")

    # Smooth for peak detection (don't alter raw values for output)
    smooth = np.convolve(bucket, np.ones(_SMOOTH_WINDOW) / _SMOOTH_WINDOW, mode="same")

    # ── Point of Control ──────────────────────────────────────────────────
    poc_idx = int(np.argmax(bucket))
    poc = float(centres[poc_idx])

    # ── Value Area (70% of volume, expanding from POC) ────────────────────
    va_vol = _VALUE_AREA * total_vol
    va_set = {poc_idx}
    va_accum = bucket[poc_idx]
    lo_ptr = poc_idx - 1
    hi_ptr = poc_idx + 1

    while va_accum < va_vol:
        lo_add = bucket[lo_ptr] if lo_ptr >= 0 else 0.0
        hi_add = bucket[hi_ptr] if hi_ptr < n_bins else 0.0
        if lo_add == 0 and hi_add == 0:
            break
        if hi_add >= lo_add:
            if hi_ptr < n_bins:
                va_set.add(hi_ptr)
                va_accum += hi_add
                hi_ptr += 1
        else:
            if lo_ptr >= 0:
                va_set.add(lo_ptr)
                va_accum += lo_add
                lo_ptr -= 1

    va_indices = sorted(va_set)
    value_area_low = float(centres[va_indices[0]])
    value_area_high = float(centres[va_indices[-1]])

    # ── HVN / LVN classification ──────────────────────────────────────────
    hvn_thresh = float(np.percentile(bucket, _HVN_PERCENTILE))
    lvn_thresh = float(np.percentile(bucket, _LVN_PERCENTILE))

    # Local maxima in smoothed histogram → HVN candidates
    hvn_prices, lvn_prices = [], []
    for i in range(1, n_bins - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and bucket[i] >= hvn_thresh:
            hvn_prices.append(float(centres[i]))
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1] and bucket[i] <= lvn_thresh:
            lvn_prices.append(float(centres[i]))

    # ── Build bin list ────────────────────────────────────────────────────
    bins: list[VPBin] = []
    for i, (c, vol) in enumerate(zip(centres, bucket, strict=False)):
        if i == poc_idx:
            btype = "poc"
        elif c in hvn_prices:
            btype = "hvn"
        elif c in lvn_prices:
            btype = "lvn"
        else:
            btype = "normal"
        bins.append(VPBin(price=float(c), volume=float(vol), pct=float(vol / total_vol), type=btype))

    # ── Current price context ─────────────────────────────────────────────
    current_price = float(closes[-1])
    if current_price > value_area_high:
        current_zone = "above_va"
    elif current_price < value_area_low:
        current_zone = "below_va"
    else:
        current_zone = "in_va"

    poc_distance_pct = (current_price - poc) / poc * 100.0

    return VolumeProfile(
        poc=poc,
        value_area_high=value_area_high,
        value_area_low=value_area_low,
        hvn=hvn_prices,
        lvn=lvn_prices,
        bins=bins,
        current_zone=current_zone,
        poc_distance_pct=poc_distance_pct,
        current_price=current_price,
    )


# ─── Zone proximity helpers ───────────────────────────────────────────────────


def nearest_node(
    price: float, vp: VolumeProfile, node_type: str = "hvn", max_pct: float = 3.0
) -> float | None:
    """
    Return the nearest HVN or LVN within `max_pct` % of price, or None.
    """
    nodes = vp.hvn if node_type == "hvn" else vp.lvn
    if not nodes:
        return None
    dists = [(abs(p - price) / price * 100, p) for p in nodes]
    dists.sort()
    best_dist, best_price = dists[0]
    return best_price if best_dist <= max_pct else None


def zone_type_at_price(price: float, vp: VolumeProfile, tol_pct: float = 0.5) -> str:
    """
    Return "hvn", "lvn", "poc" or "normal" based on where price sits in the VP.
    Used to annotate MC path segments.
    """
    tol = price * tol_pct / 100.0
    if abs(price - vp.poc) <= tol:
        return "poc"
    for h in vp.hvn:
        if abs(price - h) <= tol:
            return "hvn"
    for l in vp.lvn:
        if abs(price - l) <= tol:
            return "lvn"
    return "normal"
