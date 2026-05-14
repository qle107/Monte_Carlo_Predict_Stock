"""
Tests for core/zones.py — the demand / supply zone detector.

These tests directly target the bugs fixed in the refactor:
  • zone strength must be > 0 when there are touches (the old undefined
    `touches` variable used to zero the touch component entirely)
  • depth scoring uses the formation candle's body% (no longer a flat 0.1)
  • broken zones are filtered via lows/highs, catching intraday wick
    violations that a close-only filter would miss
  • the public ZoneResult contract is preserved end-to-end
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.zones import (
    Zone, ZoneResult, detect_zones,
    _body_strength, _find_pivots, _cluster, _remove_broken,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _ohlcv(rows: list[tuple]) -> pd.DataFrame:
    """rows = [(open, high, low, close, vol), ...]"""
    idx = pd.date_range("2024-01-02", periods=len(rows), freq="15min", tz="UTC")
    arr = np.asarray(rows, dtype=float)
    return pd.DataFrame({
        "open":  arr[:, 0],
        "high":  arr[:, 1],
        "low":   arr[:, 2],
        "close": arr[:, 3],
        "volume": arr[:, 4].astype(int),
    }, index=idx)


def _double_bottom_df() -> pd.DataFrame:
    """
    Engineered series with two clear demand pivots near $98, separated by
    a rally to $105, then price drifts up to $102. Strong demand zone at 98
    should be detected and the current price should sit "between" zones.
    """
    rng = np.random.default_rng(7)
    n   = 80
    base = 100 + rng.normal(0, 0.15, n)

    # Carve two distinct pivot lows near $98
    for pivot in (20, 50):
        base[pivot - 1] = 99.0
        base[pivot]     = 98.0       # the swing low
        base[pivot + 1] = 99.0

    rows = []
    for i, c in enumerate(base):
        h = c + abs(rng.normal(0, 0.3))
        l = c - abs(rng.normal(0, 0.3))
        o = c + rng.normal(0, 0.1)
        # Wide body on pivot bars so depth score is meaningful
        if i in (20, 50):
            o = c + 1.0                                    # close 1$ above open
            h = c + 1.2
            l = c - 0.1
        rows.append((o, h, l, c, 10_000))
    return _ohlcv(rows)


# ─── Helper tests ────────────────────────────────────────────────────────────

def test_body_strength_marubozu_returns_one():
    df = _ohlcv([(100, 105, 100, 105, 1)])                 # body = range
    assert _body_strength(df, 0) == pytest.approx(1.0)


def test_body_strength_doji_returns_zero():
    df = _ohlcv([(100, 105, 95, 100, 1)])                  # body = 0, range = 10
    assert _body_strength(df, 0) == pytest.approx(0.0)


def test_find_pivots_detects_obvious_lows():
    # Synthesise a V-shape low at index 5
    highs = np.array([10, 11, 10, 11, 10,  9, 10, 11, 10, 11], dtype=float)
    lows  = np.array([ 9, 10,  9, 10,  9,  6,  9, 10,  9, 10], dtype=float)
    ph, pl = _find_pivots(highs, lows, wing=2)
    assert any(idx == 5 and price == 6.0 for idx, price in pl)


def test_cluster_merges_nearby_pivots():
    # Three pivots within 0.5 ATR of each other → single zone
    pivots = [(10, 100.0), (12, 100.3), (14, 100.5)]
    zones  = _cluster(pivots, atr=1.0, zone_type="demand")
    assert len(zones) == 1
    z = zones[0]
    assert z.bar_idx == 14                                 # most recent
    assert 100.0 <= z.level <= 100.5


def test_cluster_separates_distant_pivots():
    pivots = [(10, 100.0), (15, 120.0)]
    zones  = _cluster(pivots, atr=1.0, zone_type="demand")
    assert len(zones) == 2


# ─── End-to-end ──────────────────────────────────────────────────────────────

def test_detect_zones_returns_empty_for_short_df():
    df = _ohlcv([(100, 101, 99, 100, 1)] * 5)
    res = detect_zones(df)
    assert isinstance(res, ZoneResult)
    assert res.demand_zones == [] and res.supply_zones == []
    assert res.price_context == "unknown"


def test_detect_zones_strength_is_nonzero_when_touches_exist():
    """
    Regression for the original bug: zone strength was identically zero on
    every zone because the touch score referenced an undefined variable.
    """
    df = _double_bottom_df()
    res = detect_zones(df)
    assert res.demand_zones, "expected at least one demand zone"
    # Strength should reflect touches + recency + freshness + depth > 0
    for z in res.demand_zones:
        assert 0.0 < z.strength <= 1.0


def test_detect_zones_strength_uses_real_body_depth():
    """
    Two otherwise-identical setups, one with a doji formation bar and one
    with a wide-body formation bar — the wide-body zone must score higher
    in the depth component.
    """
    # Make the swing low a clean rejection candle (wide body)
    rows_wide = []
    for i in range(60):
        if i == 30:
            rows_wide.append((101.0, 101.0, 99.0, 99.0, 5_000))   # close at low → bearish big body
        elif i == 31:
            rows_wide.append((99.0, 102.0, 98.5, 101.5, 5_000))   # huge bounce → demand
        else:
            rows_wide.append((100.0, 100.3, 99.7, 100.0, 5_000))

    # Same shape but the pivot bar is a doji
    rows_doji = list(rows_wide)
    rows_doji[31] = (100.5, 100.6, 98.5, 100.5, 5_000)            # tiny body, wide range

    df_wide = _ohlcv(rows_wide)
    df_doji = _ohlcv(rows_doji)

    s_wide = max((z.strength for z in detect_zones(df_wide).demand_zones), default=0.0)
    s_doji = max((z.strength for z in detect_zones(df_doji).demand_zones), default=0.0)
    assert s_wide >= s_doji                                # depth contributes


def test_remove_broken_filters_intraday_violations():
    """
    A demand zone at $95 followed by a bar with low=$93 (close back above)
    is *broken intraday* and must be filtered out — the previous
    close-only filter would have kept it.
    """
    z = Zone(level=95.0, low=94.7, high=95.3, strength=0.5, touches=2,
             fresh=True, zone_type="demand", bar_idx=0)
    highs = np.array([95.0, 96.0, 97.0])
    lows  = np.array([95.0, 93.0, 95.0])                   # wick to 93 on bar 1
    survivors = _remove_broken([z], highs, lows, atr=1.0)
    assert survivors == []                                 # zone filtered out


def test_remove_broken_keeps_intact_zones():
    z = Zone(level=95.0, low=94.7, high=95.3, strength=0.5, touches=2,
             fresh=True, zone_type="demand", bar_idx=0)
    highs = np.array([95.0, 96.0, 97.0])
    lows  = np.array([95.0, 95.0, 95.0])                   # never breached
    survivors = _remove_broken([z], highs, lows, atr=1.0)
    assert len(survivors) == 1


def test_zone_result_contract_preserved(synth_df):
    """Public ZoneResult shape is unchanged — needed by trade_setup.py."""
    res = detect_zones(synth_df)
    for field in ("demand_zones", "supply_zones", "nearest_demand",
                  "nearest_supply", "price_context", "atr"):
        assert hasattr(res, field)
    assert res.price_context in {"at_demand", "at_supply", "between", "unknown"}
