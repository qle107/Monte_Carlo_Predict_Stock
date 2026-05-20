"""Tests for core/indicators.py."""

from __future__ import annotations

import math

from core.indicators import compute_indicators


def test_indicators_basic_shape(synth_df):
    ind = compute_indicators(synth_df)
    # All numeric fields finite
    for f in (
        "rsi",
        "slope",
        "momentum",
        "ema_fast",
        "ema_slow",
        "atr_pct",
        "gap_pct",
        "mean_return",
        "std_return",
        "skewness",
        "trend_bias",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_position",
        "adx",
        "obv_slope",
        "vwap_dist",
        "kurtosis",
    ):
        v = getattr(ind, f)
        assert isinstance(v, (int, float)), f
        assert math.isfinite(v), f"{f} not finite: {v}"
    assert ind.ema_cross in {"bullish", "bearish", "neutral"}
    assert ind.vol_regime in {"low", "normal", "high"}


def test_rsi_bounds(synth_df):
    ind = compute_indicators(synth_df)
    assert 0.0 <= ind.rsi <= 100.0


def test_bollinger_bounds(synth_df):
    ind = compute_indicators(synth_df)
    assert -3.0 <= ind.bb_position <= 3.0


def test_adx_bounds(synth_df):
    ind = compute_indicators(synth_df)
    assert 0.0 <= ind.adx <= 100.0


def test_returns_attached(synth_df):
    ind = compute_indicators(synth_df)
    assert isinstance(ind.returns, list)
    assert len(ind.returns) > 0
    assert all(math.isfinite(r) for r in ind.returns)


def test_uptrend_detected(trend_up_df):
    ind = compute_indicators(trend_up_df)
    assert ind.trend_bias > 0.5
    assert ind.slope > 0


def test_downtrend_detected(trend_down_df):
    ind = compute_indicators(trend_down_df)
    assert ind.trend_bias < 0.5
    assert ind.slope < 0
