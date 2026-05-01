"""Tests for core/montecarlo.py."""

from __future__ import annotations

import math

import pytest

from core.indicators import compute_indicators
from core.montecarlo import run as run_mc
from core.signal import compute_signal


def _mc(df, model="garch", n_sim=300, n_fwd=10):
    ind = compute_indicators(df)
    sig = compute_signal(ind)
    price = float(df["close"].iloc[-1])
    return run_mc(price, sig, n_simulations=n_sim, n_candles=n_fwd,
                  model=model, recent_returns=ind.returns,
                  kurtosis_excess=ind.kurtosis), price


@pytest.mark.parametrize("model", ["gaussian", "student_t", "garch", "bootstrap", "jump"])
def test_mc_runs_each_model(synth_df, model):
    mc, _ = _mc(synth_df, model=model)
    assert mc.model in ("gaussian", "student_t", "garch", "bootstrap", "jump")
    # Probabilities each in [0, 100] and sum to 100 ± rounding
    for p in (mc.prob_up, mc.prob_flat, mc.prob_down):
        assert 0.0 <= p <= 100.0
    assert abs((mc.prob_up + mc.prob_flat + mc.prob_down) - 100.0) < 0.5


def test_prob_flat_never_negative(synth_df):
    """Regression: rounding could push prob_flat slightly negative — never again."""
    for model in ("gaussian", "student_t", "garch", "bootstrap", "jump"):
        mc, _ = _mc(synth_df, model=model)
        assert mc.prob_flat >= 0.0


def test_percentiles_ordered(synth_df):
    mc, _ = _mc(synth_df)
    assert mc.p10_price <= mc.p25_price <= mc.median_price <= mc.p75_price <= mc.p90_price


def test_paths_no_nan(synth_df):
    mc, _ = _mc(synth_df)
    for path in mc.paths[:5]:
        assert all(math.isfinite(v) and v > 0 for v in path)
    assert len(mc.median_path) >= 2
    assert all(math.isfinite(v) and v > 0 for v in mc.median_path)


def test_uptrend_drift_pulls_median_up(trend_up_df):
    mc, entry = _mc(trend_up_df, model="gaussian", n_sim=1000)
    # With a strong synthetic uptrend the median should not be far below entry
    assert mc.median_price > entry * 0.99


def test_unknown_model_falls_back(synth_df):
    mc, _ = _mc(synth_df, model="not-a-real-model")
    # Should not crash and return a valid result
    assert 0 <= mc.prob_up <= 100
