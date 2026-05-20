"""Tests for core/montecarlo.py."""

from __future__ import annotations

import math

import pytest

from core.indicators import compute_indicators
from core.montecarlo import compute_cvd_from_ohlc
from core.montecarlo import run as run_mc
from core.signal import compute_signal
from core.volume_profile import compute_volume_profile

_VALID_MODELS = ("gaussian", "student_t", "garch", "bootstrap", "jump", "ensemble", "microstructure")


def _mc(df, model="garch", n_sim=300, n_fwd=10, with_microstructure_inputs=False):
    """
    Helper: run the MC engine on a candle DataFrame.

    When `with_microstructure_inputs=True`, also supplies the volume profile,
    CVD history, and raw volume so the microstructure model can exercise its
    full code path (gravity, CVD bias, volume σ scaling).
    """
    ind = compute_indicators(df)
    sig = compute_signal(ind)
    price = float(df["close"].iloc[-1])

    extras = {}
    if with_microstructure_inputs:
        extras = {
            "price_history": df["close"].to_numpy(dtype=float),
            "volume_history": df["volume"].to_numpy(dtype=float),
            "cvd_history": compute_cvd_from_ohlc(df["open"], df["close"], df["volume"]),
            "volume_profile": compute_volume_profile(df),
        }

    return run_mc(
        price,
        sig,
        n_simulations=n_sim,
        n_candles=n_fwd,
        model=model,
        recent_returns=ind.returns,
        kurtosis_excess=ind.kurtosis,
        **extras,
    ), price


@pytest.mark.parametrize(
    "model", ["gaussian", "student_t", "garch", "bootstrap", "jump", "ensemble", "microstructure"]
)
def test_mc_runs_each_model(synth_df, model):
    mc, _ = _mc(synth_df, model=model)
    assert mc.model in _VALID_MODELS
    # Probabilities each in [0, 100] and sum to 100 ± rounding
    for p in (mc.prob_up, mc.prob_flat, mc.prob_down):
        assert 0.0 <= p <= 100.0
    assert abs((mc.prob_up + mc.prob_flat + mc.prob_down) - 100.0) < 0.5


def test_prob_flat_never_negative(synth_df):
    """Regression: rounding could push prob_flat slightly negative — never again."""
    for model in _VALID_MODELS:
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


# ─── Microstructure-model-specific tests ────────────────────────────────────


def test_microstructure_with_full_inputs_produces_valid_paths(synth_df):
    """
    The microstructure model should run end-to-end when fed volume profile,
    CVD history, and per-bar volume — exercising gravity, CVD bias, and
    volume-scaled σ.
    """
    mc, entry = _mc(synth_df, model="microstructure", with_microstructure_inputs=True, n_sim=400)
    assert mc.model == "microstructure"
    assert mc.p10_price <= mc.p25_price <= mc.median_price <= mc.p75_price <= mc.p90_price
    # All sample paths positive, finite, start at entry
    for path in mc.paths[:5]:
        assert math.isclose(path[0], entry, rel_tol=1e-6)
        assert all(math.isfinite(v) and v > 0 for v in path)


def test_microstructure_falls_back_without_microstructure_inputs(synth_df):
    """
    Without volume_profile / CVD / volume_history, the microstructure model
    must still produce a valid MCResult (degraded to GARCH + Student-t only).
    """
    mc, _ = _mc(synth_df, model="microstructure", with_microstructure_inputs=False, n_sim=300)
    assert mc.model == "microstructure"
    assert 0 <= mc.prob_up + mc.prob_flat + mc.prob_down <= 100.5
    assert all(math.isfinite(v) and v > 0 for v in mc.median_path)


def test_microstructure_uptrend_drift_pulls_median_up(trend_up_df):
    """A strong uptrend should still pull the microstructure median up."""
    mc, entry = _mc(trend_up_df, model="microstructure", with_microstructure_inputs=True, n_sim=1000)
    assert mc.median_price > entry * 0.98


def test_compute_cvd_from_ohlc_signs_correctly():
    """CVD should add volume on up-bars, subtract on down-bars, ignore dojis."""
    import numpy as np

    opens = [10.0, 10.0, 11.0, 12.0]
    closes = [11.0, 10.0, 9.5, 12.0]  # up, doji, down, doji
    volumes = [100.0, 200.0, 150.0, 50.0]
    cvd = compute_cvd_from_ohlc(opens, closes, volumes)
    # Per-bar deltas: +100, 0, -150, 0  →  cumulative: 100, 100, -50, -50
    np.testing.assert_array_equal(cvd, np.array([100.0, 100.0, -50.0, -50.0]))


# ─── Microstructure diagnostics ─────────────────────────────────────────────


def test_ms_diagnostics_populated_for_microstructure(synth_df):
    """The microstructure model must populate ms_* fields with real values."""
    mc, _ = _mc(synth_df, model="microstructure", with_microstructure_inputs=True, n_sim=300)
    assert mc.ms_regime in ("trending", "mean-reverting", "neutral")
    assert mc.ms_hurst is not None and 0.0 <= mc.ms_hurst <= 1.0
    assert mc.ms_drift_bias is not None
    assert mc.ms_key_levels is not None
    # Volume profile actually produced levels
    assert mc.ms_key_levels["POC"] is not None
    assert isinstance(mc.ms_key_levels["HVNs"], list)
    assert isinstance(mc.ms_key_levels["LVNs"], list)


def test_ms_diagnostics_none_for_other_models(synth_df):
    """Non-microstructure models must leave ms_* fields as None."""
    for model in ("gaussian", "student_t", "garch", "bootstrap", "jump", "ensemble"):
        mc, _ = _mc(synth_df, model=model, n_sim=200)
        assert mc.ms_regime is None, f"{model} leaked ms_regime"
        assert mc.ms_hurst is None, f"{model} leaked ms_hurst"
        assert mc.ms_drift_bias is None, f"{model} leaked ms_drift_bias"
        assert mc.ms_key_levels is None, f"{model} leaked ms_key_levels"


def test_ms_diagnostics_present_in_fallback_mode(synth_df):
    """Even without volume profile / CVD, ms_* should be populated (POC = None)."""
    mc, _ = _mc(synth_df, model="microstructure", with_microstructure_inputs=False, n_sim=200)
    assert mc.ms_regime in ("trending", "mean-reverting", "neutral")
    assert mc.ms_hurst is not None
    # No volume profile → POC/VAH/VAL absent, HVN/LVN lists empty
    assert mc.ms_key_levels["POC"] is None
    assert mc.ms_key_levels["HVNs"] == []
    assert mc.ms_key_levels["LVNs"] == []


def test_microstructure_perf_n_sim_2000(synth_df):
    """
    Regression guard: vectorised gravity should keep microstructure under 5 s
    even at n_sim=2000, n_steps=10. The pre-vectorisation Python loop took
    ~10–20× longer.
    """
    import time

    start = time.perf_counter()
    mc, _ = _mc(synth_df, model="microstructure", with_microstructure_inputs=True, n_sim=2000, n_fwd=10)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"microstructure too slow: {elapsed:.2f}s"
    assert mc.model == "microstructure"
