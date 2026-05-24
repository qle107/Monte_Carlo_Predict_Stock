"""
Tests for core/montecarlo.py.

Section A (lines below the first separator) — integration tests that use
`synth_df` / `trend_up_df` fixtures from conftest.py and exercise the full
pipeline (compute_indicators → compute_signal → run_mc).

Section B — standalone unit tests for the Part 4 math fixes that do not
require a candle DataFrame or live data:
  4.1  Itô / Jensen bias correction (all non-microstructure models)
  4.2  Merton jump compensator unbiasedness
  4.4  Stationary bootstrap ACF preservation
  4.5  Monte Carlo standard errors (binomial SE formula)
  4.6  Probabilities sum to exactly 100.0
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.indicators import compute_indicators
from core.montecarlo import (
    _build_mc_result,
    _simulate_bootstrap,
    _simulate_jump,
    compute_cvd_from_ohlc,
)
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
    # Probabilities each in [0, 100] and sum to exactly 100.0 (Part 4.6)
    for p in (mc.prob_up, mc.prob_flat, mc.prob_down):
        assert 0.0 <= p <= 100.0
    assert mc.prob_up + mc.prob_flat + mc.prob_down == 100.0


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
    # Both DFA field and deprecated alias must be populated and in range
    assert mc.ms_dfa_alpha is not None and 0.0 <= mc.ms_dfa_alpha <= 1.0
    assert mc.ms_hurst is not None and 0.0 <= mc.ms_hurst <= 1.0
    assert mc.ms_dfa_alpha == mc.ms_hurst, "ms_dfa_alpha and ms_hurst must be equal (same source)"
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


# ════════════════════════════════════════════════════════════════════════════
# Section B — standalone unit tests for Part 4 math fixes
# No fixture dependency; uses a minimal _FakeSignal stub.
# ════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass  # noqa: E402  (after integration imports)


@dataclass
class _FakeSignal:
    """Minimal Signal stub — avoids importing api/ or live-data dependencies."""
    drift_bias: float = 0.0
    vol_adj: float = 0.01


_SPOT = 100.0
_N_SIM = 2_000
_N_STEPS = 10
_SIGMA = 0.02
_SEED = 7


def _rng_b() -> np.random.Generator:
    return np.random.default_rng(_SEED)


def _recent_rets(n: int = 120, sigma: float = _SIGMA) -> list[float]:
    """Stationary AR(0.3) log-returns, no live data needed."""
    rng = np.random.default_rng(_SEED + 1)
    eps = rng.standard_normal(n) * sigma
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + eps[t]
    return x.tolist()


# ── 4.1  Itô / Jensen bias correction ────────────────────────────────────────

# Tolerance: 0.25 × σ × √n_steps  (≈ 0.016 at σ=0.02, n=10)
# Without the fix bias ≈ +½σ²·n_steps ≈ +0.002 — well above tolerance.
_ITO_TOL = 0.25 * _SIGMA * math.sqrt(_N_STEPS)


@pytest.mark.parametrize("model", ["gaussian", "student_t", "garch", "jump", "ensemble", "bootstrap"])
def test_ito_correction_unbiased(model):
    """
    With drift=0, E[log(S_T/S_0)] must be ≈ 0 for all non-microstructure models.

    Mathematical justification:
      Under the log-normal path model S_{t+1} = S_t · exp(μ_log + σZ),
      E[S_{t+1}/S_t] = exp(μ_log + ½σ²).
      To achieve E[S_T/S_0] = exp(T·drift) = 1 (when drift=0) we need
      μ_log = drift − ½σ², i.e. the Itô / Jensen correction.
      Without it every path drifts up by ½σ² per step.
    """
    signal = _FakeSignal(drift_bias=0.0, vol_adj=_SIGMA)
    result = run_mc(
        current_price=_SPOT,
        signal=signal,
        n_simulations=_N_SIM,
        n_candles=_N_STEPS,
        model=model,
        recent_returns=_recent_rets(),
        kurtosis_excess=1.0,
    )
    paths = np.asarray(result.paths_full)
    log_terminal = np.log(paths[:, -1] / _SPOT)
    mean_log = float(np.mean(log_terminal))
    assert abs(mean_log) < _ITO_TOL, (
        f"[{model}] Itô bias: E[log(S_T/S_0)]={mean_log:.5f}, "
        f"tolerance ±{_ITO_TOL:.5f}.  "
        f"Uncorrected bias ≈ +{0.5*_SIGMA**2*_N_STEPS:.5f}."
    )


# ── 4.2  Merton jump compensator ─────────────────────────────────────────────


def test_jump_compensator_unbiased():
    """
    Zero drift + zero mean jumps + large jump vol must keep E[sum log_ret] ≈ 0.

    The Merton compensator κ = exp(μ_J + ½σ_J²) − 1 accounts for the full
    Jensen correction over jump sizes.  The old arithmetic compensator λ·μ_J
    underestimates the bias when σ_J is large.

    Setup: λ=0.05, μ_J=0, σ_J = 5·σ = 0.10.
      κ = exp(0 + ½·0.01) − 1 ≈ 0.00501
      Without correction: bias ≈ λ·½σ_J²·n_steps = 0.05·0.005·10 = 0.0025.
    """
    rng = _rng_b()
    log_rets = _simulate_jump(
        rng,
        n_sim=_N_SIM,
        n_steps=_N_STEPS,
        drift=0.0,
        sigma=_SIGMA,
        jump_intensity=0.05,
        jump_mean=0.0,
        jump_sigma_mult=5.0,
    )
    mean_log = float(np.mean(np.sum(log_rets, axis=1)))
    tol = 0.5 * _SIGMA * math.sqrt(_N_STEPS)
    assert abs(mean_log) < tol, (
        f"Jump compensator bias: E[sum log_ret]={mean_log:.5f}, tol ±{tol:.5f}"
    )


# ── 4.4  Stationary bootstrap ACF preservation ───────────────────────────────


def test_stationary_bootstrap_preserves_acf():
    """
    The stationary bootstrap output ACF(1) must be ≥ 30 % of the input ACF(1).

    The i.i.d. resample destroys all autocorrelation (ACF → 0).  The stationary
    bootstrap with geometric block lengths ~ Geometric(1/b), b = N^{1/3} carries
    over short-range dependence.  Threshold 0.30 is conservative but would catch
    a regression to i.i.d. resampling.
    """
    recent = np.asarray(_recent_rets(n=500, sigma=_SIGMA * 3))
    acf1_in = float(np.corrcoef(recent[:-1], recent[1:])[0, 1])
    if acf1_in < 0.05:
        pytest.skip("Input ACF(1) too low")

    rng = _rng_b()
    emp_sigma = float(np.std(recent))
    raw = _simulate_bootstrap(rng, n_sim=500, n_steps=50, recent_returns=recent, drift=0.0, sigma=emp_sigma)
    # Undo the Itô offset to recover raw resampled values for ACF measurement
    resampled = raw + 0.5 * emp_sigma**2
    flat = resampled.ravel()
    acf1_out = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])

    assert acf1_out >= 0.30 * acf1_in, (
        f"Bootstrap ACF(1): out={acf1_out:.3f}, in={acf1_in:.3f}, "
        f"threshold={0.30*acf1_in:.3f}"
    )


# ── 4.5  Monte Carlo standard errors ─────────────────────────────────────────


def test_prob_se_matches_binomial_formula():
    """
    Synthetic paths with exactly 50% up and 50% down → prob_up_se = 0.5 pp.

    Formula:  SE(p̂) = sqrt(p̂·(1−p̂)/n_sim) × 100
    At p̂=0.5, n_sim=10_000:  SE = sqrt(0.25/10_000)×100 = 0.5 pp exactly.
    """
    n = 10_000
    paths = np.empty((n, 2))
    paths[:, 0] = _SPOT
    paths[: n // 2, 1] = _SPOT * 1.10   # above any band → counted as "up"
    paths[n // 2 :, 1] = _SPOT * 0.90   # below → "down"
    rng = _rng_b()
    result = _build_mc_result(rng, paths, _SPOT, "gaussian", n)
    expected_se = math.sqrt(0.25 / n) * 100   # = 0.5 pp
    assert abs(result.prob_up_se - expected_se) < 0.01, (
        f"prob_up_se={result.prob_up_se:.4f}, expected {expected_se:.4f}"
    )
    assert abs(result.prob_down_se - expected_se) < 0.01, (
        f"prob_down_se={result.prob_down_se:.4f}, expected {expected_se:.4f}"
    )


def test_cvar_se_non_negative_and_finite():
    """cvar_5_se must be non-negative and finite for any valid path matrix."""
    signal = _FakeSignal(drift_bias=0.0, vol_adj=_SIGMA)
    result = run_mc(
        current_price=_SPOT,
        signal=signal,
        n_simulations=500,
        n_candles=10,
        model="gaussian",
    )
    assert result.cvar_5_se >= 0.0
    assert math.isfinite(result.cvar_5_se)


def test_prob_se_decreases_with_more_sims():
    """SE ∝ 1/√n_sim: quadrupling n should roughly halve the SE."""
    spot = 100.0
    frac_up = 0.6

    def _paths(n: int) -> np.ndarray:
        p = np.full((n, 2), spot)
        n_up = round(n * frac_up)
        p[:n_up, 1] = spot * 1.05
        p[n_up:, 1] = spot * 0.95
        return p

    r1 = _build_mc_result(np.random.default_rng(1), _paths(1_000), spot, "gaussian", 1_000)
    r2 = _build_mc_result(np.random.default_rng(2), _paths(4_000), spot, "gaussian", 4_000)
    ratio = r1.prob_up_se / r2.prob_up_se if r2.prob_up_se > 0 else 0
    assert 1.5 < ratio < 3.0, f"SE scaling ratio={ratio:.2f} (expected ≈2)"


# ── 4.6  Probabilities sum to exactly 100.0 ──────────────────────────────────


@pytest.mark.parametrize("model", ["gaussian", "student_t", "garch", "jump", "ensemble", "bootstrap"])
def test_probs_sum_to_100_standalone(model):
    """
    prob_up + prob_flat + prob_down == 100.0 (floating-point equality) for all
    non-microstructure models without a fixture dependency.

    Without the round-to-100 fix, independent rounding of three components
    to 1 d.p. yields sums of 99.9 or 100.1 when half-up boundaries collide.
    """
    signal = _FakeSignal(drift_bias=0.0, vol_adj=_SIGMA)
    result = run_mc(
        current_price=_SPOT,
        signal=signal,
        n_simulations=400,
        n_candles=10,
        model=model,
        recent_returns=_recent_rets(),
        kurtosis_excess=1.0,
    )
    total = result.prob_up + result.prob_flat + result.prob_down
    assert total == 100.0, (
        f"[{model}] {result.prob_up} + {result.prob_flat} + {result.prob_down} = {total}"
    )


def test_probs_sum_to_100_all_up_degenerate():
    """Edge case: all paths above spot → prob_up ≈ 100, others ≈ 0, sum exact."""
    n = 500
    p = np.full((n, 2), _SPOT)
    p[:, 1] = _SPOT * 1.10
    result = _build_mc_result(_rng_b(), p, _SPOT, "gaussian", n)
    assert result.prob_up + result.prob_flat + result.prob_down == 100.0


def test_probs_sum_to_100_all_flat_degenerate():
    """Edge case: all paths at spot → prob_flat ≈ 100, sum exact."""
    n = 500
    p = np.full((n, 2), _SPOT)
    result = _build_mc_result(_rng_b(), p, _SPOT, "gaussian", n)
    assert result.prob_up + result.prob_flat + result.prob_down == 100.0


# ── ms_dfa_alpha field wiring ────────────────────────────────────────────────


def test_ms_dfa_alpha_none_for_non_ms_models():
    """ms_dfa_alpha must be None for all non-microstructure models."""
    signal = _FakeSignal()
    result = run_mc(current_price=_SPOT, signal=signal, n_simulations=100, n_candles=5, model="gaussian")
    assert result.ms_dfa_alpha is None


def test_se_fields_present_on_result():
    """All three SE fields must exist and be non-negative on every MCResult."""
    signal = _FakeSignal(drift_bias=0.001, vol_adj=_SIGMA)
    result = run_mc(current_price=_SPOT, signal=signal, n_simulations=200, n_candles=5, model="gaussian")
    assert hasattr(result, "prob_up_se") and result.prob_up_se >= 0.0
    assert hasattr(result, "prob_down_se") and result.prob_down_se >= 0.0
    assert hasattr(result, "cvar_5_se") and result.cvar_5_se >= 0.0
