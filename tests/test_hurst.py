"""
tests/test_hurst.py
Regression tests for core.hurst.dfa (DFA-1 exponent estimator).

Mathematical ground truth
─────────────────────────
White noise       x_t ~ N(0,1)                    → α ≈ 0.50
Random walk       x_t = x_{t-1} + ε_t             → α ≈ 1.00
AR(1) φ=0.95      x_t = φ·x_{t-1} + ε_t           → α > 0.60  (long-range corr)

Tolerances are deliberately loose (±0.10–0.15) so the tests remain stable
across different random seeds while still catching regressions where α drifts
to a clearly wrong regime (e.g. white noise estimated as 0.9).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.hurst import dfa


# ── helpers ───────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
N = 2_000  # long enough for reliable estimation


def _white_noise(n: int = N) -> np.ndarray:
    return RNG.standard_normal(n)


def _random_walk(n: int = N) -> np.ndarray:
    return np.cumsum(RNG.standard_normal(n))


def _ar1(phi: float = 0.95, n: int = N) -> np.ndarray:
    """Stationary AR(1) with innovations N(0, 1-φ²) so variance ≈ 1."""
    eps = RNG.standard_normal(n) * np.sqrt(1.0 - phi**2)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


# ── basic API contract ────────────────────────────────────────────────────────


def test_dfa_returns_tuple():
    alpha, se = dfa(_white_noise())
    assert isinstance(alpha, float)
    assert isinstance(se, float)


def test_dfa_alpha_in_range():
    """α must be clipped to [0, 2] for any finite input."""
    alpha, _ = dfa(_white_noise())
    assert 0.0 <= alpha <= 2.0


def test_dfa_se_non_negative():
    _, se = dfa(_white_noise())
    assert se >= 0.0


def test_dfa_short_series_fallback():
    """Fewer than 2·min_box points → safe fallback (0.5, 0.0)."""
    alpha, se = dfa(np.array([1.0, 2.0, 3.0]))
    assert alpha == 0.5
    assert se == 0.0


def test_dfa_all_constant_fallback():
    """Constant series → degenerate; must not raise."""
    alpha, se = dfa(np.ones(500))
    assert np.isfinite(alpha)
    assert np.isfinite(se)


def test_dfa_ignores_nan_inf():
    """NaN / ±Inf values are stripped before estimation."""
    base = _white_noise(500)
    corrupted = base.copy().astype(float)
    corrupted[10] = np.nan
    corrupted[20] = np.inf
    corrupted[30] = -np.inf
    alpha, _ = dfa(corrupted)
    assert np.isfinite(alpha)


# ── statistical ground-truth tests ───────────────────────────────────────────


def test_dfa_white_noise_alpha_near_half():
    """
    White noise (i.i.d. N(0,1)) has DFA exponent α ≈ 0.50.

    Derivation: the integrated centred series of white noise is a random walk
    (cumsum of mean-zero i.i.d. increments).  The fluctuation function scales
    as F(n) ~ n^{1/2}, giving α = 0.5.

    Tolerance ±0.10 is conservative enough to survive different seeds but
    tight enough to catch a broken implementation returning 0.0 or 1.0.
    """
    alpha, _ = dfa(_white_noise())
    assert abs(alpha - 0.5) < 0.10, f"White noise DFA α={alpha:.3f}, expected ≈0.50"


def test_dfa_random_walk_alpha_near_one():
    """
    A random walk (cumulative sum of white noise) has DFA exponent α ≈ 1.00.

    Derivation: the integrated series of a random walk is Brownian motion
    integrated once more (fractional Brownian motion with H=1), giving
    F(n) ~ n^1 → α = 1.0.

    Tolerance ±0.15 accommodates finite-sample variance in the log-log fit.
    """
    alpha, _ = dfa(_random_walk())
    assert abs(alpha - 1.0) < 0.15, f"Random-walk DFA α={alpha:.3f}, expected ≈1.00"


def test_dfa_ar1_alpha_above_trend_threshold():
    """
    AR(1) with φ=0.95 exhibits strong long-range correlation, so α > 0.60.

    Heuristic: persistent autocorrelation inflates F(n) relative to the n^{1/2}
    baseline; the estimated α is between 0.5 and 1.0, typically ≈ 0.7–0.9
    for φ=0.95.  We only assert α > 0.60 to avoid over-constraining on N=2000.
    """
    alpha, _ = dfa(_ar1(phi=0.95))
    assert alpha > 0.60, f"AR(1) φ=0.95 DFA α={alpha:.3f}, expected >0.60"


def test_dfa_anti_persistent_alpha_below_half():
    """
    AR(1) with φ=−0.7 is anti-persistent (mean-reverting), so α < 0.50.
    Negative autocorrelation causes the integrated series to fluctuate less
    than a random walk → F(n) grows slower → α < 0.5.
    """
    alpha, _ = dfa(_ar1(phi=-0.7))
    # Anti-persistent: allow some slack but must be clearly below 0.50
    assert alpha < 0.50, f"Anti-persistent AR(1) DFA α={alpha:.3f}, expected <0.50"


def test_dfa_ordering_white_noise_vs_random_walk():
    """
    For the same seed, α(white noise) < α(random walk).  This is a
    model-ordering invariant that must always hold regardless of N.
    """
    rng = np.random.default_rng(99)
    wn = rng.standard_normal(1000)
    rw = np.cumsum(rng.standard_normal(1000))
    a_wn, _ = dfa(wn)
    a_rw, _ = dfa(rw)
    assert a_wn < a_rw, (
        f"Ordering violated: α_WN={a_wn:.3f} ≥ α_RW={a_rw:.3f}"
    )


def test_dfa_min_box_larger_than_default():
    """Custom min_box parameter is accepted and produces finite results."""
    alpha, se = dfa(_white_noise(), min_box=8)
    assert np.isfinite(alpha)
    assert np.isfinite(se)


def test_dfa_custom_max_box():
    """Custom max_box restricts the box-size grid and still returns valid α."""
    alpha, se = dfa(_white_noise(500), min_box=4, max_box=64)
    assert np.isfinite(alpha)
    assert 0.0 <= alpha <= 2.0
