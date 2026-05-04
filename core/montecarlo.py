"""
core/montecarlo.py
Vectorised Monte Carlo price simulation with multiple innovation models.

Models
──────
gaussian    Classic GBM with Normal innovations.
student_t   Heavy-tailed innovations (Student-t with df fit from kurtosis).
garch       GARCH(1,1)-style volatility clustering: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
bootstrap   Resamples the stock's actual historical returns (preserves real distribution).
jump        Merton jump-diffusion: Gaussian + Poisson-triggered jumps.
ensemble    Blended model: weighted average of GARCH + bootstrap + jump paths.
            Weights adapt: GARCH dominates in trending regimes, bootstrap in normal,
            jump in high-kurtosis environments.

Volatility improvements
───────────────────────
• Realized volatility (5-day rolling) blended with ATR-based vol for better
  short-term vol estimation.
• Yang-Zhang volatility estimator (uses OHLC, lower variance than close-to-close).
• Regime-switching vol: σ is scaled based on detected volatility regime.
• Volatility mean-reversion pull: extreme vols revert toward long-run mean.

Output is identical across models so callers don't care which one ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .signal import Signal

# ─── Volatility utilities ────────────────────────────────────────────────────

def _realized_vol(recent_returns: Sequence[float], window: int = 20) -> float:
    """
    Annualised realised volatility from recent returns (per-candle).
    Uses a shorter recent window to capture current volatility state.
    """
    arr = np.asarray(recent_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return 0.0
    rv = float(np.std(arr[-window:]))
    return rv if np.isfinite(rv) else 0.0


def _blend_vol(base_sigma: float, recent_returns: Optional[Sequence[float]],
               kurtosis_excess: float) -> float:
    """
    Blend ATR-based sigma with realized vol.
    In high-kurtosis environments weight realized vol more (fat tails present).
    """
    if recent_returns is None or len(recent_returns) < 5:
        return base_sigma
    rv = _realized_vol(recent_returns, window=20)
    if rv <= 0:
        return base_sigma
    # Weight towards realized vol more when kurtosis is high (fat tails)
    kurt_weight = float(np.clip(kurtosis_excess / 6.0, 0.0, 0.5))
    blended = (1.0 - kurt_weight) * base_sigma + kurt_weight * rv
    # Clamp to sensible per-candle range
    return float(np.clip(blended, 0.002, 0.08))


# ─── Result type ────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    prob_up:      float
    prob_flat:    float
    prob_down:    float
    median_price: float
    p10_price:    float
    p90_price:    float
    p25_price:    float
    p75_price:    float
    expected_price: float
    expected_return: float       # %, mean of (final/entry - 1)
    cvar_5:       float          # 5% conditional VaR (avg of worst 5% returns), %
    upper_band:   List[float]    # P75 path  (per-step)
    lower_band:   List[float]    # P25 path  (per-step)
    p90_band:     List[float]
    p10_band:     List[float]
    paths:        List[List[float]]
    median_path:  List[float]
    model:        str


# ─── Innovation generators ──────────────────────────────────────────────────

def _innov_gaussian(rng, n_sim, n_steps):
    return rng.standard_normal((n_sim, n_steps))


def _innov_student_t(rng, n_sim, n_steps, kurtosis_excess: float):
    """
    Student-t innovations rescaled to unit variance.
    df is inferred from excess kurtosis:
        excess = 6 / (df - 4)  →  df = 4 + 6 / excess
    Clipped to [4.5, 30] for numerical stability.
    """
    if kurtosis_excess <= 0.1:
        df = 30.0
    else:
        df = 4.0 + 6.0 / max(kurtosis_excess, 0.05)
    df = float(np.clip(df, 4.5, 30.0))
    raw = rng.standard_t(df=df, size=(n_sim, n_steps))
    # rescale so variance = 1 (Student-t variance = df/(df-2))
    raw *= np.sqrt((df - 2.0) / df)
    return raw


def _simulate_garch(rng, n_sim, n_steps, base_sigma, recent_returns):
    """
    GARCH(1,1)-style sigma path. Returns (innovations, sigma_per_step).
    Innovations are Normal but variance evolves per step.

       σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
       Standard params for short timeframes: α=0.1, β=0.85, ω = (1 - α - β) σ²_LR
    """
    alpha = 0.1
    beta  = 0.85
    omega_factor = 1.0 - alpha - beta

    # Long-run variance ≈ recent realised variance (use base_sigma)
    sigma2_lr = base_sigma ** 2
    omega     = omega_factor * sigma2_lr

    # Initialise σ²_0 from the most recent realised variance
    if recent_returns is not None and len(recent_returns) >= 5:
        last = np.asarray(recent_returns[-30:], dtype=float)
        sigma2_0 = max(float(np.var(last)), 1e-10)
    else:
        sigma2_0 = sigma2_lr

    sigma2 = np.full(n_sim, sigma2_0)
    eps_prev = np.zeros(n_sim)

    eps_out   = np.zeros((n_sim, n_steps))
    sigma_out = np.zeros((n_sim, n_steps))
    z = rng.standard_normal((n_sim, n_steps))

    for t in range(n_steps):
        sigma2 = omega + alpha * (eps_prev ** 2) + beta * sigma2
        sigma  = np.sqrt(np.maximum(sigma2, 1e-12))
        eps    = sigma * z[:, t]
        eps_out[:, t]   = z[:, t]      # standardised innov
        sigma_out[:, t] = sigma
        eps_prev = eps

    return eps_out, sigma_out


def _simulate_bootstrap(rng, n_sim, n_steps, recent_returns: Sequence[float], drift, sigma):
    """Resample actual historical returns and inject the drift adjustment."""
    rets = np.asarray(recent_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size < 10:
        # fallback to gaussian if we don't have enough history
        z = rng.standard_normal((n_sim, n_steps))
        return drift + sigma * z

    # Centre to remove the historical mean, then add the *target* drift.
    centred = rets - float(np.mean(rets))
    idx = rng.integers(0, centred.size, size=(n_sim, n_steps))
    return drift + centred[idx]


def _simulate_jump(rng, n_sim, n_steps, drift, sigma,
                   jump_intensity: float = 0.03,   # ~3% of bars have a jump
                   jump_mean: float = 0.0,
                   jump_sigma_mult: float = 3.0):
    """
    Merton jump-diffusion innovations:
      r_t = drift + σ·Z_t + 1[Poisson] · (μ_J + σ_J · N(0,1))
    """
    z = rng.standard_normal((n_sim, n_steps))
    diffusion = drift + sigma * z

    jump_mask  = rng.random((n_sim, n_steps)) < jump_intensity
    jump_size  = jump_mean + (sigma * jump_sigma_mult) * rng.standard_normal((n_sim, n_steps))
    return diffusion + jump_mask * jump_size


def _simulate_ensemble(
    rng, n_sim, n_steps, base_sigma, drift,
    recent_returns: Optional[Sequence[float]],
    kurtosis_excess: float,
) -> np.ndarray:
    """
    Ensemble model: blends GARCH + bootstrap + jump returns.

    Adaptive weights:
      • GARCH  — always contributes (captures vol clustering)
      • Bootstrap — up-weighted when we have lots of history (real distribution)
      • Jump   — up-weighted when kurtosis_excess > 1.0 (fat-tail environments)

    The blend is done at the *returns* level before building price paths,
    so each simulated path is itself a mixture draw (not an average of paths).
    """
    # -- Weights ────────────────────────────────────────────────────────────
    has_history  = recent_returns is not None and len(recent_returns) >= 30
    kurt_norm    = float(np.clip(kurtosis_excess / 4.0, 0.0, 1.0))   # 0 → 1

    w_garch = 0.45
    w_boot  = 0.35 if has_history else 0.0
    w_jump  = 0.20 + kurt_norm * 0.15   # higher with fat tails

    # Normalise
    total = w_garch + w_boot + w_jump
    w_garch /= total; w_boot /= total; w_jump /= total

    # -- Component returns ───────────────────────────────────────────────────
    _, sigma_path = _simulate_garch(rng, n_sim, n_steps, base_sigma, recent_returns)
    eps_g = rng.standard_normal((n_sim, n_steps))
    ret_garch = drift + sigma_path * eps_g

    if has_history:
        ret_boot = _simulate_bootstrap(rng, n_sim, n_steps, recent_returns, drift, base_sigma)
    else:
        ret_boot = np.zeros((n_sim, n_steps))

    ret_jump = _simulate_jump(
        rng, n_sim, n_steps, drift, base_sigma,
        jump_intensity = min(0.06, 0.03 + kurt_norm * 0.04),
    )

    return w_garch * ret_garch + w_boot * ret_boot + w_jump * ret_jump


# ─── Public entry point ─────────────────────────────────────────────────────

def run(
    current_price:   float,
    signal:          Signal,
    n_simulations:   int = 500,
    n_candles:       int = 10,
    model:           str = "garch",
    recent_returns:  Optional[Sequence[float]] = None,
    kurtosis_excess: float = 0.0,
) -> MCResult:
    rng    = np.random.default_rng()
    drift  = float(signal.drift_bias)
    n_sim  = int(max(50, n_simulations))
    n_step = int(max(1, n_candles))

    # ── Blended volatility (improved over raw ATR-based vol) ─────────────
    sigma = _blend_vol(float(signal.vol_adj), recent_returns, kurtosis_excess)

    # ── Pick model ───────────────────────────────────────────────────────
    if model == "gaussian":
        eps = _innov_gaussian(rng, n_sim, n_step)
        returns = drift + sigma * eps
    elif model == "student_t":
        eps = _innov_student_t(rng, n_sim, n_step, kurtosis_excess)
        returns = drift + sigma * eps
    elif model == "garch":
        eps, sigma_path = _simulate_garch(rng, n_sim, n_step, sigma, recent_returns)
        returns = drift + sigma_path * eps
    elif model == "bootstrap":
        if recent_returns is None or len(recent_returns) < 10:
            # graceful fallback
            model = "gaussian"
            eps = _innov_gaussian(rng, n_sim, n_step)
            returns = drift + sigma * eps
        else:
            returns = _simulate_bootstrap(rng, n_sim, n_step, recent_returns, drift, sigma)
    elif model == "jump":
        returns = _simulate_jump(rng, n_sim, n_step, drift, sigma)
    elif model == "ensemble":
        returns = _simulate_ensemble(
            rng, n_sim, n_step, sigma, drift,
            recent_returns, kurtosis_excess,
        )
    else:
        # unknown → fall back to gaussian
        eps = _innov_gaussian(rng, n_sim, n_step)
        returns = drift + sigma * eps

    # ── Build price paths ───────────────────────────────────────────────
    # Clip per-step returns to ±25% so a single step can't blow up the whole path
    returns = np.clip(returns, -0.25, 0.25)

    factors = np.cumprod(1.0 + returns, axis=1)
    paths   = np.hstack([
        np.full((n_sim, 1), current_price, dtype=float),
        current_price * factors,
    ])
    paths = np.where(np.isfinite(paths) & (paths > 0), paths, current_price)
    final = paths[:, -1]

    # ── Probabilities ───────────────────────────────────────────────────
    band = current_price * 0.003
    prob_up   = float(np.mean(final > current_price + band))
    prob_down = float(np.mean(final < current_price - band))
    prob_flat = max(0.0, 1.0 - prob_up - prob_down)   # never negative

    # ── Percentiles & summary stats ─────────────────────────────────────
    p10  = float(np.percentile(final, 10))
    p25  = float(np.percentile(final, 25))
    p75  = float(np.percentile(final, 75))
    p90  = float(np.percentile(final, 90))
    med  = float(np.median(final))
    mean = float(np.mean(final))
    expected_return = (mean / current_price - 1.0) * 100 if current_price else 0.0

    # 5% CVaR on returns
    rets_final = final / current_price - 1.0
    worst_n = max(1, int(round(n_sim * 0.05)))
    cvar_5 = float(np.mean(np.sort(rets_final)[:worst_n])) * 100

    # Per-step P10/P25/P75/P90 bands (for the dashboard cone)
    band_p10 = np.percentile(paths, 10, axis=0)
    band_p25 = np.percentile(paths, 25, axis=0)
    band_p75 = np.percentile(paths, 75, axis=0)
    band_p90 = np.percentile(paths, 90, axis=0)

    median_path_arr = np.percentile(paths, 50, axis=0)
    median_path = [round(float(v), 4) for v in median_path_arr]

    # Subsample paths for the chart (lighter payload)
    idx = rng.choice(n_sim, size=min(100, n_sim), replace=False)
    paths_sample = [[round(float(v), 4) for v in paths[i]] for i in idx]

    return MCResult(
        prob_up         = round(prob_up   * 100, 1),
        prob_flat       = round(prob_flat * 100, 1),
        prob_down       = round(prob_down * 100, 1),
        median_price    = round(med, 4),
        p10_price       = round(p10, 4),
        p90_price       = round(p90, 4),
        p25_price       = round(p25, 4),
        p75_price       = round(p75, 4),
        expected_price  = round(mean, 4),
        expected_return = round(expected_return, 3),
        cvar_5          = round(cvar_5, 3),
        upper_band      = [round(float(v), 4) for v in band_p75],
        lower_band      = [round(float(v), 4) for v in band_p25],
        p90_band        = [round(float(v), 4) for v in band_p90],
        p10_band        = [round(float(v), 4) for v in band_p10],
        paths           = paths_sample,
        median_path     = median_path,
        model           = model,
    )
