"""
core/montecarlo.py
Vectorised Monte Carlo price simulation with multiple innovation models.

Models
──────
gaussian        Classic GBM with Normal innovations.
student_t       Heavy-tailed innovations (Student-t with df fit from kurtosis).
garch           GARCH(1,1)-style volatility clustering.
bootstrap       Resamples the stock's actual historical returns.
jump            Merton jump-diffusion: Gaussian + Poisson-triggered jumps.
ensemble        Adaptive blend of GARCH + bootstrap + jump.
microstructure  Market-microstructure-aware: GARCH(1,1) + Student-t (df=4) +
                volume-profile gravity (POC/VAH/VAL/HVN/LVN) + CVD drift bias +
                Hurst regime detection + LVN acceleration. Falls back gracefully
                when volume_profile / cvd_history / volume_history are absent.

Output schema (`MCResult`) is identical across all models so callers don't
care which one ran. The microstructure-only diagnostic fields (`ms_regime`,
`ms_hurst`, `ms_drift_bias`, `ms_key_levels`) default to None for other models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize

from .signal import Signal
from config import cfg


# ════════════════════════════════════════════════════════════════════════════
# Microstructure model — tunable parameters
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _MSParams:
    """All microstructure tuning knobs in one place. Frozen for safety."""

    # Student-t fat-tail parameter
    t_df: int = 4                      # df=4 captures gap moves & visible tails

    # Volume-profile gravity scalars
    poc_gravity:        float = 0.50   # strongest magnet
    hvn_gravity:        float = 0.30   # moderate magnet, also damps σ
    va_gravity:         float = 0.35   # VAH / VAL mean-reversion force
    acc_dist_gravity:   float = 0.20   # accumulation/distribution side-pull
    proximity_pct:      float = 0.01   # "within 1% of level"
    lvn_proximity_pct:  float = 0.015  # LVN gaps span a slightly wider band
    sigma_hvn_damp:     float = 0.85   # -15% σ when near an HVN
    sigma_lvn_boost:    float = 1.20   # +20% σ inside an LVN
    lvn_accel_extra:    float = 1.25   # additional +25% σ acceleration in LVN

    # Raw-volume → σ scaling & drift validation
    vol_high_ratio:     float = 1.5
    vol_low_ratio:      float = 0.7
    sigma_high_vol:     float = 1.25
    sigma_low_vol:      float = 0.80
    drift_distrust:     float = 0.50   # halve drift when volume diverges

    # CVD drift bias
    cvd_bias_cap:       float = 0.0025  # ±25 bps per step max
    cvd_flat_threshold: float = 0.10
    cvd_acc_dist_fact:  float = 0.30

    # Hurst regime thresholds & multipliers
    hurst_trend:         float = 0.55
    hurst_mean_rev:      float = 0.45
    reg_trend_drift:     float = 1.50
    reg_trend_gravity:   float = 0.60
    reg_trend_sigma:     float = 1.05
    reg_mr_drift:        float = 0.00
    reg_mr_gravity:      float = 1.50
    reg_mr_sigma:        float = 0.90
    reg_neut_drift:      float = 1.00
    reg_neut_gravity:    float = 1.00
    reg_neut_sigma:      float = 1.00

    # σ guard rails (per-bar log-return scale)
    sigma_min:          float = 0.005
    sigma_max:          float = 0.10

    # Lookback windows
    garch_window:       int = 90
    hurst_window:       int = 60
    cvd_slope_window:   int = 20
    vol_avg_window:     int = 20
    price_trend_window: int = 10

    # Volume-profile structure derivation
    value_area_pct:    float = 0.70
    hvn_rel_threshold: float = 0.50    # HVN ≥ 50 % of POC volume
    lvn_rel_threshold: float = 0.30    # LVN ≤ 30 % of median volume


_PARAMS = _MSParams()


# ════════════════════════════════════════════════════════════════════════════
# Result type
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MCResult:
    prob_up:        float
    prob_flat:      float
    prob_down:      float
    median_price:   float
    p10_price:      float
    p90_price:      float
    p25_price:      float
    p75_price:      float
    expected_price: float
    expected_return: float       # %, mean of (final/entry - 1)
    cvar_5:         float        # 5% conditional VaR (avg of worst 5% returns), %
    upper_band:     List[float]  # P75 path  (per-step)
    lower_band:     List[float]  # P25 path  (per-step)
    p90_band:       List[float]
    p10_band:       List[float]
    paths:          List[List[float]]   # chart sample (100 paths)
    paths_full:     object              # numpy array (n_sim, n_steps+1) for trade_setup
    median_path:    List[float]
    model:          str

    # Microstructure-only diagnostics (None for other models)
    ms_regime:      Optional[str]   = None
    ms_hurst:       Optional[float] = None
    ms_drift_bias:  Optional[float] = None
    ms_key_levels:  Optional[dict]  = field(default=None)


# ════════════════════════════════════════════════════════════════════════════
# Existing model helpers (untouched — proven & tested)
# ════════════════════════════════════════════════════════════════════════════

def _calibrate_garch(recent_returns: Optional[Sequence[float]],
                     base_alpha: float, base_beta: float
                     ) -> tuple[float, float]:
    """
    Adaptively scale GARCH α and β based on the detected volatility regime.
    Returns (alpha, beta) that still satisfy α+β < 1 for stationarity.
    """
    if recent_returns is None or len(recent_returns) < 20:
        return base_alpha, base_beta

    arr = np.asarray(recent_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return base_alpha, base_beta

    recent_vol  = float(np.std(arr[-10:]))
    longrun_vol = float(np.std(arr))
    if longrun_vol <= 0:
        return base_alpha, base_beta

    ratio = recent_vol / longrun_vol
    if ratio < 0.5:
        # Low vol regime: less reactive to shocks, more persistent
        alpha = base_alpha * 0.70
        beta  = min(base_beta * 1.05, 0.95 - alpha)
    elif ratio > 1.5:
        # High vol regime: more reactive to shocks
        alpha = min(base_alpha * 1.40, 0.30)
        beta  = min(base_beta * 0.95, 0.95 - alpha)
    else:
        alpha, beta = base_alpha, base_beta

    if alpha + beta >= 1.0:
        scale = 0.97 / (alpha + beta)
        alpha *= scale
        beta  *= scale

    return float(np.clip(alpha, 0.01, 0.49)), float(np.clip(beta, 0.10, 0.94))


def _adaptive_clip(recent_returns: Optional[Sequence[float]],
                   base_clip: float) -> float:
    """Adapt return clipping threshold to empirical tail quantiles."""
    if recent_returns is None or len(recent_returns) < 20:
        return base_clip
    arr = np.asarray(recent_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return base_clip
    p99 = float(np.percentile(np.abs(arr), 99))
    return float(np.clip(p99 * 3.0, 0.05, base_clip))


def _realized_vol(recent_returns: Sequence[float], window: int = 20) -> float:
    arr = np.asarray(recent_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return 0.0
    rv = float(np.std(arr[-window:]))
    return rv if np.isfinite(rv) else 0.0


def _blend_vol(base_sigma: float, recent_returns: Optional[Sequence[float]],
               kurtosis_excess: float) -> float:
    """Blend ATR-based σ with realised vol; weight realised more under fat tails."""
    if recent_returns is None or len(recent_returns) < 5:
        return base_sigma
    rv = _realized_vol(recent_returns, window=20)
    if rv <= 0:
        return base_sigma
    kurt_weight = float(np.clip(kurtosis_excess / 6.0, 0.0, 0.5))
    blended = (1.0 - kurt_weight) * base_sigma + kurt_weight * rv
    return float(np.clip(blended, 0.002, 0.08))


def _innov_gaussian(rng, n_sim, n_steps):
    return rng.standard_normal((n_sim, n_steps))


def _innov_student_t(rng, n_sim, n_steps, kurtosis_excess: float):
    """Student-t innovations rescaled to unit variance (df from excess kurtosis)."""
    if kurtosis_excess <= 0.1:
        df = 30.0
    else:
        df = 4.0 + 6.0 / max(kurtosis_excess, 0.05)
    df = float(np.clip(df, 4.5, 30.0))
    raw = rng.standard_t(df=df, size=(n_sim, n_steps))
    raw *= np.sqrt((df - 2.0) / df)   # variance = df/(df-2)
    return raw


def _simulate_garch(rng, n_sim, n_steps, base_sigma, recent_returns,
                    alpha: float = None, beta: float = None):
    """GARCH(1,1) σ path. Returns (standardised innov, σ-per-step)."""
    alpha = cfg.garch_alpha if alpha is None else alpha
    beta  = cfg.garch_beta  if beta  is None else beta
    if alpha + beta >= 1.0:
        scale = 0.98 / (alpha + beta)
        alpha *= scale
        beta  *= scale
    omega_factor = 1.0 - alpha - beta

    sigma2_lr = base_sigma ** 2
    omega     = omega_factor * sigma2_lr

    if recent_returns is not None and len(recent_returns) >= 5:
        last = np.asarray(recent_returns[-30:], dtype=float)
        sigma2_0 = max(float(np.var(last)), 1e-10)
    else:
        sigma2_0 = sigma2_lr

    sigma2   = np.full(n_sim, sigma2_0)
    eps_prev = np.zeros(n_sim)

    eps_out   = np.zeros((n_sim, n_steps))
    sigma_out = np.zeros((n_sim, n_steps))
    z = rng.standard_normal((n_sim, n_steps))

    for t in range(n_steps):
        sigma2 = omega + alpha * (eps_prev ** 2) + beta * sigma2
        sigma  = np.sqrt(np.maximum(sigma2, 1e-12))
        eps    = sigma * z[:, t]
        eps_out[:, t]   = z[:, t]
        sigma_out[:, t] = sigma
        eps_prev = eps

    return eps_out, sigma_out


def _simulate_bootstrap(rng, n_sim, n_steps, recent_returns: Sequence[float], drift, sigma):
    rets = np.asarray(recent_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size < 10:
        z = rng.standard_normal((n_sim, n_steps))
        return drift + sigma * z
    centred = rets - float(np.mean(rets))
    idx = rng.integers(0, centred.size, size=(n_sim, n_steps))
    return drift + centred[idx]


def _simulate_jump(rng, n_sim, n_steps, drift, sigma,
                   jump_intensity: float = None,
                   jump_mean: float = 0.0,
                   jump_sigma_mult: float = None):
    """Merton jump-diffusion innovations."""
    if jump_intensity is None:
        jump_intensity = cfg.jump_intensity
    if jump_sigma_mult is None:
        jump_sigma_mult = cfg.jump_sigma_mult
    z = rng.standard_normal((n_sim, n_steps))
    diffusion = drift + sigma * z
    jump_mask = rng.random((n_sim, n_steps)) < jump_intensity
    jump_size = jump_mean + (sigma * jump_sigma_mult) * rng.standard_normal((n_sim, n_steps))
    return diffusion + jump_mask * jump_size


def _simulate_ensemble(rng, n_sim, n_steps, base_sigma, drift,
                       recent_returns: Optional[Sequence[float]],
                       kurtosis_excess: float) -> np.ndarray:
    """Adaptive blend of GARCH + bootstrap + jump returns."""
    has_history = recent_returns is not None and len(recent_returns) >= 30
    kurt_norm   = float(np.clip(kurtosis_excess / 4.0, 0.0, 1.0))

    w_garch = 0.45
    w_boot  = 0.35 if has_history else 0.0
    w_jump  = 0.20 + kurt_norm * 0.15
    total   = w_garch + w_boot + w_jump
    w_garch /= total; w_boot /= total; w_jump /= total

    _, sigma_path = _simulate_garch(rng, n_sim, n_steps, base_sigma, recent_returns)
    eps_g     = rng.standard_normal((n_sim, n_steps))
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


# ════════════════════════════════════════════════════════════════════════════
# Microstructure model
#
# Two-phase design:
#   Phase 1: build _MSContext  — calibrate GARCH, derive levels, compute regime
#                                & CVD bias once, before any path is simulated.
#   Phase 2: step the simulation — every step calls ctx.compute_gravity(S) on the
#                                  full price vector (vectorised across n_sim).
# ════════════════════════════════════════════════════════════════════════════

# ─── Hurst exponent ─────────────────────────────────────────────────────────

def _hurst_exponent(ts: Sequence[float]) -> float:
    """Variance-of-lag-differences Hurst estimator. Returns 0.5 on degenerate input."""
    arr = np.asarray(ts, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return 0.5
    max_lag = max(2, min(20, len(arr) // 2))
    lags    = np.arange(2, max_lag)
    tau     = np.array([np.std(arr[lag:] - arr[:-lag]) for lag in lags])
    if np.any(tau <= 0):
        return 0.5
    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(np.clip(slope, 0.0, 1.0))


# ─── Volume profile normalisation ────────────────────────────────────────────

def _normalise_volume_profile(
    vp: Any, p: _MSParams = _PARAMS,
) -> Optional[tuple[float, float, float, list, list]]:
    """
    Coerce whatever the caller passed into (POC, VAH, VAL, HVNs, LVNs).

    Accepts:
      • core.volume_profile.VolumeProfile dataclass      (preferred)
      • {"POC":..., "VAH":..., "VAL":..., "HVNs":[...], "LVNs":[...]}  pre-computed
      • {price_level: volume_at_level}                   raw histogram → derive here
      • None                                             → graceful fallback
    """
    if vp is None:
        return None

    # Case 1 — VolumeProfile dataclass
    if hasattr(vp, "poc") and hasattr(vp, "value_area_high"):
        return (
            float(vp.poc),
            float(vp.value_area_high),
            float(vp.value_area_low),
            [float(x) for x in getattr(vp, "hvn", [])],
            [float(x) for x in getattr(vp, "lvn", [])],
        )

    # Case 2 — pre-computed key-levels dict
    if isinstance(vp, Mapping) and "POC" in vp:
        poc = float(vp["POC"])
        return (
            poc,
            float(vp.get("VAH", poc)),
            float(vp.get("VAL", poc)),
            [float(x) for x in vp.get("HVNs", [])],
            [float(x) for x in vp.get("LVNs", [])],
        )

    # Case 3 — raw {price: volume} → derive levels here
    if isinstance(vp, Mapping):
        items = sorted((float(pp), float(v)) for pp, v in vp.items() if v >= 0)
        if not items:
            return None
        prices  = np.array([pp for pp, _ in items])
        volumes = np.array([v  for _,  v in items])
        if volumes.sum() <= 0:
            return None

        poc_idx = int(np.argmax(volumes))
        poc     = float(prices[poc_idx])
        poc_vol = float(volumes[poc_idx])

        # Value area: expand outward from POC until 70 % captured
        target   = p.value_area_pct * volumes.sum()
        lo = hi  = poc_idx
        captured = volumes[poc_idx]
        while captured < target and (lo > 0 or hi < len(volumes) - 1):
            left  = volumes[lo - 1] if lo > 0 else -np.inf
            right = volumes[hi + 1] if hi < len(volumes) - 1 else -np.inf
            if left >= right and lo > 0:
                lo -= 1; captured += volumes[lo]
            elif hi < len(volumes) - 1:
                hi += 1; captured += volumes[hi]
            else:
                lo -= 1; captured += volumes[lo]

        # Local extrema for HVN / LVN
        median_vol = float(np.median(volumes[volumes > 0])) if np.any(volumes > 0) else 0.0
        hvns, lvns = [], []
        for i in range(1, len(volumes) - 1):
            if (i != poc_idx and
                volumes[i] > volumes[i - 1] and volumes[i] > volumes[i + 1] and
                volumes[i] >= p.hvn_rel_threshold * poc_vol):
                hvns.append(float(prices[i]))
            if (volumes[i] < volumes[i - 1] and volumes[i] < volumes[i + 1] and
                volumes[i] <= p.lvn_rel_threshold * median_vol):
                lvns.append(float(prices[i]))

        return poc, float(prices[hi]), float(prices[lo]), hvns, lvns

    return None


# ─── GARCH MLE for microstructure (richer than the legacy _calibrate_garch) ──

def _calibrate_garch_mle(returns: np.ndarray) -> tuple[float, float, float]:
    """
    Fit GARCH(1,1) by maximum likelihood on a single returns series.
    Returns (omega, alpha, beta). Falls back to config defaults on failure.
    """
    def _nll(params, r):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        n   = len(r)
        eps = r - r.mean()
        s2  = np.empty(n)
        s2[0] = max(np.var(r), 1e-10)
        for i in range(1, n):
            s2[i] = omega + alpha * eps[i - 1] ** 2 + beta * s2[i - 1]
            if s2[i] <= 0:
                return 1e10
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + eps ** 2 / s2)

    try:
        opt = optimize.minimize(
            _nll,
            x0      = [1e-6, cfg.garch_alpha, cfg.garch_beta],
            args    = (returns,),
            method  = "Nelder-Mead",
            options = {"maxiter": 400, "xatol": 1e-8},
        )
        omega, alpha, beta = opt.x
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            raise ValueError("non-stationary GARCH")
        return float(omega), float(alpha), float(beta)
    except Exception:
        alpha, beta = cfg.garch_alpha, cfg.garch_beta
        if alpha + beta >= 0.999:
            beta = max(0.10, 0.94 - alpha)
        omega = max(1e-10, (1.0 - alpha - beta) * float(np.var(returns)))
        return float(omega), float(alpha), float(beta)


# ─── Volume / CVD / regime sub-states ────────────────────────────────────────

def _compute_volume_state(
    volume_history: Optional[Sequence[float]],
    price_history:  Optional[Sequence[float]],
    p: _MSParams = _PARAMS,
) -> tuple[float, bool]:
    """
    Returns (volume_sigma_mult, volume_validates).

    volume_sigma_mult  : multiplier applied to σ globally (1.0 = no change)
    volume_validates   : False when price trends one way and volume the other
    """
    if volume_history is None or len(volume_history) < p.vol_avg_window:
        return 1.0, True
    vols    = np.asarray(volume_history, dtype=float)
    avg_vol = float(np.mean(vols[-p.vol_avg_window:]))
    cur_vol = float(vols[-1])
    ratio   = cur_vol / avg_vol if avg_vol > 0 else 1.0

    if   ratio > p.vol_high_ratio: vol_mult = p.sigma_high_vol
    elif ratio < p.vol_low_ratio:  vol_mult = p.sigma_low_vol
    else:                          vol_mult = 1.0

    validates = True
    if (price_history is not None
            and len(price_history)  >= p.price_trend_window
            and len(volume_history) >= p.price_trend_window):
        x = np.arange(p.price_trend_window)
        v_trend = float(np.polyfit(x, vols[-p.price_trend_window:], 1)[0])
        p_trend = float(np.polyfit(x, np.asarray(price_history)[-p.price_trend_window:], 1)[0])
        validates = (p_trend == 0) or (v_trend * p_trend >= 0)

    return vol_mult, validates


def _compute_cvd_state(
    cvd_history:   Optional[Sequence[float]],
    price_history: Optional[Sequence[float]],
    volume_validates: bool,
    p: _MSParams = _PARAMS,
) -> tuple[float, bool, bool]:
    """
    Returns (drift_bias, accumulation, distribution).

    Logic:
      flat   CVD slope    → drift_bias = 0  (gravity dominates)
      both rising         → strong positive bias
      both falling        → strong negative bias
      CVD up, price down  → accumulation (mild + bias)
      CVD down, price up  → distribution (mild − bias)
    Bias is halved when volume doesn't validate the price trend.
    """
    if cvd_history is None or price_history is None:
        return 0.0, False, False
    cvd = np.asarray(cvd_history,   dtype=float)
    px  = np.asarray(price_history, dtype=float)
    n = min(p.cvd_slope_window, len(cvd), len(px))
    if n < 5:
        return 0.0, False, False

    cvd_recent = cvd[-n:]
    px_recent  = px[-n:]
    x          = np.arange(n)
    cvd_slope  = float(np.polyfit(x, cvd_recent, 1)[0])
    px_slope   = float(np.polyfit(x, px_recent,  1)[0])

    cvd_std    = float(np.std(cvd_recent)) or 1.0
    norm_slope = cvd_slope / cvd_std

    accumulation = distribution = False
    if abs(norm_slope) < p.cvd_flat_threshold:
        bias = 0.0
    elif cvd_slope > 0 and px_slope > 0:
        bias = p.cvd_bias_cap * float(np.tanh(norm_slope))
    elif cvd_slope < 0 and px_slope < 0:
        bias = -p.cvd_bias_cap * float(np.tanh(abs(norm_slope)))
    elif cvd_slope > 0 and px_slope < 0:
        bias = p.cvd_acc_dist_fact * p.cvd_bias_cap
        accumulation = True
    else:
        bias = -p.cvd_acc_dist_fact * p.cvd_bias_cap
        distribution = True

    bias = float(np.clip(bias, -p.cvd_bias_cap, p.cvd_bias_cap))
    if not volume_validates:
        bias *= p.drift_distrust
    return bias, accumulation, distribution


def _compute_regime_state(
    rets: np.ndarray, p: _MSParams = _PARAMS,
) -> tuple[str, float, float, float, float]:
    """Returns (regime_label, hurst, drift_mult, gravity_mult, sigma_mult)."""
    H = _hurst_exponent(rets[-p.hurst_window:]) if rets.size >= p.hurst_window else 0.5
    if   H > p.hurst_trend:
        return "trending",       H, p.reg_trend_drift, p.reg_trend_gravity, p.reg_trend_sigma
    elif H < p.hurst_mean_rev:
        return "mean-reverting", H, p.reg_mr_drift,    p.reg_mr_gravity,    p.reg_mr_sigma
    return "neutral",            H, p.reg_neut_drift,  p.reg_neut_gravity,  p.reg_neut_sigma


# ─── Microstructure simulation context ───────────────────────────────────────

@dataclass
class _MSContext:
    """
    Precomputed state for the microstructure simulation. Built once per call
    (in `_build_ms_context`) and reused at every step.
    """
    # Drift & GARCH
    base_drift:  float
    final_drift: float
    omega:       float
    alpha:       float
    beta:        float
    sigma2_init: float
    eps_init:    float

    # Regime
    regime:        str
    hurst:         float
    drift_mult:    float
    gravity_mult:  float
    sigma_mult:    float

    # Volume profile levels (NaN values when no profile provided)
    poc: float
    vah: float
    val: float
    hvns: np.ndarray
    lvns: np.ndarray

    # Volume scaling
    volume_sigma_mult: float
    volume_validates:  bool

    # CVD state
    cvd_bias:     float
    accumulation: bool
    distribution: bool

    # Tuning params (kept on the context for self-containment)
    params: _MSParams = field(default_factory=lambda: _PARAMS)

    @property
    def has_vp(self) -> bool:
        return np.isfinite(self.poc)

    # ── Vectorised gravity ───────────────────────────────────────────────────
    def compute_gravity(self, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised over the (n_sim,) price vector.

        Returns
        -------
        gravity : np.ndarray  shape (n_sim,)
            Additive log-return contribution per path at this step.
        sigma_mult : np.ndarray  shape (n_sim,)
            Per-path multiplier applied to σ for THIS step only.
        """
        n = S.shape[0]
        if not self.has_vp:
            return np.zeros(n), np.ones(n)

        p = self.params
        rg = self.gravity_mult
        gravity = np.zeros(n)
        smult   = np.ones(n)

        # ── LVN air-pocket: kill gravity, boost σ ─────────────────────────
        in_lvn = np.zeros(n, dtype=bool)
        if self.lvns.size:
            d = np.abs(self.lvns[None, :] - S[:, None]) / np.maximum(self.lvns[None, :], 1e-9)
            in_lvn = np.any(d < p.lvn_proximity_pct, axis=1)

        # ── POC pull (strongest) ──────────────────────────────────────────
        near_poc = np.abs(S - self.poc) / max(self.poc, 1e-9) < p.proximity_pct
        gravity = np.where(
            near_poc, p.poc_gravity * rg * (self.poc - S) / self.poc, gravity,
        )

        # ── Nearest-HVN pull (also damps σ) ───────────────────────────────
        near_hvn = np.zeros(n, dtype=bool)
        if self.hvns.size:
            d        = np.abs(self.hvns[None, :] - S[:, None]) / np.maximum(self.hvns[None, :], 1e-9)
            near_mat = d < p.proximity_pct
            near_hvn = np.any(near_mat, axis=1)
            if near_hvn.any():
                # For each path, find the closest HVN index
                d_masked = np.where(near_mat, d, np.inf)
                idx      = np.argmin(d_masked, axis=1)
                tgt      = self.hvns[idx]
                pull     = p.hvn_gravity * rg * (tgt - S) / tgt
                gravity  = np.where(near_hvn, gravity + pull, gravity)
                smult    = np.where(near_hvn, smult * p.sigma_hvn_damp, smult)

        # ── VAH / VAL mean-reversion at value-area edges ─────────────────
        # Apply at most one (priority: VAH first, then VAL), matching the
        # original "first match wins" semantics from the spec.
        applied_va = np.zeros(n, dtype=bool)
        for level in (self.vah, self.val):
            if not np.isfinite(level):
                continue
            near = (np.abs(S - level) / max(level, 1e-9) < p.proximity_pct) & ~applied_va
            gravity    = np.where(near, gravity + p.va_gravity * rg * (level - S) / level, gravity)
            applied_va = applied_va | near

        # ── Accumulation / distribution side-pull ─────────────────────────
        if self.accumulation:
            for tgt in (self.val, self.poc):
                if not np.isfinite(tgt):
                    continue
                near = np.abs(S - tgt) / max(tgt, 1e-9) < 2 * p.proximity_pct
                gravity = np.where(
                    near, gravity + p.acc_dist_gravity * rg * (tgt - S) / tgt, gravity,
                )
        elif self.distribution:
            for tgt in (self.vah, self.poc):
                if not np.isfinite(tgt):
                    continue
                near = np.abs(S - tgt) / max(tgt, 1e-9) < 2 * p.proximity_pct
                gravity = np.where(
                    near, gravity + p.acc_dist_gravity * rg * (tgt - S) / tgt, gravity,
                )

        # ── Final: LVN paths override everything ─────────────────────────
        if in_lvn.any():
            lvn_smult = p.sigma_lvn_boost * p.lvn_accel_extra
            gravity   = np.where(in_lvn, 0.0,        gravity)
            smult     = np.where(in_lvn, lvn_smult,  smult)

        return gravity, smult

    # ── For surfacing through MCResult ───────────────────────────────────────
    def diagnostics(self) -> dict:
        return {
            "regime":     self.regime,
            "hurst":      float(self.hurst),
            "drift_bias": float(self.cvd_bias),
            "key_levels": {
                "POC":  float(self.poc) if self.has_vp else None,
                "VAH":  float(self.vah) if self.has_vp else None,
                "VAL":  float(self.val) if self.has_vp else None,
                "HVNs": self.hvns.tolist(),
                "LVNs": self.lvns.tolist(),
            },
            "garch":      {"omega": self.omega, "alpha": self.alpha, "beta": self.beta},
        }


def _build_ms_context(
    base_drift:     float,
    base_sigma:     float,
    recent_returns: Optional[Sequence[float]],
    price_history:  Optional[Sequence[float]],
    volume_history: Optional[Sequence[float]],
    cvd_history:    Optional[Sequence[float]],
    volume_profile: Any,
    p: _MSParams = _PARAMS,
) -> _MSContext:
    """Compute every once-per-call value the simulation loop will need."""
    # ── Volume profile levels ───────────────────────────────────────────────
    vp_levels = _normalise_volume_profile(volume_profile, p)
    if vp_levels is not None:
        poc, vah, val, hvns_list, lvns_list = vp_levels
    else:
        poc = vah = val = float("nan")
        hvns_list, lvns_list = [], []
    hvns = np.asarray(hvns_list, dtype=float) if hvns_list else np.empty(0)
    lvns = np.asarray(lvns_list, dtype=float) if lvns_list else np.empty(0)

    # ── GARCH calibration ───────────────────────────────────────────────────
    rets = (np.asarray(recent_returns, dtype=float)
            if recent_returns is not None else np.array([]))
    rets = rets[np.isfinite(rets)] if rets.size else rets

    if rets.size >= 30:
        window = rets[-p.garch_window:]
        omega, alpha, beta = _calibrate_garch_mle(window)
        sigma2_init = max(float(np.var(window)), 1e-10)
        eps_init    = float(rets[-1] - rets.mean())
    else:
        alpha, beta = _calibrate_garch(recent_returns, cfg.garch_alpha, cfg.garch_beta)
        sigma2_init = max(base_sigma ** 2, 1e-10)
        omega       = (1.0 - alpha - beta) * sigma2_init
        eps_init    = 0.0

    # ── Volume σ scaling & validation ───────────────────────────────────────
    vol_sigma_mult, volume_validates = _compute_volume_state(
        volume_history, price_history, p,
    )

    # ── CVD-driven drift bias ───────────────────────────────────────────────
    cvd_bias, accumulation, distribution = _compute_cvd_state(
        cvd_history, price_history, volume_validates, p,
    )

    # ── Hurst regime ────────────────────────────────────────────────────────
    regime, hurst, drift_mult, gravity_mult, sigma_mult = _compute_regime_state(rets, p)

    return _MSContext(
        base_drift        = base_drift,
        final_drift       = base_drift + cvd_bias * drift_mult,
        omega             = omega,
        alpha             = alpha,
        beta              = beta,
        sigma2_init       = sigma2_init,
        eps_init          = eps_init,
        regime            = regime,
        hurst             = hurst,
        drift_mult        = drift_mult,
        gravity_mult      = gravity_mult,
        sigma_mult        = sigma_mult,
        poc               = poc,
        vah               = vah,
        val               = val,
        hvns              = hvns,
        lvns              = lvns,
        volume_sigma_mult = vol_sigma_mult,
        volume_validates  = volume_validates,
        cvd_bias          = cvd_bias,
        accumulation      = accumulation,
        distribution      = distribution,
        params            = p,
    )


def _simulate_microstructure(
    rng,
    n_sim:  int,
    n_steps: int,
    spot:   float,
    ctx:    _MSContext,
) -> np.ndarray:
    """
    Vectorised microstructure path generation.
    Returns paths of shape (n_sim, n_steps + 1) including the spot at column 0.
    """
    p = ctx.params

    # Pre-draw all Student-t shocks at once (standardised to unit variance)
    t_std  = np.sqrt(p.t_df / (p.t_df - 2))
    shocks = rng.standard_t(p.t_df, size=(n_sim, n_steps)) / t_std

    paths       = np.empty((n_sim, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    sigma2      = np.full(n_sim, ctx.sigma2_init)
    eps_prev    = np.full(n_sim, ctx.eps_init)

    for step in range(n_steps):
        # GARCH variance update (vectorised across paths)
        sigma2    = ctx.omega + ctx.alpha * eps_prev ** 2 + ctx.beta * sigma2
        sigma     = np.sqrt(np.maximum(sigma2, 1e-12))
        sigma_eff = sigma * ctx.volume_sigma_mult * ctx.sigma_mult

        # Vectorised gravity & per-path σ adjustment
        grav, smult = ctx.compute_gravity(paths[:, step])
        sigma_eff   = np.clip(sigma_eff * smult, p.sigma_min, p.sigma_max)

        # Total log-return = drift + gravity + (σ_eff × Student-t shock)
        innov   = sigma_eff * shocks[:, step]
        log_ret = ctx.final_drift + grav + innov
        paths[:, step + 1] = paths[:, step] * np.exp(log_ret)
        eps_prev = innov

    return paths


# ════════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════════

def run(
    current_price:   float,
    signal:          Signal,
    n_simulations:   int = 500,
    n_candles:       int = 10,
    model:           str = "garch",
    recent_returns:  Optional[Sequence[float]] = None,
    kurtosis_excess: float = 0.0,
    *,
    # Microstructure-only inputs (ignored by other models). Optional kwargs so
    # all existing callers keep working unchanged.
    price_history:   Optional[Sequence[float]] = None,
    volume_history:  Optional[Sequence[float]] = None,
    cvd_history:     Optional[Sequence[float]] = None,
    volume_profile:  Any = None,
) -> MCResult:
    rng    = np.random.default_rng()
    drift  = float(signal.drift_bias)
    n_sim  = int(max(50, n_simulations))
    n_step = int(max(1, n_candles))

    # Blended volatility (ATR-based + realised), regime-calibrated GARCH α/β,
    # and an empirically-adapted return clip threshold.
    sigma                   = _blend_vol(float(signal.vol_adj), recent_returns, kurtosis_excess)
    garch_alpha, garch_beta = _calibrate_garch(recent_returns, cfg.garch_alpha, cfg.garch_beta)
    clip_val                = _adaptive_clip(recent_returns, float(cfg.mc_clip))

    # ── Microstructure model owns its own loop (price-dependent gravity) ───
    if model == "microstructure":
        ctx = _build_ms_context(
            base_drift     = drift,
            base_sigma     = sigma,
            recent_returns = recent_returns,
            price_history  = price_history,
            volume_history = volume_history,
            cvd_history    = cvd_history,
            volume_profile = volume_profile,
        )
        paths = _simulate_microstructure(rng, n_sim, n_step, float(current_price), ctx)
        paths = np.where(np.isfinite(paths) & (paths > 0), paths, current_price)
        return _build_mc_result(rng, paths, current_price, model, n_sim, ctx=ctx)

    # ── Other models: produce per-step returns, then build paths uniformly ──
    if model == "gaussian":
        eps = _innov_gaussian(rng, n_sim, n_step)
        returns = drift + sigma * eps
    elif model == "student_t":
        eps = _innov_student_t(rng, n_sim, n_step, kurtosis_excess)
        returns = drift + sigma * eps
    elif model == "garch":
        eps, sigma_path = _simulate_garch(rng, n_sim, n_step, sigma, recent_returns,
                                          alpha=garch_alpha, beta=garch_beta)
        returns = drift + sigma_path * eps
    elif model == "bootstrap":
        if recent_returns is None or len(recent_returns) < 10:
            model = "gaussian"
            returns = drift + sigma * _innov_gaussian(rng, n_sim, n_step)
        else:
            returns = _simulate_bootstrap(rng, n_sim, n_step, recent_returns, drift, sigma)
    elif model == "jump":
        returns = _simulate_jump(rng, n_sim, n_step, drift, sigma)
    elif model == "ensemble":
        returns = _simulate_ensemble(rng, n_sim, n_step, sigma, drift,
                                     recent_returns, kurtosis_excess)
    else:
        # unknown → gaussian fallback
        returns = drift + sigma * _innov_gaussian(rng, n_sim, n_step)

    # Build price paths with adaptive clipping
    returns = np.clip(returns, -clip_val, clip_val)
    factors = np.cumprod(1.0 + returns, axis=1)
    paths   = np.hstack([
        np.full((n_sim, 1), current_price, dtype=float),
        current_price * factors,
    ])
    paths = np.where(np.isfinite(paths) & (paths > 0), paths, current_price)

    return _build_mc_result(rng, paths, current_price, model, n_sim)


# ════════════════════════════════════════════════════════════════════════════
# Result builder (shared)
# ════════════════════════════════════════════════════════════════════════════

def _build_mc_result(
    rng,
    paths: np.ndarray,
    current_price: float,
    model: str,
    n_sim: int,
    ctx:   Optional[_MSContext] = None,
) -> MCResult:
    """
    Aggregate a (n_sim, n_steps+1) price-path matrix into MCResult.
    Schema is identical across models; the optional `ctx` argument only attaches
    microstructure diagnostics for that one model.
    """
    final = paths[:, -1]

    # Probabilities ─ wider band for tiny tickers to avoid noise-driven flips
    band = current_price * 0.003
    prob_up   = float(np.mean(final > current_price + band))
    prob_down = float(np.mean(final < current_price - band))
    prob_flat = max(0.0, 1.0 - prob_up - prob_down)   # never negative

    # Final-price percentiles & summary stats — single batched call
    p10, p25, med, p75, p90 = (float(v) for v in np.percentile(final, [10, 25, 50, 75, 90]))
    mean = float(np.mean(final))
    expected_return = (mean / current_price - 1.0) * 100 if current_price else 0.0

    # 5% CVaR on returns
    rets_final = final / current_price - 1.0
    worst_n    = max(1, int(round(n_sim * 0.05)))
    cvar_5     = float(np.mean(np.sort(rets_final)[:worst_n])) * 100

    # Per-step bands — single batched percentile pass
    band_p10, band_p25, median_path_arr, band_p75, band_p90 = np.percentile(
        paths, [10, 25, 50, 75, 90], axis=0,
    )
    median_path = [round(float(v), 4) for v in median_path_arr]

    # Subsample 100 paths for the chart payload
    chart_idx    = rng.choice(n_sim, size=min(100, n_sim), replace=False)
    paths_sample = [[round(float(v), 4) for v in paths[i]] for i in chart_idx]

    # Microstructure diagnostics (None for non-microstructure models)
    ms = ctx.diagnostics() if ctx is not None else None

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
        paths_full      = paths,
        median_path     = median_path,
        model           = model,
        ms_regime       = ms["regime"]     if ms else None,
        ms_hurst        = ms["hurst"]      if ms else None,
        ms_drift_bias   = ms["drift_bias"] if ms else None,
        ms_key_levels   = ms["key_levels"] if ms else None,
    )


# ════════════════════════════════════════════════════════════════════════════
# CVD computation helper
# ════════════════════════════════════════════════════════════════════════════

def compute_cvd_from_ohlc(
    opens:   Sequence[float],
    closes:  Sequence[float],
    volumes: Sequence[float],
) -> np.ndarray:
    """
    Bar-classified Cumulative Volume Delta.

      +volume on up-bars   (close > open)
      −volume on down-bars (close < open)
       0      on dojis     (close == open)

    Returns the cumulative sum — same length as the inputs.
    """
    o = np.asarray(opens,   dtype=float)
    c = np.asarray(closes,  dtype=float)
    v = np.asarray(volumes, dtype=float)
    delta = np.where(c > o, v, np.where(c < o, -v, 0.0))
    return np.cumsum(delta)
