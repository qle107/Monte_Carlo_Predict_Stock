"""
core/hmm_regime.py — Hidden Markov Model for probabilistic market regime states.

Pure NumPy/SciPy implementation — NO compiled extensions required.
Works on any Python version without a C++ compiler.

Fits a Gaussian HMM via the Baum-Welch (EM) algorithm to log-return data
and identifies latent market states.

States (auto-labelled by learned properties after fitting)
----------------------------------------------------------
  "trending"  : high |mean| return, moderate vol → momentum; zones may break
  "ranging"   : near-zero mean, low vol → mean-reversion; zones tend to hold
  "volatile"  : high vol, mixed mean → unpredictable; elevated risk

Zone-reaction conditional priors
---------------------------------
  Ranging  → bounce: ~65%,  break: ~15%,  consolidate: ~20%
  Trending → bounce: ~28%,  break: ~52%,  consolidate: ~20%
  Volatile → bounce: ~35%,  break: ~35%,  consolidate: ~30%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

N_STATES    = 3
MIN_BARS    = 40
MAX_ITER    = 80        # Baum-Welch iterations
TOL         = 1e-4      # convergence tolerance on log-likelihood
N_RESTARTS  = 3         # random restarts to avoid local optima

# ── Zone reaction priors per regime ─────────────────────────────────────────
_PRIORS: Dict[str, Dict[str, float]] = {
    "ranging":  {"bounce": 0.65, "break": 0.15, "consolidate": 0.20},
    "trending": {"bounce": 0.28, "break": 0.52, "consolidate": 0.20},
    "volatile": {"bounce": 0.35, "break": 0.35, "consolidate": 0.30},
    "unknown":  {"bounce": 0.40, "break": 0.30, "consolidate": 0.30},
}


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class HMMState:
    label:       str
    mean_return: float
    volatility:  float
    probability: float          # P(currently in this state)


@dataclass
class HMMResult:
    current_state:     str
    state_probs:       List[HMMState]
    zone_priors:       Dict[str, float]
    transition_matrix: List[List[float]]
    fit_method:        str
    n_bars_used:       int
    error:             Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "current_state": self.current_state,
            "fit_method":    self.fit_method,
            "n_bars_used":   self.n_bars_used,
            "error":         self.error,
            "zone_priors": {
                "bounce":      round(self.zone_priors.get("bounce",      0.40), 3),
                "break":       round(self.zone_priors.get("break",       0.30), 3),
                "consolidate": round(self.zone_priors.get("consolidate", 0.30), 3),
            },
            "state_probs": [
                {
                    "label":       s.label,
                    "mean_return": round(s.mean_return, 5),
                    "volatility":  round(s.volatility,  5),
                    "probability": round(s.probability, 4),
                }
                for s in self.state_probs
            ],
            "transition_matrix": [
                [round(v, 4) for v in row]
                for row in self.transition_matrix
            ],
        }


# ─── Gaussian HMM (pure NumPy Baum-Welch) ────────────────────────────────────

def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian pdf — safe against zero sigma."""
    sigma = max(sigma, 1e-9)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _forward(x: np.ndarray, pi: np.ndarray, A: np.ndarray,
             mus: np.ndarray, sigs: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled forward algorithm.
    Returns alpha (T × K) and scale factors c (T,).
    """
    T, K = len(x), len(pi)
    alpha = np.zeros((T, K))
    c     = np.zeros(T)

    # t = 0
    alpha[0] = pi * np.array([_gauss_pdf(x[0], mus[k], sigs[k]) for k in range(K)])
    c[0] = alpha[0].sum()
    if c[0] < 1e-300:
        c[0] = 1e-300
    alpha[0] /= c[0]

    for t in range(1, T):
        b = np.array([_gauss_pdf(x[t], mus[k], sigs[k]) for k in range(K)])
        alpha[t] = (alpha[t - 1] @ A) * b
        c[t] = alpha[t].sum()
        if c[t] < 1e-300:
            c[t] = 1e-300
        alpha[t] /= c[t]

    return alpha, c


def _backward(x: np.ndarray, c: np.ndarray, A: np.ndarray,
              mus: np.ndarray, sigs: np.ndarray) -> np.ndarray:
    """Scaled backward algorithm. Returns beta (T × K)."""
    T, K = len(x), A.shape[0]
    beta = np.zeros((T, K))
    beta[-1] = 1.0

    for t in range(T - 2, -1, -1):
        b = np.array([_gauss_pdf(x[t + 1], mus[k], sigs[k]) for k in range(K)])
        beta[t] = (A @ (b * beta[t + 1])) / c[t + 1]

    return beta


def _baum_welch(x: np.ndarray, K: int, seed: int = 0
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run Baum-Welch EM for a K-state Gaussian HMM on sequence x.

    Returns (pi, A, mus, sigs, log_likelihood).
    """
    rng = np.random.default_rng(seed)
    T   = len(x)

    # ── Initialise parameters ──────────────────────────────────────────────
    pi   = np.ones(K) / K
    A    = rng.dirichlet(np.ones(K) * 5, size=K)   # row-stochastic
    A   /= A.sum(axis=1, keepdims=True)

    # K-means-like init for means
    idx  = rng.choice(T, K, replace=False)
    mus  = x[idx].copy()
    sigs = np.full(K, float(np.std(x)) or 0.01)

    prev_ll = -np.inf

    for iteration in range(MAX_ITER):
        # E-step
        alpha, c = _forward(x, pi, A, mus, sigs)
        beta      = _backward(x, c, A, mus, sigs)

        gamma = alpha * beta                          # (T, K)
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.where(gamma_sum < 1e-300, 1e-300, gamma_sum)
        gamma /= gamma_sum

        # xi (T-1, K, K)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            b = np.array([_gauss_pdf(x[t + 1], mus[k], sigs[k]) for k in range(K)])
            xi[t] = (alpha[t][:, None] * A * (b * beta[t + 1])[None, :])
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum

        # M-step
        pi   = gamma[0] / gamma[0].sum()
        A    = xi.sum(axis=0)
        A   /= (A.sum(axis=1, keepdims=True) + 1e-300)

        gamma_k = gamma.sum(axis=0) + 1e-9           # (K,)
        mus     = (gamma * x[:, None]).sum(axis=0) / gamma_k
        sigs    = np.sqrt(
            (gamma * (x[:, None] - mus[None, :]) ** 2).sum(axis=0) / gamma_k
        )
        sigs = np.maximum(sigs, 1e-6)

        # Log-likelihood (sum of log scale factors)
        ll = float(np.log(np.maximum(c, 1e-300)).sum())
        if abs(ll - prev_ll) < TOL:
            break
        prev_ll = ll

    return pi, A, mus, sigs, prev_ll


def _label_states(mus: np.ndarray, sigs: np.ndarray) -> List[str]:
    """Auto-label states: highest σ → volatile, highest |μ| → trending, rest → ranging."""
    K      = len(mus)
    labels = ["unknown"] * K
    remaining = set(range(K))

    # 1. Volatile = highest σ
    vol_idx = int(np.argmax(sigs))
    labels[vol_idx] = "volatile"
    remaining.discard(vol_idx)

    if remaining:
        abs_mus   = {i: abs(float(mus[i])) for i in remaining}
        trend_idx = max(abs_mus, key=abs_mus.get)
        labels[trend_idx] = "trending"
        remaining.discard(trend_idx)

    for i in remaining:
        labels[i] = "ranging"

    return labels


def _fit_hmm(returns: np.ndarray) -> HMMResult:
    """Fit Gaussian HMM with multiple random restarts, pick best log-likelihood."""
    x = returns.astype(float)

    best_ll  = -np.inf
    best_res = None

    for seed in range(N_RESTARTS):
        try:
            pi, A, mus, sigs, ll = _baum_welch(x, N_STATES, seed=seed)
            if ll > best_ll:
                best_ll  = ll
                best_res = (pi, A, mus, sigs)
        except Exception as exc:
            logger.debug("[HMM] restart %d failed: %s", seed, exc)

    if best_res is None:
        raise RuntimeError("All Baum-Welch restarts failed")

    pi, A, mus, sigs = best_res
    labels = _label_states(mus, sigs)

    # Compute final posteriors (γ at last time step)
    alpha, _ = _forward(x, pi, A, mus, sigs)
    last_gamma = alpha[-1]
    last_gamma = last_gamma / (last_gamma.sum() + 1e-300)

    current_idx   = int(np.argmax(last_gamma))
    current_label = labels[current_idx]

    state_list = [
        HMMState(
            label=labels[i],
            mean_return=float(mus[i]),
            volatility=float(sigs[i]),
            probability=float(last_gamma[i]),
        )
        for i in range(N_STATES)
    ]
    state_list.sort(key=lambda s: -s.probability)

    logger.debug(
        "[HMM] state=%s ll=%.2f (probs: %s)",
        current_label, best_ll,
        {s.label: round(s.probability, 2) for s in state_list},
    )

    return HMMResult(
        current_state=current_label,
        state_probs=state_list,
        zone_priors=_PRIORS.get(current_label, _PRIORS["unknown"]),
        transition_matrix=A.tolist(),
        fit_method="hmm_baum_welch",
        n_bars_used=len(returns),
    )


# ─── Heuristic fallback ──────────────────────────────────────────────────────

def _heuristic_regime(returns: np.ndarray) -> HMMResult:
    """
    Volatility/momentum 3-way classifier — O(n), no fitting needed.

    • Recent σ > 75th-percentile rolling σ  →  volatile
    • |recent mean| > 0.8 × recent σ        →  trending
    • otherwise                              →  ranging
    """
    n = len(returns)
    w = min(20, n // 2)
    if w < 2:
        return _unknown_result(n, "too_few_bars")

    recent  = returns[-w:]
    sigma   = float(np.std(recent, ddof=1)) or 1e-9
    mu      = float(np.mean(recent))

    roll_stds = [float(np.std(returns[max(0, i - w): i], ddof=1))
                 for i in range(w, n)]
    sigma_75 = float(np.percentile(roll_stds, 75)) if roll_stds else sigma

    if sigma > sigma_75 * 1.2:
        label = "volatile"
    elif abs(mu) > 0.8 * sigma:
        label = "trending"
    else:
        label = "ranging"

    global_sig = float(np.std(returns, ddof=1)) or 1e-9
    state_list = [
        HMMState(label=lbl,
                 mean_return=mu if lbl == label else 0.0,
                 volatility=sigma if lbl == label else global_sig * 0.5,
                 probability=0.70 if lbl == label else 0.15)
        for lbl in ("ranging", "trending", "volatile")
    ]
    state_list.sort(key=lambda s: -s.probability)

    identity = [[1.0 / N_STATES] * N_STATES for _ in range(N_STATES)]
    return HMMResult(
        current_state=label,
        state_probs=state_list,
        zone_priors=_PRIORS.get(label, _PRIORS["unknown"]),
        transition_matrix=identity,
        fit_method="heuristic",
        n_bars_used=n,
    )


def _unknown_result(n: int, error: str) -> HMMResult:
    state_list = [
        HMMState(label="ranging",  mean_return=0.0, volatility=0.01, probability=0.50),
        HMMState(label="trending", mean_return=0.0, volatility=0.02, probability=0.30),
        HMMState(label="volatile", mean_return=0.0, volatility=0.03, probability=0.20),
    ]
    identity = [[1.0 / N_STATES] * N_STATES for _ in range(N_STATES)]
    return HMMResult(
        current_state="unknown",
        state_probs=state_list,
        zone_priors=_PRIORS["unknown"],
        transition_matrix=identity,
        fit_method="none",
        n_bars_used=n,
        error=error,
    )


# ─── Public entry point ───────────────────────────────────────────────────────

def analyse_hmm(returns: Sequence[float]) -> HMMResult:
    """
    Fit Gaussian HMM (Baum-Welch EM, pure NumPy) and identify current market regime.
    Falls back to heuristic classifier on any error.
    """
    try:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]

        if len(arr) < MIN_BARS:
            return _unknown_result(len(arr), "too_few_bars")

        try:
            return _fit_hmm(arr)
        except Exception as exc:
            logger.debug("[HMM] Baum-Welch failed (%s) — using heuristic", exc)
            return _heuristic_regime(arr)

    except Exception as exc:
        logger.warning("[HMM] analyse_hmm error: %s", exc)
        try:
            arr = np.asarray(returns, dtype=float)
            arr = arr[np.isfinite(arr)]
            return _heuristic_regime(arr) if len(arr) >= 10 else _unknown_result(0, str(exc))
        except Exception:
            return _unknown_result(0, str(exc)[:80])


# ─── Regime-blended zone probability ─────────────────────────────────────────

def blend_zone_probability(
    hmm: HMMResult,
    hawkes_probs: Optional[Dict[str, float]] = None,
    zone_strength: float = 0.5,
) -> Dict[str, float]:
    """
    Combine HMM regime priors with Hawkes excitation and zone strength.

    zone_strength : 0..1 score from core.zones (higher = more reliable zone).
    """
    base = dict(hmm.zone_priors)

    if hawkes_probs:
        w_hmm, w_hk = 0.60, 0.40
        blended = {
            k: w_hmm * base.get(k, 0.33) + w_hk * hawkes_probs.get(k, 0.33)
            for k in ("bounce", "break", "consolidate")
        }
    else:
        blended = base.copy()

    # Zone strength nudge: stronger zone → tilt toward bounce (max ±7.5%)
    bonus = (zone_strength - 0.5) * 0.15
    blended["bounce"]      = min(1.0, max(0.0, blended["bounce"]      + bonus))
    blended["break"]       = min(1.0, max(0.0, blended["break"]       - bonus * 0.6))
    blended["consolidate"] = min(1.0, max(0.0, blended["consolidate"] - bonus * 0.4))

    total = sum(blended.values()) or 1.0
    return {k: round(v / total, 4) for k, v in blended.items()}
