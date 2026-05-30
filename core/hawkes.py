"""Hawkes process zone-touch excitation model."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_EVENT_THRESH_SIGMA = 1.5  # events = |return| > 1.5 × rolling σ
_ROLLING_WINDOW = 20  # window for rolling σ
_MAX_ITER = 200  # MLE iterations
_MIN_EVENTS = 8  # minimum events needed to fit


@dataclass
class HawkesParams:
    mu: float  # baseline intensity
    alpha: float  # excitation
    beta: float  # decay
    branching_ratio: float  # α/β - must be < 1 for stability
    n_events: int  # number of events used in fitting
    fit_ok: bool  # True if MLE converged


@dataclass
class ZoneReaction:
    level: float
    zone_type: str  # "demand" | "supply"
    current_lambda: float  # Hawkes intensity at this moment
    bounce_prob: float  # 0-1
    break_prob: float  # 0-1
    consolidate_prob: float  # 0-1
    excitement_label: str  # "calm" | "moderate" | "excited"
    explanation: str


@dataclass
class HawkesResult:
    params: HawkesParams
    current_lambda: float
    excitement_label: str  # "calm" | "moderate" | "excited"
    lambda_percentile: float  # where current λ sits in historical distribution
    zone_reactions: list[ZoneReaction]
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "params": {
                "mu": round(self.params.mu, 6),
                "alpha": round(self.params.alpha, 6),
                "beta": round(self.params.beta, 6),
                "branching_ratio": round(self.params.branching_ratio, 4),
                "n_events": self.params.n_events,
                "fit_ok": self.params.fit_ok,
            },
            "current_lambda": round(self.current_lambda, 6),
            "excitement_label": self.excitement_label,
            "lambda_percentile": round(self.lambda_percentile, 1),
            "error": self.error,
            "zone_reactions": [
                {
                    "level": round(z.level, 4),
                    "zone_type": z.zone_type,
                    "current_lambda": round(z.current_lambda, 6),
                    "bounce_prob": round(z.bounce_prob, 3),
                    "break_prob": round(z.break_prob, 3),
                    "consolidate_prob": round(z.consolidate_prob, 3),
                    "excitement_label": z.excitement_label,
                    "explanation": z.explanation,
                }
                for z in self.zone_reactions
            ],
        }


def _extract_events(returns: np.ndarray) -> np.ndarray:
    """
    Return the *indices* of candles whose |return| exceeds _EVENT_THRESH_SIGMA
    standard deviations of the rolling window.
    """
    n = len(returns)
    if n < _ROLLING_WINDOW + 2:
        return np.array([], dtype=float)

    events = []
    for i in range(_ROLLING_WINDOW, n):
        window = returns[i - _ROLLING_WINDOW : i]
        sigma = float(np.std(window, ddof=1))
        if sigma > 0 and abs(returns[i]) > _EVENT_THRESH_SIGMA * sigma:
            events.append(float(i))  # use bar index as "time"

    return np.array(events, dtype=float)


def _hawkes_loglik(params: np.ndarray, times: np.ndarray) -> float:
    """
    Negative log-likelihood of a univariate Hawkes process.
    params = [log_mu, log_alpha, log_beta]  (log-space to keep positivity)
    """
    mu, alpha, beta = np.exp(params)
    if alpha >= beta:
        return 1e10  # enforce n < 1

    len(times)
    T = times[-1] - times[0]

    # Recursive intensity computation for efficiency O(n)
    A = 0.0  # A(i) = Σ_{j<i} exp(-β(t_i - t_j))
    ll = 0.0

    for i, ti in enumerate(times):
        if i == 0:
            A = 0.0
        else:
            A = math.exp(-beta * (ti - times[i - 1])) * (A + 1.0)

        lam_i = mu + alpha * A
        if lam_i <= 0:
            return 1e10
        ll += math.log(lam_i)

    # Compensator integral ∫₀ᵀ λ(t) dt = μT + α/β · Σ_i (1 - exp(-β(T-tᵢ)))
    integral = mu * T + (alpha / beta) * sum(1.0 - math.exp(-beta * (times[-1] - t)) for t in times)

    return -(ll - integral)


def _fit_hawkes(event_times: np.ndarray) -> HawkesParams:
    """
    MLE fit of Hawkes parameters using scipy.optimize.minimize.
    Falls back to moment-matching heuristics if scipy or MLE fails.
    """
    n = len(event_times)
    if n < _MIN_EVENTS:
        return HawkesParams(mu=0.05, alpha=0.1, beta=0.5, branching_ratio=0.2, n_events=n, fit_ok=False)

    # Normalise times to [0, N] so numerical scale is manageable
    t0 = event_times[0]
    t_norm = event_times - t0

    try:
        from scipy.optimize import minimize

        best_ll = float("inf")
        best_x = np.array([-3.0, -2.0, -0.5])  # log(mu), log(alpha), log(beta)

        # Try several starting points to avoid local minima
        starts = [
            [-3.0, -2.0, -0.5],
            [-2.0, -1.5, -0.2],
            [-4.0, -2.5, -0.8],
        ]
        for s in starts:
            res = minimize(
                _hawkes_loglik,
                np.array(s),
                args=(t_norm,),
                method="Nelder-Mead",
                options={"maxiter": _MAX_ITER, "xatol": 1e-5, "fatol": 1e-5},
            )
            if res.fun < best_ll:
                best_ll = res.fun
                best_x = res.x

        mu, alpha, beta = np.exp(best_x)
        br = alpha / beta
        ok = (0 < br < 0.99) and (mu > 0) and (alpha > 0) and (beta > 0)

        if not ok:
            raise ValueError(f"Unstable solution: α/β={br:.3f}")

        return HawkesParams(
            mu=float(mu),
            alpha=float(alpha),
            beta=float(beta),
            branching_ratio=float(br),
            n_events=n,
            fit_ok=True,
        )

    except Exception as exc:
        logger.debug("[Hawkes] MLE failed (%s), using heuristics", exc)
        return _heuristic_params(t_norm, n)


def _heuristic_params(t_norm: np.ndarray, n_events: int) -> HawkesParams:
    """Simple moment-matching fallback when scipy/MLE is unavailable."""
    T = float(t_norm[-1]) if len(t_norm) > 1 else float(n_events)
    mu = n_events / (T + 1e-9) * 0.6  # baseline ~ 60% of observed rate
    alpha = mu * 0.4  # excitation ~ 40% of baseline
    beta = alpha * 2.5  # branching ratio ~ 0.4
    return HawkesParams(
        mu=float(mu),
        alpha=float(alpha),
        beta=float(beta),
        branching_ratio=float(alpha / beta),
        n_events=n_events,
        fit_ok=False,
    )


def _compute_lambda(params: HawkesParams, event_times: np.ndarray, query_time: float) -> float:
    """
    Evaluate λ(query_time) given the Hawkes parameters and observed event times.
    """
    mu, alpha, beta = params.mu, params.alpha, params.beta
    past_events = event_times[event_times < query_time]
    excitation = sum(alpha * math.exp(-beta * (query_time - t)) for t in past_events)
    return float(mu + excitation)


def _historical_lambda_dist(params: HawkesParams, event_times: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Compute λ at N evenly-spaced times to build a reference distribution."""
    if len(event_times) < 2:
        return np.array([params.mu])
    T = event_times[-1]
    times = np.linspace(0, T, n_points)
    return np.array([_compute_lambda(params, event_times, t) for t in times])


def _zone_probs(
    lam: float, lam_lo: float, lam_hi: float, zone_type: str
) -> tuple[float, float, float, str, str]:
    """
    Map current λ to (bounce, break, consolidate) probabilities.

    Logic:
      Calm market (λ near baseline): clean bounce at demand / clean rejection at supply.
      Excited market (λ very high): zones tend to get overwhelmed → break-through.
      Moderate: consolidation / chop.
    """
    # Percentile of current λ in historical distribution
    span = (lam_hi - lam_lo) + 1e-12
    pct = min(max((lam - lam_lo) / span, 0.0), 1.0)

    if pct < 0.33:
        label = "calm"
        if zone_type == "demand":
            bounce, brk, cons = 0.65, 0.12, 0.23
            expl = "Low activity - demand zone likely to hold; clean bounce expected."
        else:
            bounce, brk, cons = 0.62, 0.14, 0.24
            expl = "Low activity - supply zone likely to reject; clean sell-off expected."
    elif pct < 0.67:
        label = "moderate"
        if zone_type == "demand":
            bounce, brk, cons = 0.38, 0.22, 0.40
            expl = "Moderate activity - likely consolidation; watch for volume confirmation."
        else:
            bounce, brk, cons = 0.36, 0.24, 0.40
            expl = "Moderate activity - possible consolidation under supply; unclear direction."
    else:
        label = "excited"
        if zone_type == "demand":
            bounce, brk, cons = 0.22, 0.58, 0.20
            expl = "High excitation - market is active; demand zone at risk of breaking."
        else:
            bounce, brk, cons = 0.20, 0.60, 0.20
            expl = "High excitation - strong directional pressure; supply may not hold."

    return bounce, brk, cons, label, expl


def analyse_hawkes(
    returns: Sequence[float],
    zones: list[dict] | None = None,  # list of {"level": float, "zone_type": str}
) -> HawkesResult:
    """
    Fit a Hawkes process to the return series and compute zone-reaction
    probabilities for each supplied zone.

    `returns` should be log-returns or % returns of recent candles.
    `zones`   is a list of demand/supply zone dicts from core.zones.
    """
    try:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 20:
            raise ValueError("Too few returns")

        # Extract event times
        event_times = _extract_events(arr)
        if len(event_times) < _MIN_EVENTS:
            # Fallback: use all large-return bars with lower threshold
            thresh = float(np.std(arr, ddof=1)) * 1.0
            event_times = np.where(np.abs(arr) > thresh)[0].astype(float)

        if len(event_times) < 2:
            return _fallback_result(zones)

        params = _fit_hawkes(event_times)

        # Current intensity = at last bar
        current_lambda = _compute_lambda(params, event_times, float(len(arr) - 1))

        # Historical λ distribution for percentile labelling
        hist_lam = _historical_lambda_dist(params, event_times)
        lam_lo = float(np.percentile(hist_lam, 5))
        lam_hi = float(np.percentile(hist_lam, 95))
        lam_pct = float(np.mean(hist_lam <= current_lambda) * 100)

        if lam_pct < 33:
            exc_label = "calm"
        elif lam_pct < 67:
            exc_label = "moderate"
        else:
            exc_label = "excited"

        zone_reactions: list[ZoneReaction] = []
        for z in zones or []:
            lv = float(z.get("level", 0))
            ztype = str(z.get("zone_type", "demand"))
            zl = _compute_lambda(params, event_times, float(len(arr) - 1))
            b, br, c, lbl, expl = _zone_probs(zl, lam_lo, lam_hi, ztype)
            zone_reactions.append(
                ZoneReaction(
                    level=lv,
                    zone_type=ztype,
                    current_lambda=float(zl),
                    bounce_prob=b,
                    break_prob=br,
                    consolidate_prob=c,
                    excitement_label=lbl,
                    explanation=expl,
                )
            )

        return HawkesResult(
            params=params,
            current_lambda=current_lambda,
            excitement_label=exc_label,
            lambda_percentile=lam_pct,
            zone_reactions=zone_reactions,
        )

    except Exception as exc:
        logger.warning("[Hawkes] analyse failed: %s", exc)
        return _fallback_result(zones, error=str(exc)[:120])


def _fallback_result(zones, error: str = "insufficient_data") -> HawkesResult:
    """Return a neutral result when fitting fails."""
    zone_reactions = []
    for z in zones or []:
        lv = float(z.get("level", 0))
        ztype = str(z.get("zone_type", "demand"))
        zone_reactions.append(
            ZoneReaction(
                level=lv,
                zone_type=ztype,
                current_lambda=0.0,
                bounce_prob=0.45,
                break_prob=0.30,
                consolidate_prob=0.25,
                excitement_label="unknown",
                explanation="Insufficient data for Hawkes fit - showing neutral priors.",
            )
        )
    return HawkesResult(
        params=HawkesParams(mu=0.0, alpha=0.0, beta=0.0, branching_ratio=0.0, n_events=0, fit_ok=False),
        current_lambda=0.0,
        excitement_label="unknown",
        lambda_percentile=50.0,
        zone_reactions=zone_reactions,
        error=error,
    )
