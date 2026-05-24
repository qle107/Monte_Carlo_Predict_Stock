"""
core/hurst.py
Detrended Fluctuation Analysis (DFA) exponent estimator.

References
──────────
Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides.
  Physical Review E, 49(2), 1685–1689.  doi:10.1103/PhysRevE.49.1685

Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
  Journal of the American Statistical Association, 89(428), 1303–1313.

Algorithm (DFA-1)
─────────────────
1. Compute the integrated centred series: Y(i) = Σ_{k≤i} (x_k − x̄)
2. For each box size n in {4, 8, 16, …, N//4}:
     a. Split Y into ⌊N/n⌋ non-overlapping windows.
     b. In each window, fit a linear trend with OLS.
     c. Compute F(n) = sqrt( mean of squared residuals over all windows ).
3. Regress log F(n) on log n using OLS → slope α is the DFA exponent.
4. Return (α, se_α) where se_α is the OLS standard error of the slope.

Interpretation
──────────────
  α < 0.45  anti-persistent (mean-reverting)
  α ≈ 0.50  white noise
  α > 0.55  long-range correlated (trending)
  α ≈ 1.00  random walk (1/f noise) – expected for price LEVELS
  α > 1.0   non-stationary (e.g. Brownian motion)

Usage
─────
  Call dfa(np.log(prices))  for the microstructure / regime model.
  Call dfa(log_returns)     for stationary series analysis.
"""

from __future__ import annotations

import numpy as np


def dfa(
    series: np.ndarray,
    min_box: int = 4,
    max_box: int | None = None,
) -> tuple[float, float]:
    """Detrended Fluctuation Analysis (DFA-1) exponent.

    Parameters
    ----------
    series:
        1-D array of values. For price-level inputs pass ``np.log(prices)``;
        for log-return inputs pass the returns directly.
    min_box:
        Smallest box size (default 4).  Must be ≥ 4.
    max_box:
        Largest box size (default ``len(series) // 4``).

    Returns
    -------
    alpha:
        DFA exponent (float).  Clipped to [0.0, 2.0].
    se_alpha:
        OLS standard error of the log-log slope.  Returns 0.0 on degenerate
        input (fewer than 3 valid box-sizes).

    Notes
    -----
    Pure NumPy — no scipy dependency.  Typically runs in < 1 ms for N = 2 000.
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_pts = len(arr)

    # ── Guard: need at least 2*min_box points to form a single box ────────────
    if n_pts < 2 * min_box:
        return 0.5, 0.0

    _max_box = min_box if max_box is None else max_box
    if max_box is None:
        _max_box = max(min_box + 1, n_pts // 4)

    # ── Step 1: Integrate the centred series ──────────────────────────────────
    centred = arr - float(np.mean(arr))
    Y = np.cumsum(centred)  # shape (N,)

    # ── Step 2: Compute F(n) for each box size ────────────────────────────────
    box_sizes: list[int] = []
    f_values: list[float] = []

    # Generate powers-of-2 box sizes from min_box up to _max_box (inclusive)
    n = min_box
    while n <= _max_box:
        n_windows = n_pts // n
        if n_windows < 1:
            break

        residuals_sq: list[float] = []
        for w in range(n_windows):
            segment = Y[w * n : (w + 1) * n]
            x = np.arange(n, dtype=float)
            # OLS fit: y = a + b*x
            x_mean = (n - 1) / 2.0
            xy_cov = float(np.dot(x - x_mean, segment - segment.mean()))
            x_var = float(np.dot(x - x_mean, x - x_mean))
            if x_var <= 0:
                continue
            b = xy_cov / x_var
            a = float(segment.mean()) - b * x_mean
            trend = a + b * x
            residuals_sq.append(float(np.mean((segment - trend) ** 2)))

        if residuals_sq:
            f_n = float(np.sqrt(np.mean(residuals_sq)))
            if f_n > 0:
                box_sizes.append(n)
                f_values.append(f_n)

        n *= 2  # next power of 2

    # ── Step 3: OLS on log-log ────────────────────────────────────────────────
    if len(box_sizes) < 3:
        return 0.5, 0.0

    log_n = np.log(box_sizes, dtype=float)
    log_f = np.log(f_values, dtype=float)

    # OLS: α = Cov(log_n, log_f) / Var(log_n)
    n_pts_reg = len(log_n)
    x_mean = float(np.mean(log_n))
    y_mean = float(np.mean(log_f))
    ss_xy = float(np.dot(log_n - x_mean, log_f - y_mean))
    ss_xx = float(np.dot(log_n - x_mean, log_n - x_mean))

    if ss_xx <= 0:
        return 0.5, 0.0

    alpha = ss_xy / ss_xx

    # ── Standard error of slope (OLS SE) ─────────────────────────────────────
    y_pred = y_mean + alpha * (log_n - x_mean)
    residuals = log_f - y_pred
    mse = float(np.sum(residuals**2)) / max(1, n_pts_reg - 2)
    se_alpha = float(np.sqrt(mse / ss_xx)) if mse >= 0 else 0.0

    return float(np.clip(alpha, 0.0, 2.0)), se_alpha
