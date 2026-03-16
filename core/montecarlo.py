"""
core/montecarlo.py
Vectorised Monte Carlo price simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from .signal import Signal


@dataclass
class MCResult:
    prob_up:      float          # % of paths ending above entry + threshold
    prob_flat:    float
    prob_down:    float
    median_price: float
    p10_price:    float          # worst 10th percentile
    p90_price:    float          # best 90th percentile
    paths:        List[List[float]]   # 100-path sample for charting
    median_path:  List[float]


def run(
    current_price: float,
    signal: Signal,
    n_simulations: int = 500,
    n_candles: int = 10,
) -> MCResult:
    rng = np.random.default_rng()

    # shape: (n_simulations, n_candles)
    daily_returns = (
        signal.drift_bias +
        signal.vol_adj * rng.standard_normal((n_simulations, n_candles))
    )

    # Compound forward: col 0 = current_price, cols 1..n = projected
    factors      = np.cumprod(1 + daily_returns, axis=1)
    paths_matrix = np.hstack([
        np.full((n_simulations, 1), current_price),
        current_price * factors
    ])

    final_prices = paths_matrix[:, -1]

    # Flat band = ±0.3% of entry price (avoids noise flips)
    band = current_price * 0.003
    prob_up   = float(np.mean(final_prices > current_price + band))
    prob_down = float(np.mean(final_prices < current_price - band))
    prob_flat = 1.0 - prob_up - prob_down

    sorted_f     = np.sort(final_prices)
    median_price = float(np.median(final_prices))
    p10          = float(sorted_f[int(n_simulations * 0.10)])
    p90          = float(sorted_f[int(n_simulations * 0.90)])

    # Median path = path whose final price is closest to the median
    med_idx     = int(np.argmin(np.abs(final_prices - median_price)))
    median_path = [round(float(v), 4) for v in paths_matrix[med_idx]]

    # Subsample 100 paths for the chart
    idx_sample = rng.choice(n_simulations, size=min(100, n_simulations), replace=False)
    paths_sample = [
        [round(float(v), 4) for v in paths_matrix[i]]
        for i in idx_sample
    ]

    return MCResult(
        prob_up      = round(prob_up   * 100, 1),
        prob_flat    = round(prob_flat * 100, 1),
        prob_down    = round(prob_down * 100, 1),
        median_price = round(median_price, 4),
        p10_price    = round(p10, 4),
        p90_price    = round(p90, 4),
        paths        = paths_sample,
        median_path  = median_path,
    )
