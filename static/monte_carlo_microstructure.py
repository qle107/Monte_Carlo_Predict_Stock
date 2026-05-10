"""
DEPRECATED — this standalone module has been folded into core/montecarlo.py.

The microstructure-aware Monte Carlo simulation now lives behind the standard
project entry point:

    from core import analyse
    result = analyse(df, mc_model="microstructure")

Or directly:

    from core.montecarlo import run as run_mc, compute_cvd_from_ohlc
    mc = run_mc(
        current_price, signal,
        model           = "microstructure",
        recent_returns  = ind.returns,
        kurtosis_excess = ind.kurtosis,
        price_history   = df["close"].to_numpy(),
        volume_history  = df["volume"].to_numpy(),
        cvd_history     = compute_cvd_from_ohlc(df["open"], df["close"], df["volume"]),
        volume_profile  = compute_volume_profile(df),
    )

This file is kept only to redirect any old imports.
"""

from core.montecarlo import (  # noqa: F401  (re-export)
    run as run_monte_carlo,
    compute_cvd_from_ohlc as compute_cvd,
)
