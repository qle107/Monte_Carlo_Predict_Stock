"""
core/__init__.py
Public API for the core analysis pipeline.

   from core import analyse
   result = analyse(df, n_sim=500, n_forward=10, mc_model="garch")

`result` is a JSON-serialisable dict with everything the dashboard needs:
  • current_price
  • indicators       (RSI, MACD, ADX, Bollinger, OBV, …)
  • regime           (strong_uptrend | range_bound | breakout_up | …)
  •     potential_up / potential_down / potential_flat (each 0..100)
  •     verdict — plain-English summary
  • signal           (composite, confidence, drift_bias, label, …)
  • mc               (Monte Carlo projection: probs, percentiles, paths)
  • candles          (raw OHLCV payload for the chart)
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .fetcher    import fetch_candles, get_latest_price
from .indicators import compute_indicators
from .regime     import detect_regime
from .signal     import compute_signal
from .montecarlo import run as run_mc


def analyse(
    df: pd.DataFrame,
    n_simulations: int = 500,
    n_forward:     int = 10,
    mc_model:      str = "garch",
) -> dict:
    ind  = compute_indicators(df)
    reg  = detect_regime(df, adx=ind.adx, obv_slope=ind.obv_slope)
    sig  = compute_signal(ind, regime=reg)         # regime-aware

    current_price = float(df["close"].iloc[-1])
    mc = run_mc(
        current_price,
        sig,
        n_simulations   = n_simulations,
        n_candles       = n_forward,
        model           = mc_model,
        recent_returns  = ind.returns,
        kurtosis_excess = ind.kurtosis,
    )

    candles = [
        {"t": ts.isoformat(),
         "o": round(float(row["open"]),  4),
         "h": round(float(row["high"]),  4),
         "l": round(float(row["low"]),   4),
         "c": round(float(row["close"]), 4),
         "v": int(row["volume"])}
        for ts, row in df.iterrows()
    ]

    attrs       = getattr(df, "attrs", {})
    session     = attrs.get("session",     "unknown")
    session_now = attrs.get("session_now", "unknown")
    extended    = attrs.get("extended",    False)

    warnings = []
    if sig.gap_warning:
        warnings.append(sig.gap_warning)

    # Strip the heavy returns array from the indicators payload (already used by MC)
    ind_dict = asdict(ind)
    ind_dict.pop("returns", None)

    return {
        "current_price": current_price,
        "indicators":    ind_dict,
        "regime":        asdict(reg),
        "signal":        asdict(sig),
        "mc":            asdict(mc),
        "candles":       candles,
        "warnings":      warnings,
        "session":       session,
        "session_now":   session_now,
        "extended":      extended,
    }
