"""
core/__init__.py
Public API for the core analysis pipeline.
Call `analyse(df)` to get everything in one shot.
"""

from dataclasses import asdict
from .fetcher    import fetch_candles, get_latest_price
from .indicators import compute_indicators
from .signal     import compute_signal
from .montecarlo import run as run_mc


def analyse(df, n_simulations: int = 500, n_forward: int = 10) -> dict:
    ind           = compute_indicators(df)
    sig           = compute_signal(ind)
    current_price = float(df["close"].iloc[-1])
    mc            = run_mc(current_price, sig, n_simulations, n_forward)

    candles = [
        {"t": ts.isoformat(), "o": round(float(row["open"]),4),
         "h": round(float(row["high"]),4), "l": round(float(row["low"]),4),
         "c": round(float(row["close"]),4), "v": int(row["volume"])}
        for ts, row in df.iterrows()
    ]

    attrs       = getattr(df, "attrs", {})
    session     = attrs.get("session",     "unknown")
    session_now = attrs.get("session_now", "unknown")
    extended    = attrs.get("extended",    False)
    warnings = []
    if sig.gap_warning:
        warnings.append(sig.gap_warning)

    return {
        "current_price": current_price,
        "indicators":    asdict(ind),
        "signal":        asdict(sig),
        "mc":            asdict(mc),
        "candles":       candles,
        "warnings":      warnings,
        "session":       session,
        "session_now":   session_now,
        "extended":      extended,
    }
