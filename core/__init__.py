"""Core analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .indicators import compute_indicators
from .montecarlo import compute_cvd_from_ohlc
from .montecarlo import run as run_mc
from .regime import detect_regime
from .signal import compute_signal
from .volume_profile import compute_volume_profile


def _df_to_candles(df: pd.DataFrame) -> list:
    """Serialize OHLCV rows for JSON."""
    ts_list = [t.isoformat() for t in df.index]
    o = df["open"].round(4).tolist()
    h = df["high"].round(4).tolist()
    lo = df["low"].round(4).tolist()
    c = df["close"].round(4).tolist()
    v = df["volume"].astype(int).tolist()
    return [
        {"t": t, "o": oi, "h": hi, "l": li, "c": ci, "v": vi}
        for t, oi, hi, li, ci, vi in zip(ts_list, o, h, lo, c, v, strict=False)
    ]


def analyse(
    df: pd.DataFrame,
    n_simulations: int = 10000,
    n_forward: int = 10,
    mc_model: str = "garch",
    band_alpha: float = 0.20,
) -> dict:
    ind = compute_indicators(df)
    reg = detect_regime(df, adx=ind.adx, obv_slope=ind.obv_slope)
    sig = compute_signal(ind, regime=reg)

    current_price = float(df["close"].iloc[-1])

    # Microstructure inputs for MC (volume profile also returned in the payload).
    vp = compute_volume_profile(df)
    cvd_history = compute_cvd_from_ohlc(df["open"], df["close"], df["volume"])
    price_history = df["close"].to_numpy(dtype=float)
    volume_history = df["volume"].to_numpy(dtype=float)

    mc = run_mc(
        current_price,
        sig,
        n_simulations=n_simulations,
        n_candles=n_forward,
        model=mc_model,
        recent_returns=ind.returns,
        kurtosis_excess=ind.kurtosis,
        price_history=price_history,
        volume_history=volume_history,
        cvd_history=cvd_history,
        volume_profile=vp,
        band_alpha=band_alpha,
    )

    candles = _df_to_candles(df)
    attrs = getattr(df, "attrs", {})
    session = attrs.get("session", "unknown")
    session_now = attrs.get("session_now", "unknown")
    extended = attrs.get("extended", False)

    warnings = []
    if sig.gap_warning:
        warnings.append(sig.gap_warning)

    ind_dict = asdict(ind)
    ind_dict.pop("returns", None)  # already consumed by MC

    mc_dict = asdict(mc)
    mc_dict.pop("paths_full", None)  # numpy array - not JSON-safe

    return {
        "current_price": current_price,
        "indicators": ind_dict,
        "regime": asdict(reg),
        "signal": asdict(sig),
        "mc": mc_dict,
        "volume_profile": vp.to_dict() if vp else None,
        "candles": candles,
        "warnings": warnings,
        "session": session,
        "session_now": session_now,
        "extended": extended,
        "_mc_paths_full": mc.paths_full,  # popped in server.py before broadcast
    }
