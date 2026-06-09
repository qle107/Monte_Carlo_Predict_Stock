"""Higher-timeframe confirmation helper."""

from __future__ import annotations

import logging

from core.data.fetcher import fetch_candles

from .indicators import compute_indicators
from .regime import detect_regime
from .signal import compute_signal

logger = logging.getLogger(__name__)

_HTF_MAP = {
    "1m": "15m",
    "2m": "15m",
    "5m": "1h",
    "15m": "1h",
    "30m": "4h",
    "1h": "1d",
    "4h": "1d",
    "1d": "1d",  # already daily - skip HTF
}


async def htf_confirmation(ticker: str, base_interval: str, extended: bool, loop) -> dict:
    """Fetch higher timeframe regime + signal snapshot."""
    htf = _HTF_MAP.get(base_interval)
    if not htf or htf == base_interval:
        return {"available": False, "reason": "no higher timeframe"}
    try:
        df = await loop.run_in_executor(None, fetch_candles, ticker, htf, 60, extended)
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(None, detect_regime, df, ind.adx, ind.obv_slope)
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)
        return {
            "available": True,
            "interval": htf,
            "regime": reg.regime,
            "trend_score": reg.trend_score,
            "potential_up": reg.potential_up,
            "potential_down": reg.potential_down,
            "signal_label": sig.label,
            "composite": sig.composite,
            "confidence": sig.confidence,
            "rsi": ind.rsi,
            "adx": ind.adx,
            "ema_cross": ind.ema_cross,
        }
    except Exception as e:
        logger.debug("HTF confirmation failed (%s %s): %s", ticker, htf, e)
        return {"available": False, "reason": str(e)}
