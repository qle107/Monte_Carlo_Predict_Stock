"""
core/indicators.py
All indicator calculations with full NaN guards.
Every function returns a safe fallback value if data is insufficient.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Indicators:
    rsi:          float
    slope:        float
    momentum:     float
    ema_fast:     float
    ema_slow:     float
    ema_cross:    str
    atr_pct:      float
    gap_pct:      float
    is_gap_up:    bool
    is_gap_down:  bool
    mean_return:  float
    std_return:   float
    skewness:     float
    trend_bias:   float
    vol_regime:   str


def _safe(val, fallback=0.0):
    """Return fallback if val is NaN, None, or infinite."""
    try:
        if val is None:
            return fallback
        f = float(val)
        return fallback if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return fallback


def _returns(closes: np.ndarray) -> np.ndarray:
    if len(closes) < 2:
        return np.array([0.0])
    rets = np.diff(closes) / closes[:-1]
    # Remove any NaN/inf that sneak in from bad data
    rets = rets[np.isfinite(rets)]
    return rets if len(rets) > 0 else np.array([0.0])


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = float(deltas[deltas > 0].sum())
    losses = float(-deltas[deltas < 0].sum())
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return _safe(100 - 100 / (1 + rs), 50.0)


def _ema(closes: np.ndarray, period: int) -> float:
    if len(closes) < 2:
        return _safe(closes[-1]) if len(closes) > 0 else 0.0
    try:
        val = pd.Series(closes.astype(float)).ewm(span=period, adjust=False).mean().iloc[-1]
        return _safe(val, float(closes[-1]))
    except Exception:
        return float(closes[-1]) if len(closes) > 0 else 0.0


def _slope(closes: np.ndarray, n: int = 8) -> float:
    n = min(n, len(closes))
    if n < 2:
        return 0.0
    try:
        y = closes[-n:].astype(float)
        x = np.arange(n, dtype=float)
        m = np.polyfit(x, y, 1)[0]
        base = float(closes[-1])
        return _safe(m / base * 100, 0.0) if base != 0 else 0.0
    except Exception:
        return 0.0


def _momentum(closes: np.ndarray, n: int = 5) -> float:
    if len(closes) <= n:
        return 0.0
    prev = float(closes[-1 - n])
    if prev == 0:
        return 0.0
    return _safe((float(closes[-1]) - prev) / prev * 100, 0.0)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < 2:
        return 1.0
    try:
        highs  = df["high"].values.astype(float)
        lows   = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        trs = []
        for i in range(1, len(df)):
            tr = max(
                highs[i]  - lows[i],
                abs(highs[i]  - closes[i - 1]),
                abs(lows[i]   - closes[i - 1]),
            )
            if np.isfinite(tr):
                trs.append(tr)
        if not trs:
            return 1.0
        atr  = float(np.mean(trs[-period:]))
        base = float(closes[-1])
        return _safe(atr / base * 100, 1.0) if base != 0 else 1.0
    except Exception:
        return 1.0


def _gap(df: pd.DataFrame) -> tuple:
    if len(df) < 2:
        return 0.0, False, False
    try:
        prev  = float(df["close"].iloc[-2])
        curr  = float(df["open"].iloc[-1])
        if prev == 0:
            return 0.0, False, False
        gap = _safe((curr - prev) / prev * 100, 0.0)
        return gap, gap > 3.0, gap < -3.0
    except Exception:
        return 0.0, False, False


def _mean_return(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    return _safe(float(np.mean(rets)) * 100, 0.0)


def _std_return(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 1.0   # safe fallback: 1% vol
    val = float(np.std(rets)) * 100
    return _safe(val, 1.0) if val > 0 else 1.0


def _skewness(rets: np.ndarray) -> float:
    if len(rets) < 4:
        return 0.0
    try:
        return _safe(float(pd.Series(rets).skew()), 0.0)
    except Exception:
        return 0.0


def _trend_bias(closes: np.ndarray) -> float:
    if len(closes) < 2:
        return 0.5
    ups   = int(np.sum(np.diff(closes) > 0))
    total = len(closes) - 1
    return _safe(ups / total, 0.5) if total > 0 else 0.5


def _vol_regime(rets: np.ndarray, w_recent: int = 10, w_long: int = 30) -> str:
    if len(rets) < w_recent + 1:
        return "normal"
    try:
        recent_vol = float(np.std(rets[-w_recent:]))
        long_vol   = float(np.std(rets[-w_long:]) if len(rets) >= w_long else np.std(rets))
        if long_vol == 0 or not np.isfinite(long_vol) or not np.isfinite(recent_vol):
            return "normal"
        ratio = recent_vol / long_vol
        if ratio > 1.5: return "high"
        if ratio < 0.6: return "low"
        return "normal"
    except Exception:
        return "normal"


def compute_indicators(df: pd.DataFrame) -> Indicators:
    closes = df["close"].values.astype(float)
    # Remove any non-finite closes
    closes = closes[np.isfinite(closes)]
    if len(closes) == 0:
        closes = np.array([1.0])

    rets  = _returns(closes)
    ema_f = _ema(closes, 9)
    ema_s = _ema(closes, 21)

    cross = "neutral"
    if ema_f > 0 and ema_s > 0:
        if   ema_f > ema_s * 1.001: cross = "bullish"
        elif ema_f < ema_s * 0.999: cross = "bearish"

    gap, gap_up, gap_down = _gap(df)

    return Indicators(
        rsi         = round(_rsi(closes),         2),
        slope       = round(_slope(closes),        4),
        momentum    = round(_momentum(closes),     4),
        ema_fast    = round(ema_f,                 4),
        ema_slow    = round(ema_s,                 4),
        ema_cross   = cross,
        atr_pct     = round(_atr_pct(df),          4),
        gap_pct     = round(gap,                   2),
        is_gap_up   = gap_up,
        is_gap_down = gap_down,
        mean_return = round(_mean_return(rets),    5),
        std_return  = round(_std_return(rets),     5),
        skewness    = round(_skewness(rets),       3),
        trend_bias  = round(_trend_bias(closes),   3),
        vol_regime  = _vol_regime(rets),
    )
