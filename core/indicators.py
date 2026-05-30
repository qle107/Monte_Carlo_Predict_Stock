"""Technical indicator computation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import cfg

@dataclass
class Indicators:
    # legacy
    rsi: float
    slope: float
    momentum: float
    ema_fast: float
    ema_slow: float
    ema_cross: str
    atr_pct: float
    gap_pct: float
    is_gap_up: bool
    is_gap_down: bool
    mean_return: float
    std_return: float
    skewness: float
    trend_bias: float
    vol_regime: str

    # v2 indicators
    macd: float
    macd_signal: float
    macd_hist: float
    bb_position: float  # -1 (below lower band) … +1 (above upper)
    adx: float  # 0..100; >25 ≈ trending
    obv_slope: float  # %/candle of OBV slope (volume momentum)
    vwap_dist: float  # (price - vwap) / price * 100
    kurtosis: float  # excess kurtosis of returns

    # v3 indicators
    rsi_divergence: float  # +1 bullish div, -1 bearish div, 0 none
    vol_of_vol: float  # std of rolling vol (vol regime instability)
    price_vs_52w: float  # % distance from 52-week high (negative = below)
    ema_200: float  # 200-bar EMA (long-term trend anchor)
    ema200_dist: float  # % distance of price from EMA200

    returns: list[float] = field(default_factory=list)  # historical returns (decimal)

def _safe(val, fallback: float = 0.0) -> float:
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
    rets = rets[np.isfinite(rets)]
    return rets if len(rets) > 0 else np.array([0.0])

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1) :])
    gains = float(deltas[deltas > 0].sum())
    losses = float(-deltas[deltas < 0].sum())
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return _safe(100 - 100 / (1 + rs), 50.0)

def _ema(closes: np.ndarray, period: int) -> float:
    return _safe(
        _ema_series(closes, period)[-1] if len(closes) else 0.0, float(closes[-1]) if len(closes) else 0.0
    )

def _ema_series(closes: np.ndarray, period: int) -> np.ndarray:
    """Full EMA series (used by MACD)."""
    if len(closes) == 0:
        return np.array([0.0])
    return pd.Series(closes.astype(float)).ewm(span=period, adjust=False).mean().to_numpy()

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
    """
    Wilder's ATR - uses Wilder's exponential smoothing (alpha = 1/period),
    identical to the formula used in _adx for consistency.
    """
    if len(df) < 2:
        return 1.0
    try:
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        tr = np.maximum.reduce(
            [
                h - l,
                np.abs(h - prev_c),
                np.abs(l - prev_c),
            ]
        )
        tr = tr[np.isfinite(tr)]
        if tr.size == 0:
            return 1.0
        # Wilder's smoothing: EMA with alpha = 1/period
        atr_series = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
        atr = float(atr_series[-1])
        base = float(c[-1])
        return _safe(atr / base * 100, 1.0) if base != 0 else 1.0
    except Exception:
        return 1.0

def _gap(df: pd.DataFrame, threshold: float = 3.0) -> tuple:
    if len(df) < 2:
        return 0.0, False, False
    try:
        prev = float(df["close"].iloc[-2])
        curr = float(df["open"].iloc[-1])
        if prev == 0:
            return 0.0, False, False
        gap = _safe((curr - prev) / prev * 100, 0.0)
        return gap, gap > threshold, gap < -threshold
    except Exception:
        return 0.0, False, False

def _mean_return(rets: np.ndarray) -> float:
    return _safe(float(np.mean(rets)) * 100, 0.0) if len(rets) else 0.0

def _std_return(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 1.0
    val = float(np.std(rets)) * 100
    return _safe(val, 1.0) if val > 0 else 1.0

def _skewness(rets: np.ndarray) -> float:
    if len(rets) < 4:
        return 0.0
    try:
        return _safe(float(pd.Series(rets).skew()), 0.0)
    except Exception:
        return 0.0

def _kurtosis(rets: np.ndarray) -> float:
    """Excess kurtosis (Fisher). 0 for Normal; >0 = fat tails."""
    if len(rets) < 4:
        return 0.0
    try:
        return _safe(float(pd.Series(rets).kurt()), 0.0)
    except Exception:
        return 0.0

def _trend_bias(closes: np.ndarray) -> float:
    if len(closes) < 2:
        return 0.5
    ups = int(np.sum(np.diff(closes) > 0))
    total = len(closes) - 1
    return _safe(ups / total, 0.5) if total > 0 else 0.5

def _vol_regime(rets: np.ndarray, w_recent: int = 10, w_long: int = 30) -> str:
    if len(rets) < w_recent + 1:
        return "normal"
    try:
        recent_vol = float(np.std(rets[-w_recent:]))
        long_vol = float(np.std(rets[-w_long:]) if len(rets) >= w_long else np.std(rets))
        if long_vol == 0 or not np.isfinite(long_vol) or not np.isfinite(recent_vol):
            return "normal"
        ratio = recent_vol / long_vol
        if ratio > 1.5:
            return "high"
        if ratio < 0.6:
            return "low"
        return "normal"
    except Exception:
        return "normal"

def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9):
    """Returns (macd_line, signal_line, histogram) - last values only."""
    if len(closes) < slow + sig:
        return 0.0, 0.0, 0.0
    try:
        ema_f = _ema_series(closes, fast)
        ema_s = _ema_series(closes, slow)
        macd_line = ema_f - ema_s
        signal_line = pd.Series(macd_line).ewm(span=sig, adjust=False).mean().to_numpy()
        hist = macd_line - signal_line
        # Express as % of price for scale-invariance
        base = float(closes[-1]) or 1.0
        return (
            _safe(macd_line[-1] / base * 100, 0.0),
            _safe(signal_line[-1] / base * 100, 0.0),
            _safe(hist[-1] / base * 100, 0.0),
        )
    except Exception:
        return 0.0, 0.0, 0.0

def _bollinger_position(closes: np.ndarray, period: int = 20, k: float = 2.0) -> float:
    """
    Position within Bollinger bands.
       0   ≈ on the SMA
      +1   ≈ at the upper band     (overbought)
      -1   ≈ at the lower band     (oversold)
      |x|>1 means price punched through.
    """
    if len(closes) < period:
        return 0.0
    try:
        window = closes[-period:].astype(float)
        sma = float(np.mean(window))
        std = float(np.std(window))
        if std == 0 or not np.isfinite(std):
            return 0.0
        z = (float(closes[-1]) - sma) / (k * std)
        return _safe(float(np.clip(z, -3.0, 3.0)), 0.0)
    except Exception:
        return 0.0

def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Average Directional Index - trend strength (0..100).
    Wilder's smoothing approximated with EMA for simplicity (good enough).
    """
    if len(df) < period * 2 + 1:
        return 0.0
    try:
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)

        up_move = np.diff(h)
        down_move = -np.diff(l)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_c = c[:-1]
        tr = np.maximum.reduce(
            [
                h[1:] - l[1:],
                np.abs(h[1:] - prev_c),
                np.abs(l[1:] - prev_c),
            ]
        )

        atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
        atr_safe = np.where(atr == 0, 1e-12, atr)

        plus_di = 100 * pd.Series(plus_dm / atr_safe).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
        minus_di = 100 * pd.Series(minus_dm / atr_safe).ewm(alpha=1 / period, adjust=False).mean().to_numpy()

        denom = plus_di + minus_di
        denom_safe = np.where(denom == 0, 1e-12, denom)
        dx = 100 * np.abs(plus_di - minus_di) / denom_safe
        adx = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().to_numpy()

        return _safe(float(adx[-1]), 0.0)
    except Exception:
        return 0.0

def _obv_slope(df: pd.DataFrame, n: int = 14) -> float:
    """
    Slope of On-Balance Volume over the last n candles, normalised
    by current OBV magnitude. Positive = volume confirming uptrend.
    Returned as %/candle (analogous to closes' slope).
    """
    if len(df) < max(n, 3) + 1:
        return 0.0
    try:
        c = df["close"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        signs = np.sign(np.diff(c))
        obv = np.concatenate([[0.0], np.cumsum(signs * v[1:])])
        if len(obv) < n:
            return 0.0
        y = obv[-n:].astype(float)
        x = np.arange(n, dtype=float)
        m = np.polyfit(x, y, 1)[0]
        base = float(np.max(np.abs(y))) or 1.0
        return _safe(m / base * 100, 0.0)
    except Exception:
        return 0.0

def _vwap_distance(df: pd.DataFrame, n_bars: int = 26) -> float:
    """
    Distance of current price from session VWAP (%, signed).
    Approximate VWAP = sum(typical_price * volume) / sum(volume) over the
    last ~1 trading day worth of bars.
    """
    if len(df) < 5:
        return 0.0
    try:
        n = min(len(df), n_bars)  # ~ one trading day at 15m
        sub = df.tail(n)
        tp = (sub["high"] + sub["low"] + sub["close"]).to_numpy(float) / 3.0
        v = sub["volume"].to_numpy(float)
        v_sum = float(np.sum(v))
        if v_sum <= 0 or not np.isfinite(v_sum):
            return 0.0
        vwap = float(np.sum(tp * v) / v_sum)
        price = float(df["close"].iloc[-1])
        if price == 0:
            return 0.0
        return _safe((price - vwap) / price * 100, 0.0)
    except Exception:
        return 0.0

def _rsi_series_vectorized(c: np.ndarray, p: int) -> np.ndarray:
    """
    Vectorised RSI series using Wilder EWM smoothing.
    ~50× faster than the previous Python for-loop implementation.
    """
    s = pd.Series(c.astype(float))
    delta = s.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / p, adjust=False).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / p, adjust=False).mean()
    # Avoid division by zero: where loss==0, RSI=100 (all gains) or 50 (no movement)
    rs = gain / loss.replace(0.0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + rs)).fillna(np.where(gain > 0, 100.0, 50.0))
    return rsi.to_numpy(dtype=float)

def _rsi_divergence(closes: np.ndarray, period: int = 14, lookback: int = 30) -> float:
    """
    Detect RSI divergence over the last `lookback` bars.
    Returns: +1.0 bullish (price lower low, RSI higher low),
             -1.0 bearish (price higher high, RSI lower high),
              0.0 no clear divergence.
    """
    n = min(lookback, len(closes))
    if n < period + 5:
        return 0.0
    try:
        window = closes[-n:]
        rsi_arr = _rsi_series_vectorized(window, period)
        mid = n // 2

        price_first_half = window[:mid]
        price_second_half = window[mid:]
        rsi_first_half = rsi_arr[:mid]
        rsi_second_half = rsi_arr[mid:]

        # Bullish: price makes lower low but RSI makes higher low
        if price_second_half.min() < price_first_half.min() and rsi_second_half.min() > rsi_first_half.min():
            return 1.0
        # Bearish: price makes higher high but RSI makes lower high
        if price_second_half.max() > price_first_half.max() and rsi_second_half.max() < rsi_first_half.max():
            return -1.0
        return 0.0
    except Exception:
        return 0.0

def _vol_of_vol(rets: np.ndarray, window: int = 10, n_windows: int = 5) -> float:
    """
    Volatility of volatility: std of rolling realized vols.
    High VoV = unstable vol regime → regime change warning.
    Normalised to [0..1] using a rough empirical scale.
    """
    if len(rets) < window * n_windows:
        return 0.0
    try:
        vols = []
        for i in range(n_windows):
            start = -(window * (n_windows - i))
            end = -(window * (n_windows - i - 1)) or None
            sub = rets[start:end]
            if len(sub) >= 3:
                vols.append(float(np.std(sub)))
        if len(vols) < 2:
            return 0.0
        vov = float(np.std(vols))
        return _safe(float(np.clip(vov / 0.01, 0.0, 1.0)), 0.0)
    except Exception:
        return 0.0

def _price_vs_52w(closes: np.ndarray) -> float:
    """
    % distance of current price from the highest close in the window.
    Negative means below the 52-week (window) high.
    At all-time-high = 0. Down 10% from ATH = -10.0.
    """
    if len(closes) < 2:
        return 0.0
    try:
        high = float(np.max(closes))
        cur = float(closes[-1])
        if high <= 0:
            return 0.0
        return _safe((cur / high - 1.0) * 100.0, 0.0)
    except Exception:
        return 0.0

def _ema200_distance(closes: np.ndarray) -> tuple:
    """Returns (ema200_value, pct_distance_of_price_from_ema200)."""
    if len(closes) < 10:
        return float(closes[-1]), 0.0
    try:
        ema200 = float(_ema_series(closes, min(200, len(closes)))[-1])
        cur = float(closes[-1])
        if ema200 <= 0:
            return ema200, 0.0
        dist = _safe((cur / ema200 - 1.0) * 100.0, 0.0)
        return round(ema200, 4), round(dist, 3)
    except Exception:
        return float(closes[-1]), 0.0

def compute_indicators(df: pd.DataFrame) -> Indicators:
    closes = df["close"].to_numpy(float)
    closes = closes[np.isfinite(closes)]
    if len(closes) == 0:
        closes = np.array([1.0])

    rets = _returns(closes)
    ema_f = _ema(closes, cfg.ema_fast)
    ema_s = _ema(closes, cfg.ema_slow)
    ema200, ema200_dist = _ema200_distance(closes)

    cross = "neutral"
    if ema_f > 0 and ema_s > 0:
        if ema_f > ema_s * 1.001:
            cross = "bullish"
        elif ema_f < ema_s * 0.999:
            cross = "bearish"

    gap, gap_up, gap_down = _gap(df, threshold=cfg.gap_threshold)
    macd_l, macd_s, macd_h = _macd(closes, fast=cfg.macd_fast, slow=cfg.macd_slow, sig=cfg.macd_signal)

    return Indicators(
        rsi=round(_rsi(closes, period=cfg.rsi_period), 2),
        slope=round(_slope(closes, n=cfg.slope_period), 4),
        momentum=round(_momentum(closes, n=cfg.mom_period), 4),
        ema_fast=round(ema_f, 4),
        ema_slow=round(ema_s, 4),
        ema_cross=cross,
        atr_pct=round(_atr_pct(df, period=cfg.atr_period), 4),
        gap_pct=round(gap, 2),
        is_gap_up=gap_up,
        is_gap_down=gap_down,
        mean_return=round(_mean_return(rets), 5),
        std_return=round(_std_return(rets), 5),
        skewness=round(_skewness(rets), 3),
        trend_bias=round(_trend_bias(closes), 3),
        vol_regime=_vol_regime(rets),
        macd=round(macd_l, 4),
        macd_signal=round(macd_s, 4),
        macd_hist=round(macd_h, 4),
        bb_position=round(_bollinger_position(closes, period=cfg.bb_period, k=cfg.bb_k), 3),
        adx=round(_adx(df, period=cfg.adx_period), 2),
        obv_slope=round(_obv_slope(df, n=cfg.obv_period), 4),
        vwap_dist=round(_vwap_distance(df, n_bars=cfg.vwap_period), 3),
        kurtosis=round(_kurtosis(rets), 3),
        # v3
        rsi_divergence=round(
            _rsi_divergence(closes, period=cfg.rsi_period, lookback=cfg.rsi_div_lookback), 1
        ),
        vol_of_vol=round(_vol_of_vol(rets), 3),
        price_vs_52w=round(_price_vs_52w(closes), 2),
        ema_200=ema200,
        ema200_dist=ema200_dist,
        returns=[float(r) for r in rets[-200:]],  # cap stored
    )
