"""core/regime.py — Market regime detector.

Components: Hurst R/S, multi-window linear R², Donchian breakout, HH/HL pattern, ADX, OBV slope.
Outputs a Regime with: label, verdict, potential_up/down/flat (0–100),
trend_score (−1…+1), range_score (0…1), breakout flags, Hurst, R², Donchian, HH/HL counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from config import cfg


@dataclass
class Regime:
    regime: str
    verdict: str

    potential_up: float  # 0..100
    potential_down: float
    potential_flat: float

    trend_score: float  # −1 (strong down) … +1 (strong up)
    range_score: float  #  0 (trending)    …  1 (range-bound)
    breakout_up: bool
    breakout_down: bool

    hurst: float  # 0.5 random walk, >0.6 trending, <0.4 mean-rev
    r2_short: float  # 10-bar regression R²
    r2_mid: float  # 20-bar regression R²
    r2_long: float  # 50-bar regression R²

    donchian_pos: float  # −1 at lower band, +1 at upper
    donchian_high: float
    donchian_low: float

    hh_count: int
    hl_count: int
    lh_count: int
    ll_count: int

    range_compression: float  # 0 = wide, 1 = very tight

    components: dict[str, float] = field(default_factory=dict)


def _hurst(series: np.ndarray, max_lag: int = 20) -> float:
    """Rescaled-range Hurst exponent. H≈0.5 random walk, >0.6 trending, <0.4 mean-rev."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max_lag + 5:
        return 0.5
    try:
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            diffs = s[lag:] - s[:-lag]
            std = np.std(diffs)
            if std <= 0 or not np.isfinite(std):
                continue
            tau.append(std)
        if len(tau) < 5:
            return 0.5
        log_lags = np.log(np.arange(2, 2 + len(tau)))
        log_tau = np.log(tau)
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        if not np.isfinite(slope):
            return 0.5
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def _r2_and_slope(closes: np.ndarray, n: int) -> tuple[float, float]:
    """Linear regression R² and slope (%/candle) over the last n closes."""
    n = min(n, len(closes))
    if n < 4:
        return 0.0, 0.0
    try:
        y = closes[-n:].astype(float)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0 or not np.isfinite(ss_tot):
            return 0.0, 0.0
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        base = float(closes[-1]) or 1.0
        slope_pct = slope / base * 100
        return float(r2), float(slope_pct)
    except Exception:
        return 0.0, 0.0


def _donchian(df: pd.DataFrame, n: int = 20) -> tuple[float, float, float, bool, bool]:
    """
    Returns (position, high, low, breakout_up, breakout_down).
    position: −1 at lower band, +1 at upper.
    Checks primary (N-bar), secondary (10-bar), and post-breakout consolidation (within 3% of high).
    """
    if len(df) < n + 2:
        c = float(df["close"].iloc[-1]) if len(df) else 0.0
        return 0.0, c, c, False, False
    try:
        prev = df.iloc[-(n + 1) : -1]
        hi = float(prev["high"].max())
        lo = float(prev["low"].min())
        cur_close = float(df["close"].iloc[-1])
        cur_high = float(df["high"].iloc[-1])
        cur_low = float(df["low"].iloc[-1])

        rng = max(hi - lo, 1e-9)
        pos = (cur_close - (hi + lo) / 2.0) / (rng / 2.0)
        pos = float(np.clip(pos, -1.5, 1.5))

        brk_up = cur_high > hi
        brk_down = cur_low < lo

        # Secondary: 10-bar window catches recent breakouts
        if not brk_up and len(df) >= 12:
            prev10 = df.iloc[-11:-1]
            hi10 = float(prev10["high"].max())
            lo10 = float(prev10["low"].min())
            if cur_high > hi10:
                brk_up = True
            if cur_low < lo10:
                brk_down = True

        # Tertiary: within 3% of N-bar high = post-breakout consolidation
        if not brk_up and hi > 0 and (hi - cur_close) / hi < 0.03 and cur_close > (hi + lo) / 2.0:
            brk_up = True

        return pos, hi, lo, brk_up, brk_down
    except Exception:
        c = float(df["close"].iloc[-1]) if len(df) else 0.0
        return 0.0, c, c, False, False


def _swing_pivots(highs: np.ndarray, lows: np.ndarray, lookback: int = 3) -> tuple[list[float], list[float]]:
    """Pivot highs/lows: a bar is a pivot if it's max (min) of the ±lookback window."""
    n = len(highs)
    pivots_h, pivots_l = [], []
    for i in range(lookback, n - lookback):
        win_h = highs[i - lookback : i + lookback + 1]
        win_l = lows[i - lookback : i + lookback + 1]
        if highs[i] == np.max(win_h):
            pivots_h.append(float(highs[i]))
        if lows[i] == np.min(win_l):
            pivots_l.append(float(lows[i]))
    return pivots_h, pivots_l


def _hh_hl_counts(df: pd.DataFrame, lookback: int = 3, last_pivots: int = 6) -> tuple[int, int, int, int]:
    """Count HH/HL/LH/LL among the last `last_pivots` swing pivots."""
    if len(df) < lookback * 4:
        return 0, 0, 0, 0
    try:
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        ph, pl = _swing_pivots(h, l, lookback=lookback)
        ph = ph[-last_pivots:]
        pl = pl[-last_pivots:]
        hh = sum(1 for i in range(1, len(ph)) if ph[i] > ph[i - 1])
        lh = sum(1 for i in range(1, len(ph)) if ph[i] < ph[i - 1])
        hl = sum(1 for i in range(1, len(pl)) if pl[i] > pl[i - 1])
        ll = sum(1 for i in range(1, len(pl)) if pl[i] < pl[i - 1])
        return hh, hl, lh, ll
    except Exception:
        return 0, 0, 0, 0


def _range_compression(df: pd.DataFrame, n: int = 20) -> float:
    """
    1.0 = recent range unusually tight (consolidation); 0.0 = unusually wide.
    Suppressed when a large gap (>3%) was present — post-breakout consolidation is not range-bound.
    """
    if len(df) < n * 2:
        return 0.5
    try:
        c = df["close"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)

        if "open" in df.columns:
            opens = df["open"].to_numpy(float)
            gap_pcts = np.abs(opens[1:] - c[:-1]) / np.clip(c[:-1], 1e-9, None)
            max_gap = float(np.max(gap_pcts[-n:])) if len(gap_pcts) >= n else 0.0
        else:
            rets = np.abs(np.diff(c[-n - 1 :]) / np.clip(c[-n - 2 : -1], 1e-9, None))
            max_gap = float(np.max(rets)) if len(rets) > 0 else 0.0

        recent_spread = float(np.max(h[-n:]) - np.min(l[-n:]))
        long_spread = float(np.max(h[-2 * n :]) - np.min(l[-2 * n :]))
        if long_spread <= 0 or not np.isfinite(long_spread):
            return 0.5
        ratio = recent_spread / long_spread
        rc = float(np.clip(1.0 - (ratio - 0.3) / 0.7, 0.0, 1.0))

        # Tight range after a gap = post-breakout consolidation, not true range-bound
        if max_gap > 0.03:
            rc = rc * max(0.0, 1.0 - (max_gap - 0.03) / 0.10)

        return float(np.clip(rc, 0.0, 1.0))
    except Exception:
        return 0.5


def _label_regime(trend: float, range_score: float, brk_up: bool, brk_dn: bool, adx: float) -> str:
    """Map scores to a regime label. Priority: breakout > strong ADX > range > choppy."""
    if brk_up and trend > -0.05:
        return "breakout_up"
    if brk_dn and trend < 0.05:
        return "breakout_down"

    # Strong ADX overrides range_bound — a trending stock is never range-bound
    if adx > 30:
        if trend > 0.10:
            return "strong_uptrend" if trend > 0.35 else "weak_uptrend"
        if trend < -0.10:
            return "strong_downtrend" if trend < -0.35 else "weak_downtrend"

    # Range-bound only when ADX is genuinely weak
    if range_score > 0.6 and abs(trend) < 0.25 and adx < 25:
        return "range_bound"

    if adx > 20:
        if trend > 0.10:
            return "weak_uptrend"
        if trend < -0.10:
            return "weak_downtrend"

    if abs(trend) < 0.15 and range_score < 0.4:
        return "choppy"
    if trend > 0.45:
        return "strong_uptrend"
    if trend < -0.45:
        return "strong_downtrend"
    if trend > 0.15:
        return "weak_uptrend"
    if trend < -0.15:
        return "weak_downtrend"
    return "choppy"


def _verdict(
    regime: str,
    hurst: float,
    r2_mid: float,
    potential_up: float,
    potential_down: float,
    potential_flat: float,
    range_compression: float,
) -> str:
    desc = {
        "strong_uptrend": "Strong uptrend",
        "weak_uptrend": "Mild uptrend",
        "strong_downtrend": "Strong downtrend",
        "weak_downtrend": "Mild downtrend",
        "breakout_up": "Breaking out to the upside",
        "breakout_down": "Breaking down",
        "range_bound": "Consolidating in a range",
        "choppy": "No clear direction",
    }[regime]
    persist = (
        f"high persistence (Hurst {hurst:.2f})"
        if hurst > 0.6
        else f"weak persistence (Hurst {hurst:.2f})"
        if hurst > 0.55
        else f"mean-reverting tendency (Hurst {hurst:.2f})"
        if hurst < 0.45
        else f"near random-walk persistence (Hurst {hurst:.2f})"
    )
    cleanness = (
        f"clean trend (R² {r2_mid:.2f})"
        if r2_mid > 0.7
        else f"ragged trend (R² {r2_mid:.2f})"
        if r2_mid > 0.4
        else f"no clear linear path (R² {r2_mid:.2f})"
    )
    if regime in ("range_bound", "choppy"):
        compression = (
            "tight range"
            if range_compression > 0.6
            else "moderate range"
            if range_compression > 0.3
            else "wide range"
        )
        leans = (
            "upside lean"
            if potential_up - potential_down > 10
            else "downside lean"
            if potential_down - potential_up > 10
            else "no clear bias"
        )
        return f"{desc}: {compression}, {leans} ({persist})."
    return f"{desc}: {cleanness}, {persist}."


def detect_regime(df: pd.DataFrame, adx: float = 0.0, obv_slope: float = 0.0) -> Regime:
    """Compute market regime. adx/obv_slope come from compute_indicators (avoids re-computing)."""
    closes = df["close"].to_numpy(float)
    closes = closes[np.isfinite(closes)]
    if len(closes) < 8:
        return Regime(
            regime="choppy",
            verdict="Not enough data.",
            potential_up=33.3,
            potential_down=33.3,
            potential_flat=33.3,
            trend_score=0.0,
            range_score=0.5,
            breakout_up=False,
            breakout_down=False,
            hurst=0.5,
            r2_short=0.0,
            r2_mid=0.0,
            r2_long=0.0,
            donchian_pos=0.0,
            donchian_high=float(closes[-1] if len(closes) else 0),
            donchian_low=float(closes[-1] if len(closes) else 0),
            hh_count=0,
            hl_count=0,
            lh_count=0,
            ll_count=0,
            range_compression=0.5,
            components={},
        )

    don_n = cfg.regime_donchian_n
    r2_short, slope_short = _r2_and_slope(closes, max(10, don_n // 2))
    r2_mid, slope_mid = _r2_and_slope(closes, don_n)
    r2_long, slope_long = _r2_and_slope(closes, min(don_n * 2 + 10, len(closes)))

    log_close = np.log(np.clip(closes, 1e-9, None))
    hurst = _hurst(log_close, max_lag=min(cfg.regime_hurst_lags, len(closes) // 4))

    don_pos, don_hi, don_lo, brk_up, brk_dn = _donchian(df, n=don_n)
    hh, hl, lh, ll = _hh_hl_counts(df, lookback=cfg.regime_pivot_wing, last_pivots=6)
    rc = _range_compression(df, n=don_n)

    # Component scores: each in [−1, 1] (signed = direction); range_score in [0, 1]
    comp: dict[str, float] = {}

    # R² × sign(slope) — trend cleanness weighted by direction
    sgn_short = np.sign(slope_short)
    sgn_mid = np.sign(slope_mid)
    sgn_long = np.sign(slope_long)
    comp["regression_short"] = float(sgn_short * r2_short)
    comp["regression_mid"] = float(sgn_mid * r2_mid)
    comp["regression_long"] = float(sgn_long * r2_long)

    # Hurst signed by overall slope direction
    sgn_overall = np.sign(slope_long if abs(slope_long) > 1e-9 else slope_mid)
    hurst_signed = (hurst - 0.5) * 2.0 * float(sgn_overall)
    comp["hurst"] = float(np.clip(hurst_signed, -1.0, 1.0))

    comp["donchian"] = float(np.clip(don_pos, -1.0, 1.0))

    # HH/HL → up bias; LH/LL → down bias
    n_pivots_h = max(1, hh + lh)
    n_pivots_l = max(1, hl + ll)
    pattern_up = (hh / n_pivots_h) * 0.5 + (hl / n_pivots_l) * 0.5
    pattern_down = (lh / n_pivots_h) * 0.5 + (ll / n_pivots_l) * 0.5
    comp["pattern"] = float(np.clip(pattern_up - pattern_down, -1.0, 1.0))

    # ADX signed by slope direction (strength × direction)
    adx_signed = float(np.clip((adx - 20) / 30, 0.0, 1.0)) * float(sgn_overall)
    comp["adx_signed"] = float(np.clip(adx_signed, -1.0, 1.0))

    comp["obv"] = float(np.clip(obv_slope / 1.0, -1.0, 1.0))

    weights = {
        "regression_short": 0.10,
        "regression_mid": 0.18,
        "regression_long": 0.20,
        "hurst": 0.18,
        "donchian": 0.12,
        "pattern": 0.12,
        "adx_signed": 0.06,
        "obv": 0.04,
    }
    trend_score = float(np.clip(sum(comp[k] * w for k, w in weights.items()), -1.0, 1.0))

    # Range score: high when trend weak, Hurst near 0.5, range compressed, ADX low, Donchian mid.
    # ADX > 25 strongly suppresses range_score — a trending stock is not range-bound.
    adx_range_penalty = min(1.0, adx / 30.0)
    range_pieces = [
        1.0 - min(1.0, abs(trend_score) * 2.0),
        1.0 - min(1.0, abs(hurst - 0.5) * 2.0),
        rc * (1.0 - adx_range_penalty * 0.6),
        1.0 - min(1.0, adx / 40.0),
        1.0 - min(1.0, abs(don_pos)),
    ]
    range_score = float(np.clip(np.mean(range_pieces), 0.0, 1.0))

    # Potential gauges (0–100), normalised to sum to 100
    base_up = max(0.0, trend_score) * 100
    base_down = max(0.0, -trend_score) * 100
    base_flat = range_score * 100

    if brk_up:
        base_up += 25.0
    if brk_dn:
        base_down += 25.0

    if hurst > 0.55:
        amp = (hurst - 0.55) * 100
        if trend_score > 0:
            base_up += amp
        if trend_score < 0:
            base_down += amp

    total = base_up + base_down + base_flat
    if total <= 0:
        potential_up = potential_down = potential_flat = 33.3
    else:
        potential_up = round(base_up / total * 100, 1)
        potential_down = round(base_down / total * 100, 1)
        potential_flat = round(100 - potential_up - potential_down, 1)

    regime = _label_regime(trend_score, range_score, brk_up, brk_dn, adx)
    verdict = _verdict(regime, hurst, r2_mid, potential_up, potential_down, potential_flat, rc)

    return Regime(
        regime=regime,
        verdict=verdict,
        potential_up=potential_up,
        potential_down=potential_down,
        potential_flat=potential_flat,
        trend_score=round(trend_score, 3),
        range_score=round(range_score, 3),
        breakout_up=bool(brk_up),
        breakout_down=bool(brk_dn),
        hurst=round(hurst, 3),
        r2_short=round(r2_short, 3),
        r2_mid=round(r2_mid, 3),
        r2_long=round(r2_long, 3),
        donchian_pos=round(float(don_pos), 3),
        donchian_high=round(float(don_hi), 4),
        donchian_low=round(float(don_lo), 4),
        hh_count=int(hh),
        hl_count=int(hl),
        lh_count=int(lh),
        ll_count=int(ll),
        range_compression=round(float(rc), 3),
        components={k: round(float(v), 3) for k, v in comp.items()},
    )


def regime_to_dict(reg: Regime) -> dict:
    return asdict(reg)
