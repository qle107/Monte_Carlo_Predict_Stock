"""
core/regime.py — Market regime detector.

Answers the question the user actually cares about:
   "Is this stock trending up, trending down, or consolidating?"
   "How confident should I be in that read, and what's the potential
    for a move in each direction over the next N candles?"

Outputs a single Regime dataclass with:
  • regime          one of: strong_uptrend, weak_uptrend,
                            strong_downtrend, weak_downtrend,
                            breakout_up, breakout_down,
                            range_bound, choppy
  • verdict         a plain-English summary
  • potential_up    0..100 score for upside potential
  • potential_down  0..100 score for downside potential
  • potential_flat  0..100 score for consolidation
  • all the underlying numbers so the dashboard can show them

Components combined here
────────────────────────
  Hurst exponent (R/S analysis)         persistence vs mean-reversion
  Multi-window linear regression R²     "how clean is the trend?"
  Donchian channel position             breakout detection
  Higher-highs / lower-lows pattern     classical trend structure
  ADX                                   trend *strength* (signed by slope)
  Range compression                     ATR vs N-bar high-low spread
  Volume confirmation                   OBV slope sign

Each component contributes to two normalised 0..1 sub-scores: trend_score
(positive = up, negative = down) and range_score (high = consolidation).
The regime label is derived from those + breakout flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np
import pandas as pd


# ─── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class Regime:
    regime:          str
    verdict:         str

    # 0..100 potential scores — these are what the user reads
    potential_up:    float
    potential_down:  float
    potential_flat:  float

    # Underlying signals (also nice on the dashboard)
    trend_score:     float        # -1 (strong down) … +1 (strong up)
    range_score:     float        #  0 (trending)    …  1 (range-bound)
    breakout_up:     bool
    breakout_down:   bool

    hurst:           float        # 0.5 random walk, >0.6 trending, <0.4 mean-rev
    r2_short:        float        # 10-bar regression R²
    r2_mid:          float        # 20-bar regression R²
    r2_long:         float        # 50-bar regression R²

    donchian_pos:    float        # -1 at lower band, +1 at upper, 0 mid
    donchian_high:   float
    donchian_low:    float

    hh_count:        int          # higher-highs in last N pivots
    hl_count:        int
    lh_count:        int
    ll_count:        int

    range_compression: float      # 0 = wide range, 1 = very tight

    components:      Dict[str, float] = field(default_factory=dict)


# ─── Components ─────────────────────────────────────────────────────────────

def _hurst(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Rescaled-range Hurst exponent.
       H ≈ 0.5  random walk
       H >  0.6 persistent (trend continues)
       H <  0.4 mean-reverting
    """
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
        log_tau  = np.log(tau)
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
       position = -1 at lower band, +1 at upper, 0 at midline.
       breakout_up   = current close > previous-N-bars max  (any of n=20 OR n=10)
       breakout_down = current close < previous-N-bars min  (any of n=20 OR n=10)

    Also fires breakout_up when price is within 1.5% of the N-bar high and
    ADX is strong — catches "just broke out and now consolidating at highs" scenarios.
    """
    if len(df) < n + 2:
        c = float(df["close"].iloc[-1]) if len(df) else 0.0
        return 0.0, c, c, False, False
    try:
        prev = df.iloc[-(n + 1):-1]            # exclude current bar
        hi = float(prev["high"].max())
        lo = float(prev["low"].min())
        cur_close = float(df["close"].iloc[-1])
        cur_high  = float(df["high"].iloc[-1])
        cur_low   = float(df["low"].iloc[-1])

        rng = max(hi - lo, 1e-9)
        pos = (cur_close - (hi + lo) / 2.0) / (rng / 2.0)
        pos = float(np.clip(pos, -1.5, 1.5))

        brk_up   = cur_high > hi   # primary: new N-bar high
        brk_down = cur_low  < lo

        # Secondary check on shorter window (10 bars) — catches recent breakouts
        if not brk_up and len(df) >= 12:
            prev10 = df.iloc[-11:-1]
            hi10   = float(prev10["high"].max())
            lo10   = float(prev10["low"].min())
            if cur_high > hi10:
                brk_up = True
            if cur_low < lo10:
                brk_down = True

        # Tertiary: price held near N-bar high (post-breakout consolidation)
        # If close is within 3% of the N-bar high, treat as near-breakout territory
        if not brk_up and hi > 0:
            pct_from_hi = (hi - cur_close) / hi
            if pct_from_hi < 0.03 and cur_close > (hi + lo) / 2.0:
                brk_up = True

        return pos, hi, lo, brk_up, brk_down
    except Exception:
        c = float(df["close"].iloc[-1]) if len(df) else 0.0
        return 0.0, c, c, False, False


def _swing_pivots(highs: np.ndarray, lows: np.ndarray, lookback: int = 3
                  ) -> tuple[List[float], List[float]]:
    """
    Crude swing-pivot detector:
       a high is a pivot if it's the max of [i-lookback, i+lookback]
       a low  is a pivot if it's the min of that window.
    Returns (pivot_highs, pivot_lows) in chronological order.
    """
    n = len(highs)
    pivots_h, pivots_l = [], []
    for i in range(lookback, n - lookback):
        win_h = highs[i - lookback:i + lookback + 1]
        win_l = lows [i - lookback:i + lookback + 1]
        if highs[i] == np.max(win_h):
            pivots_h.append(float(highs[i]))
        if lows[i]  == np.min(win_l):
            pivots_l.append(float(lows[i]))
    return pivots_h, pivots_l


def _hh_hl_counts(df: pd.DataFrame, lookback: int = 3, last_pivots: int = 6
                  ) -> tuple[int, int, int, int]:
    """
    Count higher-highs / higher-lows / lower-highs / lower-lows
    among the last `last_pivots` swing pivots.
    """
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
    1.0 when the recent range is unusually tight relative to the long-term
    daily range; 0.0 when range is unusually wide. Used to flag consolidation.

    Gap-up correction: if price gapped up significantly (>3%) at any point
    in the recent window, the "tight recent range" is actually post-breakout
    consolidation — not a range-bound setup. Return a reduced compression value.
    """
    if len(df) < n * 2:
        return 0.5
    try:
        c = df["close"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)

        # Detect large gaps in recent N bars (open vs prev close)
        if "open" in df.columns:
            opens  = df["open"].to_numpy(float)
            closes = c
            # Bar-to-bar gap: open of bar vs close of previous bar
            gap_pcts = np.abs(opens[1:] - closes[:-1]) / np.clip(closes[:-1], 1e-9, None)
            max_gap = float(np.max(gap_pcts[-n:])) if len(gap_pcts) >= n else 0.0
        else:
            # Estimate from close-to-close jumps as proxy
            rets = np.abs(np.diff(c[-n - 1:]) / np.clip(c[-n - 2:-1], 1e-9, None))
            max_gap = float(np.max(rets)) if len(rets) > 0 else 0.0

        # Recent N-bar high/low spread vs long N*2-bar
        recent_spread = float(np.max(h[-n:]) - np.min(l[-n:]))
        long_spread   = float(np.max(h[-2 * n:]) - np.min(l[-2 * n:]))
        if long_spread <= 0 or not np.isfinite(long_spread):
            return 0.5
        ratio = recent_spread / long_spread
        rc = float(np.clip(1.0 - (ratio - 0.3) / 0.7, 0.0, 1.0))

        # If there was a significant gap (>3%), suppress compression reading
        # — this is post-breakout consolidation, not a true range
        if max_gap > 0.03:
            rc = rc * max(0.0, 1.0 - (max_gap - 0.03) / 0.10)

        return float(np.clip(rc, 0.0, 1.0))
    except Exception:
        return 0.5


# ─── Aggregator ─────────────────────────────────────────────────────────────

def _label_regime(trend: float, range_score: float,
                  brk_up: bool, brk_dn: bool, adx: float) -> str:
    """Map underlying scores into a single regime label.

    Priority order (highest to lowest):
      1. Donchian breakout with directional trend
      2. Strong trend confirmed by ADX — overrides range_bound
      3. Range-bound (only when ADX is weak — avoids mislabelling
         post-breakout consolidations as range-bound)
      4. Choppy / weak trend fallthrough
    """
    # 1. Breakout signals (Donchian fired + mild directional lean)
    if brk_up and trend > -0.05:   return "breakout_up"    # allow near-zero trend
    if brk_dn and trend <  0.05:   return "breakout_down"

    # 2. Strong trend confirmed by ADX — never call this range_bound
    if adx > 30:
        if trend >  0.10: return "strong_uptrend" if trend > 0.35 else "weak_uptrend"
        if trend < -0.10: return "strong_downtrend" if trend < -0.35 else "weak_downtrend"

    # 3. Range-bound — only when ADX is genuinely weak (< 25) AND trend is flat
    if range_score > 0.6 and abs(trend) < 0.25 and adx < 25:
        return "range_bound"

    # 4. Moderate ADX (20-30) with directional trend
    if adx > 20:
        if trend >  0.10: return "weak_uptrend"
        if trend < -0.10: return "weak_downtrend"

    # 5. Weak / choppy
    if abs(trend) < 0.15 and range_score < 0.4:
        return "choppy"
    if trend >  0.45: return "strong_uptrend"
    if trend < -0.45: return "strong_downtrend"
    if trend >  0.15: return "weak_uptrend"
    if trend < -0.15: return "weak_downtrend"
    return "choppy"


def _verdict(regime: str, hurst: float, r2_mid: float,
             potential_up: float, potential_down: float, potential_flat: float,
             range_compression: float) -> str:
    desc = {
        "strong_uptrend":   "Strong uptrend",
        "weak_uptrend":     "Mild uptrend",
        "strong_downtrend": "Strong downtrend",
        "weak_downtrend":   "Mild downtrend",
        "breakout_up":      "Breaking out to the upside",
        "breakout_down":    "Breaking down",
        "range_bound":      "Consolidating in a range",
        "choppy":           "No clear direction",
    }[regime]
    persist = (
        "high persistence (Hurst {:.2f})".format(hurst) if hurst > 0.6 else
        "weak persistence (Hurst {:.2f})".format(hurst) if hurst > 0.55 else
        "mean-reverting tendency (Hurst {:.2f})".format(hurst) if hurst < 0.45 else
        "near random-walk persistence (Hurst {:.2f})".format(hurst)
    )
    cleanness = (
        "clean trend (R² {:.2f})".format(r2_mid) if r2_mid > 0.7 else
        "ragged trend (R² {:.2f})".format(r2_mid) if r2_mid > 0.4 else
        "no clear linear path (R² {:.2f})".format(r2_mid)
    )
    if regime in ("range_bound", "choppy"):
        compression = (
            "tight range" if range_compression > 0.6 else
            "moderate range" if range_compression > 0.3 else
            "wide range"
        )
        leans = (
            ("upside lean" if potential_up - potential_down > 10 else
             "downside lean" if potential_down - potential_up > 10 else
             "no clear bias")
        )
        return f"{desc}: {compression}, {leans} ({persist})."
    return f"{desc}: {cleanness}, {persist}."


def detect_regime(df: pd.DataFrame, adx: float = 0.0,
                  obv_slope: float = 0.0) -> Regime:
    """
    Compute the regime read.
    `adx` and `obv_slope` come from core.indicators (avoids re-computing).
    """
    closes = df["close"].to_numpy(float)
    closes = closes[np.isfinite(closes)]
    if len(closes) < 8:
        # Not enough data to commit to anything
        return Regime(
            regime="choppy", verdict="Not enough data.",
            potential_up=33.3, potential_down=33.3, potential_flat=33.3,
            trend_score=0.0, range_score=0.5,
            breakout_up=False, breakout_down=False,
            hurst=0.5, r2_short=0.0, r2_mid=0.0, r2_long=0.0,
            donchian_pos=0.0, donchian_high=float(closes[-1] if len(closes) else 0),
            donchian_low=float(closes[-1] if len(closes) else 0),
            hh_count=0, hl_count=0, lh_count=0, ll_count=0,
            range_compression=0.5,
            components={},
        )

    # Multi-window regression
    r2_short, slope_short = _r2_and_slope(closes, 10)
    r2_mid,   slope_mid   = _r2_and_slope(closes, 20)
    r2_long,  slope_long  = _r2_and_slope(closes, min(50, len(closes)))

    # Hurst on log-prices
    log_close = np.log(np.clip(closes, 1e-9, None))
    hurst = _hurst(log_close, max_lag=min(20, len(closes) // 4))

    # Donchian
    don_pos, don_hi, don_lo, brk_up, brk_dn = _donchian(df, n=20)

    # HH/HL counts
    hh, hl, lh, ll = _hh_hl_counts(df, lookback=3, last_pivots=6)

    # Range compression
    rc = _range_compression(df, n=20)

    # ── Component scores ────────────────────────────────────────────────
    # Each is in [-1, 1] (signed = direction); range_score is [0, 1].
    comp: Dict[str, float] = {}

    # 1. Cleanness × direction:  R² weighted sign of slope
    sgn_short = np.sign(slope_short); sgn_mid = np.sign(slope_mid); sgn_long = np.sign(slope_long)
    comp["regression_short"] = float(sgn_short * r2_short)
    comp["regression_mid"]   = float(sgn_mid   * r2_mid)
    comp["regression_long"]  = float(sgn_long  * r2_long)

    # 2. Hurst — only meaningful when the trend has direction.
    sgn_overall = np.sign(slope_long if abs(slope_long) > 1e-9 else slope_mid)
    hurst_signed = (hurst - 0.5) * 2.0 * float(sgn_overall)  # -1..+1
    comp["hurst"] = float(np.clip(hurst_signed, -1.0, 1.0))

    # 3. Donchian: trend-confirmer (price near upper band → up bias)
    comp["donchian"] = float(np.clip(don_pos, -1.0, 1.0))

    # 4. Pattern: HH/HL → up; LH/LL → down
    n_pivots_h = max(1, hh + lh)
    n_pivots_l = max(1, hl + ll)
    pattern_up   = (hh / n_pivots_h) * 0.5 + (hl / n_pivots_l) * 0.5
    pattern_down = (lh / n_pivots_h) * 0.5 + (ll / n_pivots_l) * 0.5
    comp["pattern"] = float(np.clip(pattern_up - pattern_down, -1.0, 1.0))

    # 5. ADX signed by slope (strength × direction)
    adx_signed = float(np.clip((adx - 20) / 30, 0.0, 1.0)) * float(sgn_overall)
    comp["adx_signed"] = float(np.clip(adx_signed, -1.0, 1.0))

    # 6. Volume confirmation (OBV slope normalised → ±1)
    comp["obv"] = float(np.clip(obv_slope / 1.0, -1.0, 1.0))

    # ── Aggregate trend_score (signed) ──────────────────────────────────
    weights = {
        "regression_short": 0.10,
        "regression_mid":   0.18,
        "regression_long":  0.20,
        "hurst":            0.18,
        "donchian":         0.12,
        "pattern":          0.12,
        "adx_signed":       0.06,
        "obv":              0.04,
    }
    trend_score = float(np.clip(sum(comp[k] * w for k, w in weights.items()), -1.0, 1.0))

    # ── Range score (0..1) ──────────────────────────────────────────────
    # High when: low |trend|, low Hurst, high range_compression, low ADX,
    # and price near Donchian midline.
    # ADX > 25 strongly suppresses range_score — a trending stock is not range-bound.
    adx_range_penalty = min(1.0, adx / 30.0)   # 0 at ADX=0, 1.0 at ADX=30+
    range_pieces = [
        1.0 - min(1.0, abs(trend_score) * 2.0),
        1.0 - min(1.0, abs(hurst - 0.5) * 2.0),
        rc * (1.0 - adx_range_penalty * 0.6),   # ADX dampens compression weight
        1.0 - min(1.0, adx / 40.0),
        1.0 - min(1.0, abs(don_pos)),
    ]
    range_score = float(np.clip(np.mean(range_pieces), 0.0, 1.0))

    # ── Potential gauges (0..100) ───────────────────────────────────────
    # Trend bias contributes to up/down; range_score contributes to flat.
    base_up   = max(0.0,  trend_score) * 100
    base_down = max(0.0, -trend_score) * 100
    base_flat = range_score * 100

    # Breakout bumps direction sharply
    if brk_up:   base_up   += 25.0
    if brk_dn:   base_down += 25.0

    # Persistence amplifies the dominant side
    if hurst > 0.55:
        amp = (hurst - 0.55) * 100  # max +9
        if trend_score > 0: base_up   += amp
        if trend_score < 0: base_down += amp

    # Re-normalise so they sum to 100 (interpretable as probabilities)
    total = base_up + base_down + base_flat
    if total <= 0:
        potential_up = potential_down = potential_flat = 33.3
    else:
        potential_up   = round(base_up   / total * 100, 1)
        potential_down = round(base_down / total * 100, 1)
        potential_flat = round(100 - potential_up - potential_down, 1)

    regime = _label_regime(trend_score, range_score, brk_up, brk_dn, adx)
    verdict = _verdict(regime, hurst, r2_mid,
                       potential_up, potential_down, potential_flat, rc)

    return Regime(
        regime           = regime,
        verdict          = verdict,
        potential_up     = potential_up,
        potential_down   = potential_down,
        potential_flat   = potential_flat,
        trend_score      = round(trend_score, 3),
        range_score      = round(range_score, 3),
        breakout_up      = bool(brk_up),
        breakout_down    = bool(brk_dn),
        hurst            = round(hurst, 3),
        r2_short         = round(r2_short, 3),
        r2_mid           = round(r2_mid,   3),
        r2_long          = round(r2_long,  3),
        donchian_pos     = round(float(don_pos), 3),
        donchian_high    = round(float(don_hi), 4),
        donchian_low     = round(float(don_lo), 4),
        hh_count         = int(hh),
        hl_count         = int(hl),
        lh_count         = int(lh),
        ll_count         = int(ll),
        range_compression= round(float(rc), 3),
        components       = {k: round(float(v), 3) for k, v in comp.items()},
    )


def regime_to_dict(reg: Regime) -> dict:
    return asdict(reg)
