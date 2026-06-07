"""Expected-move bands from realized volatility (Yang-Zhang + EWMA blend). See docs/math.md."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np

try:
    from scipy import stats as _sps

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

# Horizon label -> trading days
DEFAULT_HORIZONS: dict[str, int] = {"1d": 1, "1w": 5}

# Band coverage (central probability mass) and per-side tails, in percent
_INSIDE_1S = 68.3  # quantiles 0.15866 / 0.84134
_INSIDE_2S = 95.4  # quantiles 0.02275 / 0.97725
_TAIL_1S = 15.9
_TAIL_2S = 2.3
_Q_1S = 0.841345
_Q_2S = 0.977250

_MIN_RETURNS = 15  # minimum daily returns for a meaningful vol estimate
_EWMA_LAMBDA = 0.94  # RiskMetrics decay for daily data
_W_YZ = 0.60  # blend weight on Yang-Zhang variance
_DF_MIN, _DF_MAX = 4.5, 30.0  # Student-t df clamp (30 ~ normal)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fit_t_df(log_rets: np.ndarray) -> float:
    """Method-of-moments Student-t df from excess kurtosis (df = 4 + 6/k)."""
    n = log_rets.size
    if n < 20:
        return _DF_MAX
    m = float(np.mean(log_rets))
    s2 = float(np.var(log_rets))
    if s2 <= 0:
        return _DF_MAX
    kurt_ex = float(np.mean((log_rets - m) ** 4) / s2**2) - 3.0
    # Small-sample kurtosis is noisy; shrink mildly toward normal
    kurt_ex = max(0.0, kurt_ex * (n / (n + 20.0)))
    if kurt_ex < 0.05:
        return _DF_MAX
    return float(np.clip(4.0 + 6.0 / kurt_ex, _DF_MIN, _DF_MAX))


def _t_multiplier(q: float, df: float) -> float:
    """Quantile of a unit-variance Student-t (normal fallback w/o scipy)."""
    if not _HAS_SCIPY or df >= _DF_MAX - 1e-9:
        # Normal quantiles for the two coverages we use
        return 1.0 if abs(q - _Q_1S) < 1e-6 else 2.0 if abs(q - _Q_2S) < 1e-6 else 1.0
    raw = float(_sps.t.ppf(q, df))
    return raw / math.sqrt(df / (df - 2.0))


def _t_sf(z: float, df: float) -> float:
    """P(X > z) for unit-variance Student-t z-score (normal fallback)."""
    if not _HAS_SCIPY or df >= _DF_MAX - 1e-9:
        return 1.0 - _norm_cdf(z)
    return float(_sps.t.sf(z * math.sqrt(df / (df - 2.0)), df))


def _ewma_var(log_rets: np.ndarray, lam: float = _EWMA_LAMBDA) -> float:
    """RiskMetrics EWMA daily variance (zero-mean convention)."""
    r2 = log_rets**2
    n = r2.size
    w = lam ** np.arange(n - 1, -1, -1)  # newest gets weight 1
    w /= w.sum()
    return float(np.dot(w, r2))


def _yang_zhang_var(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> float | None:
    """
    Yang-Zhang (2000) daily variance from aligned OHLC arrays.

    sigma2_yz = sigma2_overnight + k*sigma2_open_to_close + (1-k)*sigma2_rs
    with k = 0.34 / (1.34 + (n+1)/(n-1)).  Requires >= _MIN_RETURNS days.
    Returns None when inputs are unusable (missing/zero prices, mismatched
    lengths, degenerate H/L).
    """
    n_bars = min(opens.size, highs.size, lows.size, closes.size)
    if n_bars < _MIN_RETURNS + 1:
        return None
    o, h, lo, c = (a[-n_bars:] for a in (opens, highs, lows, closes))
    ok = (o > 0) & (h > 0) & (lo > 0) & (c > 0) & (h >= lo)
    if not bool(np.all(ok)):
        return None

    # Per-day terms (day i uses previous close); n = number of usable days
    on = np.log(o[1:] / c[:-1])  # overnight
    oc = np.log(c[1:] / o[1:])  # open-to-close
    u = np.log(h[1:] / o[1:])
    d = np.log(lo[1:] / o[1:])
    n = on.size
    if n < _MIN_RETURNS:
        return None

    var_on = float(np.var(on, ddof=1))
    var_oc = float(np.var(oc, ddof=1))
    rs = float(np.mean(u * (u - oc) + d * (d - oc)))  # Rogers-Satchell
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = var_on + k * var_oc + (1.0 - k) * max(rs, 0.0)
    return yz if np.isfinite(yz) and yz > 0 else None


def prob_above(spot: float, target: float, sd_pct: float, df: float | None = None) -> float:
    """
    P(close > target) in percent for a horizon with SD `sd_pct` (fraction).

    Lognormal/Student-t model with martingale anchor:
        z = (ln(target/spot) + sd^2/2) / sd ,  P = SF_t(z; df)
    `df=None` -> normal tails (backwards compatible).
    """
    if spot <= 0 or target <= 0 or sd_pct <= 0:
        return 0.0
    z = (math.log(target / spot) + 0.5 * sd_pct**2) / sd_pct
    p = _t_sf(z, df if df is not None else _DF_MAX)
    return round(p * 100.0, 1)


def prob_below(spot: float, target: float, sd_pct: float, df: float | None = None) -> float:
    """P(close < target) in percent. Complement of prob_above."""
    if spot <= 0 or target <= 0 or sd_pct <= 0:
        return 0.0
    return round(100.0 - prob_above(spot, target, sd_pct, df), 1)


def compute_expected_move(
    daily_closes: Sequence[float],
    spot: float | None = None,
    horizons: Mapping[str, int] | None = None,
    window: int = 60,
    daily_opens: Sequence[float] | None = None,
    daily_highs: Sequence[float] | None = None,
    daily_lows: Sequence[float] | None = None,
) -> dict | None:
    """
    Compute 68% / 95% expected-move price bands per horizon.

    Parameters
    ----------
    daily_closes : trailing DAILY close prices (oldest -> newest).
    spot         : anchor price for the bands; defaults to last close.
    horizons     : {label: trading_days}; defaults to {"1d": 1, "1w": 5}.
    window       : max number of trailing daily returns to use.
    daily_opens/highs/lows : optional OHLC aligned with `daily_closes`;
                   when provided, the Yang-Zhang estimator is blended in.

    Returns a JSON-safe dict, or None when there is not enough history.
    """
    closes = np.asarray(daily_closes, dtype=float)
    valid = np.isfinite(closes) & (closes > 0)
    closes_f = closes[valid]
    if closes_f.size < _MIN_RETURNS + 1:
        return None

    tail = min(window + 1, closes_f.size)
    cw = closes_f[-tail:]
    log_rets = np.diff(np.log(cw))
    if log_rets.size < _MIN_RETURNS:
        return None

    # Close-to-close estimators
    var_cc = float(np.var(log_rets, ddof=1))
    var_ewma = _ewma_var(log_rets)

    # Yang-Zhang (only when full OHLC aligns with the close series)
    var_yz = None
    if daily_opens is not None and daily_highs is not None and daily_lows is not None:
        o = np.asarray(daily_opens, dtype=float)
        h = np.asarray(daily_highs, dtype=float)
        lo = np.asarray(daily_lows, dtype=float)
        if o.size == closes.size and h.size == closes.size and lo.size == closes.size:
            o, h, lo = o[valid], h[valid], lo[valid]
            var_yz = _yang_zhang_var(o[-tail:], h[-tail:], lo[-tail:], cw)

    if var_yz is not None and var_ewma > 0:
        var_d = _W_YZ * var_yz + (1.0 - _W_YZ) * var_ewma
        method = "yang_zhang_ewma_t"
    elif var_ewma > 0:
        var_d = var_ewma
        method = "ewma_t"
    elif var_cc > 0:
        var_d = var_cc
        method = "close_std_t"
    else:
        return None

    daily_vol = math.sqrt(var_d)
    annual_vol = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Fat-tail quantile multipliers (unit-variance Student-t)
    df_t = _fit_t_df(log_rets)
    m1 = _t_multiplier(_Q_1S, df_t)
    m2 = _t_multiplier(_Q_2S, df_t)

    anchor = float(spot) if spot is not None and spot > 0 else float(cw[-1])

    out_horizons: dict[str, dict] = {}
    for label, days in (horizons or DEFAULT_HORIZONS).items():
        sd = daily_vol * math.sqrt(days)
        drift = -0.5 * sd * sd  # martingale anchor: E[S_T] = spot
        band = lambda m: (anchor * math.exp(drift - m * sd), anchor * math.exp(drift + m * sd))
        s1_low, s1_high = band(m1)
        s2_low, s2_high = band(m2)
        out_horizons[label] = {
            "days": int(days),
            "sd_pct": round(sd * 100.0, 2),
            "sigma1": {
                "low": round(s1_low, 2),
                "high": round(s1_high, 2),
                "prob_inside": _INSIDE_1S,
                "prob_above_high": _TAIL_1S,
                "prob_below_low": _TAIL_1S,
            },
            "sigma2": {
                "low": round(s2_low, 2),
                "high": round(s2_high, 2),
                "prob_inside": _INSIDE_2S,
                "prob_above_high": _TAIL_2S,
                "prob_below_low": _TAIL_2S,
            },
        }

    return {
        "method": method,
        "spot": round(anchor, 4),
        "annual_vol_pct": round(annual_vol * 100.0, 1),
        "daily_vol_pct": round(daily_vol * 100.0, 2),
        "window_days": int(log_rets.size),
        "df_t": round(df_t, 1),
        "band_mult": {"m68": round(m1, 3), "m95": round(m2, 3)},
        "estimators_ann_pct": {
            "yang_zhang": round(math.sqrt(var_yz * TRADING_DAYS_PER_YEAR) * 100.0, 1)
            if var_yz is not None
            else None,
            "ewma": round(math.sqrt(var_ewma * TRADING_DAYS_PER_YEAR) * 100.0, 1)
            if var_ewma > 0
            else None,
            "close_std": round(math.sqrt(var_cc * TRADING_DAYS_PER_YEAR) * 100.0, 1)
            if var_cc > 0
            else None,
        },
        "horizons": out_horizons,
        "note": (
            "Realized-vol bands (Yang-Zhang + EWMA blend, Student-t tails, "
            "lognormal geometry). Implied vol, if available, is preferred "
            "for market-priced ranges."
        ),
    }


def expected_move_for_ticker(
    ticker: str,
    spot: float | None = None,
    window: int = 60,
    horizons: Mapping[str, int] | None = None,
) -> dict | None:
    """Fetch trailing daily candles and compute expected-move bands."""
    try:
        from .fetcher import fetch_candles

        df = fetch_candles(ticker, "1d", window + 10, False)
        return compute_expected_move(
            df["close"].to_numpy(dtype=float),
            spot=spot,
            horizons=horizons,
            window=window,
            daily_opens=df["open"].to_numpy(dtype=float),
            daily_highs=df["high"].to_numpy(dtype=float),
            daily_lows=df["low"].to_numpy(dtype=float),
        )
    except Exception as e:
        logger.debug("[expected_move] %s failed: %s", ticker, e)
        return None


if __name__ == "__main__":
    # Self-test 1: GBM with known vol - YZ+EWMA blend should recover it.
    rng = np.random.default_rng(7)
    TRUE_ANN = 0.80
    d = TRUE_ANN / math.sqrt(TRADING_DAYS_PER_YEAR)
    n = 61
    c = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rng.normal(0, d, n - 1)])))
    # Simple intraday structure: open = prev close, H/L bracket O and C
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, d / 2, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, d / 2, n)))
    em = compute_expected_move(c, daily_opens=o, daily_highs=h, daily_lows=lo)
    assert em is not None and em["method"] == "yang_zhang_ewma_t", em and em["method"]
    err = abs(em["annual_vol_pct"] - TRUE_ANN * 100) / (TRUE_ANN * 100)
    print(f"GBM recovery: true {TRUE_ANN*100:.0f}%  est {em['annual_vol_pct']}%  (err {err*100:.0f}%)")
    assert err < 0.35, "blend should land near true vol"

    # Self-test 2: band geometry & ordering
    w = em["horizons"]["1w"]
    s = em["spot"]
    assert w["sigma2"]["low"] < w["sigma1"]["low"] < s < w["sigma1"]["high"] < w["sigma2"]["high"]
    assert w["sigma2"]["low"] > 0, "lognormal bands can never go negative"

    # Self-test 3: close-only fallback still works (old API)
    em2 = compute_expected_move(c)
    assert em2 is not None and em2["method"] in ("ewma_t", "close_std_t")

    # Self-test 4: fat tails widen the 95% band, tighten the 68% band
    if _HAS_SCIPY:
        assert _t_multiplier(_Q_2S, 5.0) > 2.0 > _t_multiplier(_Q_2S, _DF_MAX)
        assert _t_multiplier(_Q_1S, 5.0) < 1.0
    print("m68/m95 at df=5:", round(_t_multiplier(_Q_1S, 5.0), 3), round(_t_multiplier(_Q_2S, 5.0), 3))

    # Self-test 5: probability sanity - P(> +68% band edge) ~ 15.9%
    sd_1w = w["sd_pct"] / 100.0
    p = prob_above(s, w["sigma1"]["high"], sd_1w, df=em["df_t"])
    print(f"P(> 68% band edge) = {p}%  (expect ~15.9)")
    assert abs(p - 15.9) < 1.5

    # Self-test 6: degenerate inputs
    assert compute_expected_move([100.0] * 5) is None
    assert compute_expected_move([100.0] * 50) is None
    print("All self-tests passed.")
