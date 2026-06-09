"""Drift forecast for the POP scanner (momentum, reversal, volume, sentiment)."""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
_MONTH = 21          # trading days in the reference (1-month) horizon
_YEAR = 252

# Monthly return contribution per unit of tanh-signal.
K_MOM = 0.020
K_REV = 0.020
K_VOL = 0.010
K_SENT = 0.015

_MAX_MONTHLY_FACTOR = 0.035
_MAX_MONTHLY_TOTAL = 0.08

_MIN_CLOSES = 40


def _annual_vol(log_rets: np.ndarray) -> float:
    """Annualized close-to-close volatility (fraction)."""
    if log_rets.size < 5:
        return 0.0
    return float(np.std(log_rets, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _tanh(x: float) -> float:
    return math.tanh(max(-6.0, min(6.0, x)))


def compute_forecast(
    closes,
    volumes=None,
    *,
    spot: float | None = None,
    sentiment_score: float | None = None,
) -> dict | None:
    """Blend momentum, reversal, volume, and sentiment into a drift forecast."""
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < _MIN_CLOSES:
        return None

    log_rets = np.diff(np.log(c))
    sigma_ann = _annual_vol(log_rets)
    if sigma_ann <= 0:
        return None
    anchor = float(spot) if (spot and spot > 0) else float(c[-1])

    factors: list[dict] = []

    # 12-1 momentum (skip the most recent month)
    if c.size > _MONTH + 5:
        lookback = min(_YEAR, c.size - 1)
        start = c[-(lookback + 1)] if c.size > lookback else c[0]
        end = c[-(_MONTH + 1)]
        mom_raw = end / start - 1.0
        period_yrs = max((lookback - _MONTH), 1) / TRADING_DAYS_PER_YEAR
        scale = sigma_ann * math.sqrt(max(period_yrs, 1e-6))
        mom_z = mom_raw / scale if scale > 0 else 0.0
        mom_c = max(-_MAX_MONTHLY_FACTOR, min(_MAX_MONTHLY_FACTOR, K_MOM * _tanh(mom_z)))
        factors.append({
            "name": "momentum", "label": "Intermediate momentum (12-1m)",
            "raw_pct": round(mom_raw * 100, 2), "signal": round(mom_z, 3),
            "contribution_pct": round(mom_c * 100, 3), "direction": "bullish" if mom_c > 0 else "bearish",
            "source": "Jegadeesh & Titman (1993)",
        })

    # 1-month reversal (contrarian)
    if c.size > _MONTH:
        rev_raw = c[-1] / c[-(_MONTH + 1)] - 1.0
        scale = sigma_ann * math.sqrt(_MONTH / TRADING_DAYS_PER_YEAR)
        rev_z = rev_raw / scale if scale > 0 else 0.0
        rev_c = -K_REV * _tanh(rev_z)
        rev_c = max(-_MAX_MONTHLY_FACTOR, min(_MAX_MONTHLY_FACTOR, rev_c))
        factors.append({
            "name": "reversal", "label": "Short-term reversal (1m)",
            "raw_pct": round(rev_raw * 100, 2), "signal": round(rev_z, 3),
            "contribution_pct": round(rev_c * 100, 3), "direction": "bullish" if rev_c > 0 else "bearish",
            "source": "Jegadeesh (1990) / Lehmann (1990)",
        })

    # High-volume premium
    if volumes is not None:
        v = np.asarray(volumes, dtype=float)
        v = v[np.isfinite(v) & (v >= 0)]
        if v.size >= 30:
            recent = float(np.mean(v[-5:]))
            base = float(np.median(v[-60:-5])) if v.size >= 65 else float(np.median(v[:-5]))
            if base > 0 and recent > 0:
                vol_raw = math.log(recent / base)
                vol_z = vol_raw / 0.5
                vol_c = max(0.0, min(_MAX_MONTHLY_FACTOR, K_VOL * _tanh(vol_z)))
                factors.append({
                    "name": "volume", "label": "High-volume premium",
                    "raw_pct": round((recent / base - 1.0) * 100, 1), "signal": round(vol_z, 3),
                    "contribution_pct": round(vol_c * 100, 3),
                    "direction": "bullish" if vol_c > 0 else "neutral",
                    "source": "Gervais, Kaniel & Mingelgrin (2001)",
                })

    # Sentiment tilt (optional)
    if sentiment_score is not None:
        s = max(-1.0, min(1.0, float(sentiment_score)))
        sent_c = max(-_MAX_MONTHLY_FACTOR, min(_MAX_MONTHLY_FACTOR, K_SENT * s))
        factors.append({
            "name": "sentiment", "label": "News / social sentiment",
            "raw_pct": round(s * 100, 1), "signal": round(s, 3),
            "contribution_pct": round(sent_c * 100, 3),
            "direction": "bullish" if sent_c > 0 else "bearish" if sent_c < 0 else "neutral",
            "source": "Tetlock (2007)",
        })

    if not factors:
        return None

    contribs = [f["contribution_pct"] / 100.0 for f in factors]
    monthly = sum(contribs)
    monthly = max(-_MAX_MONTHLY_TOTAL, min(_MAX_MONTHLY_TOTAL, monthly))

    mu_annual = math.log1p(monthly) * (TRADING_DAYS_PER_YEAR / _MONTH)

    denom = sum(abs(x) for x in contribs)
    alignment = abs(monthly) / denom if denom > 1e-12 else 0.0

    direction = ("bullish" if monthly > 0.003 else
                 "bearish" if monthly < -0.003 else "neutral")

    return {
        "spot": round(anchor, 4),
        "annual_vol_pct": round(sigma_ann * 100, 1),
        "factors": factors,
        "expected_monthly_return_pct": round(monthly * 100, 2),
        "mu_annual_pct": round(mu_annual * 100, 2),
        "mu_annual": mu_annual,
        "target_1m": round(anchor * math.exp(mu_annual * (_MONTH / TRADING_DAYS_PER_YEAR)), 2),
        "direction": direction,
        "confidence": round(alignment, 2),
        "method": "research_composite_v1",
        "citations": [
            "Jegadeesh & Titman (1993), J. Finance 48(1):65-91 - momentum",
            "Jegadeesh (1990) / Lehmann (1990) - short-term reversal",
            "Gervais, Kaniel & Mingelgrin (2001), J. Finance 56(3):877-919 - high-volume premium",
            "Tetlock (2007), J. Finance 62(3):1139-1168 - media sentiment",
        ],
        "disclaimer": "Literature-scaled drift tilt, not a fitted model or prediction.",
    }


def expected_return_at(mu_annual: float, T_years: float) -> float:
    """Expected total return over horizon T under the forecast drift."""
    return math.exp(mu_annual * T_years) - 1.0


def target_at(spot: float, mu_annual: float, T_years: float) -> float:
    """Expected price at horizon T = spot * exp(mu*T)."""
    return spot * math.exp(mu_annual * T_years)


def forecast_for_ticker(
    ticker: str,
    spot: float | None = None,
    *,
    window: int = 300,
    sentiment_score: float | None = None,
) -> dict | None:
    """Fetch trailing daily candles and compute the drift forecast."""
    try:
        from core.data.fetcher import fetch_candles

        df = fetch_candles(ticker, "1d", window, False)
        return compute_forecast(
            df["close"].to_numpy(dtype=float),
            df["volume"].to_numpy(dtype=float) if "volume" in df else None,
            spot=spot,
            sentiment_score=sentiment_score,
        )
    except Exception as e:
        logger.debug("[forecast] %s failed: %s", ticker, e)
        return None


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    base = np.linspace(0, 0.6, 260)
    noise = rng.normal(0, 0.01, 260).cumsum()
    up = 100 * np.exp(base + noise)
    vol = np.full(260, 1_000_000.0)
    f_up = compute_forecast(up, vol, sentiment_score=0.4)
    assert f_up is not None
    print("UPTREND:", f_up["direction"], f_up["expected_monthly_return_pct"], "%/mo",
          "mu", f_up["mu_annual_pct"], "% target1m", f_up["target_1m"])
    assert f_up["direction"] == "bullish"

    flat = 100 * np.exp(rng.normal(0, 0.005, 260).cumsum())
    flat[-21:] = flat[-22] * np.exp(np.linspace(0, 0.25, 21))
    f_spike = compute_forecast(flat, vol)
    rev = next(x for x in f_spike["factors"] if x["name"] == "reversal")
    assert rev["contribution_pct"] < 0, "recent spike should give negative reversal"
    print("SPIKE reversal contrib:", rev["contribution_pct"], "%")

    f_pos = compute_forecast(flat, vol, sentiment_score=0.8)
    f_neg = compute_forecast(flat, vol, sentiment_score=-0.8)
    assert f_pos["expected_monthly_return_pct"] > f_neg["expected_monthly_return_pct"]
    print("sentiment +/-:", f_pos["expected_monthly_return_pct"], f_neg["expected_monthly_return_pct"])

    mu = f_up["mu_annual"]
    assert abs(expected_return_at(mu, 90 / 365)) > abs(expected_return_at(mu, 7 / 365))
    print("All forecast self-tests passed.")
