"""Tests for core/forecast.py."""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core.analysis.forecast import (
    _MAX_MONTHLY_TOTAL,
    _MONTH,
    compute_forecast,
    expected_return_at,
    target_at,
)

RNG = np.random.default_rng(0)
VOL = np.full(260, 1_000_000.0)


def _uptrend():
    base = np.linspace(0, 0.6, 260)
    noise = RNG.normal(0, 0.01, 260).cumsum()
    p = 100 * np.exp(base + noise)
    p[-_MONTH - 1 :] = p[-_MONTH - 1]
    return p


def _flat():
    return 100 * np.exp(RNG.normal(0, 0.005, 260).cumsum())


def test_too_short_returns_none():
    assert compute_forecast([100.0] * 10) is None


def test_uptrend_is_bullish():
    f = compute_forecast(_uptrend(), VOL)
    assert f is not None
    assert f["direction"] == "bullish"
    assert f["mu_annual"] > 0
    mom = next(x for x in f["factors"] if x["name"] == "momentum")
    assert mom["contribution_pct"] > 0


def test_recent_spike_gives_negative_reversal():
    p = _flat()
    p[-21:] = p[-22] * np.exp(np.linspace(0, 0.25, 21))
    f = compute_forecast(p, VOL)
    rev = next(x for x in f["factors"] if x["name"] == "reversal")
    assert rev["contribution_pct"] < 0


def test_high_volume_factor_is_nonnegative_and_present():
    p = _flat()
    v = VOL.copy()
    v[-5:] = 4_000_000.0
    f = compute_forecast(p, v)
    vol = next(x for x in f["factors"] if x["name"] == "volume")
    assert vol["contribution_pct"] >= 0
    assert vol["signal"] > 0


def test_sentiment_moves_drift_monotonically():
    p = _flat()
    f_pos = compute_forecast(p, VOL, sentiment_score=0.8)
    f_neg = compute_forecast(p, VOL, sentiment_score=-0.8)
    assert (f_pos["expected_monthly_return_pct"]
            > f_neg["expected_monthly_return_pct"])


def test_output_is_bounded():
    p = 100 * np.exp(np.linspace(0, 3.0, 260) + RNG.normal(0, 0.01, 260).cumsum())
    f = compute_forecast(p, VOL, sentiment_score=1.0)
    assert f is not None
    assert abs(f["expected_monthly_return_pct"]) <= _MAX_MONTHLY_TOTAL * 100 + 1e-6


def test_horizon_scaling_and_target():
    f = compute_forecast(_uptrend(), VOL)
    mu = f["mu_annual"]
    assert abs(expected_return_at(mu, 90 / 365)) > abs(expected_return_at(mu, 7 / 365))
    spot = f["spot"]
    tgt = target_at(spot, mu, 30 / 365)
    assert abs(tgt - spot * math.exp(mu * 30 / 365)) < 1e-9
    if mu > 0:
        assert f["target_1m"] > spot


def test_confidence_in_unit_range():
    f = compute_forecast(_uptrend(), VOL, sentiment_score=0.3)
    assert 0.0 <= f["confidence"] <= 1.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} forecast tests passed.")


if __name__ == "__main__":
    _run_all()
