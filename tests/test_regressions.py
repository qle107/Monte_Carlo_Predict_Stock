"""Regression tests for OHLC sanitization, DFA, and bootstrap simulation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from core.analysis.hurst import dfa
from core.analysis.montecarlo import _simulate_bootstrap
from core.data.fetcher import _sanitize_ohlc


def _base_frame(n: int = 60, price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = price + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2026-06-01 13:30", periods=n, freq="15min", tz="UTC")
    opens = closes + rng.normal(0, 0.2, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.3, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.integers(1000, 5000, n),
        },
        index=idx,
    )


def test_sanitize_ohlc_spike_wick_clamped():
    df = _base_frame()
    bad = 30
    body_hi = max(df["open"].iloc[bad], df["close"].iloc[bad])
    df.loc[df.index[bad], "high"] = body_hi + 50.0  # phantom upper wick
    out = _sanitize_ohlc(df)
    assert out["high"].iloc[bad] <= body_hi + 1e-6
    assert np.allclose(out["close"].to_numpy(), df["close"].to_numpy())


def test_sanitize_ohlc_zero_low_clamped():
    df = _base_frame(seed=1)
    bad = 20
    body_lo = min(df["open"].iloc[bad], df["close"].iloc[bad])
    df.loc[df.index[bad], "low"] = 0.0
    out = _sanitize_ohlc(df)
    assert out["low"].iloc[bad] >= body_lo - 1e-6
    assert len(out) == len(df)


def test_sanitize_ohlc_isolated_close_glitch_dropped():
    df = _base_frame(seed=2)
    bad = 25
    df.loc[df.index[bad], "close"] = df["close"].iloc[bad] * 100.0
    df.loc[df.index[bad], "high"] = df["close"].iloc[bad]
    out = _sanitize_ohlc(df)
    assert len(out) == len(df) - 1
    assert bad not in range(len(out)) or out["close"].iloc[bad] < df["close"].iloc[bad - 1] * 5


def test_sanitize_ohlc_clean_bars_untouched():
    df = _base_frame(seed=3)
    out = _sanitize_ohlc(df)
    assert np.allclose(out.to_numpy(), df.to_numpy())


def test_dfa_gbm_returns_alpha_near_half():
    rng = np.random.default_rng(7)
    n = 512
    sigma = 0.01
    log_returns = rng.normal(-0.5 * sigma**2, sigma, n)
    alpha, _se = dfa(log_returns)
    assert 0.40 <= alpha <= 0.60, f"DFA alpha {alpha} outside [0.40, 0.60]"


def test_simulate_bootstrap_shape_and_sigma():
    rng = np.random.default_rng(11)
    n_sim, n_steps = 500, 10
    sigma = 0.02
    recent = rng.standard_normal(200) * sigma
    out = _simulate_bootstrap(rng, n_sim, n_steps, recent, 0.0, sigma)
    assert out.shape == (n_sim, n_steps)
    assert np.isfinite(out).all()
    emp_std = float(np.std(out))
    assert abs(emp_std - sigma) / sigma <= 0.15, f"std {emp_std} not within 15% of {sigma}"
