"""Tests for core/backtest.py."""

from __future__ import annotations

from core.backtest import walk_forward


def test_backtest_runs(synth_df):
    rep = walk_forward(synth_df, n_forward=5, n_sim=100, mc_model="gaussian", step=4)
    assert rep["ok"] is True
    assert rep["n_evaluated"] > 5
    # Probabilities sane
    assert 0 <= rep["mean_prob_up"] <= 100
    assert 0 <= rep["real_up_rate"] <= 100
    assert 0 <= rep["brier_score"] <= 1
    assert rep["log_loss"] >= 0
    assert -1 <= rep["expected_vs_real"] <= 1


def test_backtest_too_few_bars():
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2024-01-02", periods=20, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(100, 110, 20), "high": np.linspace(101, 111, 20),
        "low": np.linspace(99, 109, 20),   "close": np.linspace(100, 110, 20),
        "volume": [1000] * 20,
    }, index=idx)
    rep = walk_forward(df, n_forward=10, n_sim=50)
    assert rep["ok"] is False
    assert rep["n_evaluated"] == 0
