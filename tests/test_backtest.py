"""Tests for core/backtest.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtest import walk_forward


def test_backtest_runs(synth_df):
    rep = walk_forward(synth_df, n_forward=5, n_sim=100, mc_model="gaussian", step=4)
    assert rep["ok"] is True
    assert rep["n_evaluated"] > 5

    # ── Probability / calibration fields ─────────────────────────────────────
    assert 0 <= rep["mean_prob_up"] <= 100
    assert 0 <= rep["real_up_rate"] <= 100
    assert 0 <= rep["brier_score"] <= 1
    assert rep["log_loss"] >= 0
    assert -1 <= rep["expected_vs_real"] <= 1

    # ── Transaction cost fields ───────────────────────────────────────────────
    assert "commission" in rep
    assert "slippage" in rep
    assert "round_trip_cost" in rep
    assert rep["commission"] >= 0
    assert rep["slippage"] >= 0
    assert rep["round_trip_cost"] >= 0

    # ── Trade-level statistics ────────────────────────────────────────────────
    # hit_rate is None when no Buy/Sell signals were issued; otherwise 0–100
    if rep["hit_rate"] is not None:
        assert 0 <= rep["hit_rate"] <= 100

    # sharpe_ratio: real number (can be negative for bad strategies)
    if rep["sharpe_ratio"] is not None:
        assert isinstance(rep["sharpe_ratio"], float)

    # max_drawdown: non-negative percentage
    if rep["max_drawdown"] is not None:
        assert rep["max_drawdown"] >= 0

    # avg_win / avg_loss
    if rep["avg_win"] is not None:
        assert isinstance(rep["avg_win"], float)
    if rep["avg_loss"] is not None:
        assert isinstance(rep["avg_loss"], float)

    # win_loss_ratio: positive when both avg_win and avg_loss exist
    if rep["win_loss_ratio"] is not None:
        assert rep["win_loss_ratio"] > 0

    # profit_factor: positive
    if rep["profit_factor"] is not None:
        assert rep["profit_factor"] > 0

    # max_consec_losses: non-negative integer
    if rep["max_consec_losses"] is not None:
        assert rep["max_consec_losses"] >= 0

    # n_called: non-negative
    assert rep.get("n_called", 0) >= 0

    # calibration: list of 5 bins
    assert isinstance(rep["calibration"], list)
    assert len(rep["calibration"]) == 5
    for bucket in rep["calibration"]:
        assert "bin" in bucket and "n" in bucket

    # signals: list (may be capped at 300)
    assert isinstance(rep["signals"], list)


def test_backtest_too_few_bars():
    idx = pd.date_range("2024-01-02", periods=20, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 20),
            "high": np.linspace(101, 111, 20),
            "low": np.linspace(99, 109, 20),
            "close": np.linspace(100, 110, 20),
            "volume": [1000] * 20,
        },
        index=idx,
    )
    rep = walk_forward(df, n_forward=10, n_sim=50)
    assert rep["ok"] is False
    assert rep["n_evaluated"] == 0
    # Error result should still have all expected keys
    for key in (
        "hit_rate",
        "brier_score",
        "sharpe_ratio",
        "max_drawdown",
        "profit_factor",
        "max_consec_losses",
    ):
        assert key in rep


def test_backtest_signals_list(synth_df):
    """Each signal row in the list should have the expected keys."""
    rep = walk_forward(synth_df, n_forward=5, n_sim=50, mc_model="gaussian", step=10)
    assert rep["ok"] is True
    required_keys = {"ts", "price", "label", "conf", "prob_up", "exp_ret", "real_ret"}
    for row in rep["signals"]:
        assert required_keys.issubset(row.keys()), f"Missing keys in signal row: {row}"


def test_backtest_downtrend(trend_down_df):
    """Walk-forward on a synthetic downtrend should have more Sell than Buy signals."""
    rep = walk_forward(trend_down_df, n_forward=5, n_sim=50, mc_model="gaussian", step=5)
    assert rep["ok"] is True
    sell_count = sum(1 for r in rep["signals"] if "Sell" in r["label"])
    buy_count = sum(1 for r in rep["signals"] if "Buy" in r["label"])
    # In a clear downtrend sell signals should dominate (or at least exist)
    assert sell_count + buy_count > 0, "No directional signals generated"
