"""Tests for core/signal.py."""

from __future__ import annotations

from core.indicators import compute_indicators
from core.signal import compute_signal


def test_signal_basic(synth_df):
    sig = compute_signal(compute_indicators(synth_df))
    assert -1.0 <= sig.composite <= 1.0
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.label in {"Strong buy", "Buy", "Neutral", "Sell", "Strong sell"}
    # vol_adj is clamped between 0.003 and 0.06
    assert 0.003 <= sig.vol_adj <= 0.06
    assert isinstance(sig.sub_scores, dict) and sig.sub_scores
    # sub_scores all in [-1, 1]
    for v in sig.sub_scores.values():
        assert -1.0 <= v <= 1.0


def test_drift_clamped_to_2sigma(synth_df):
    ind = compute_indicators(synth_df)
    sig = compute_signal(ind)
    std_dec = ind.std_return / 100.0
    assert abs(sig.drift_bias) <= 2.0 * std_dec + 1e-9


def test_uptrend_label_bullish(trend_up_df):
    sig = compute_signal(compute_indicators(trend_up_df))
    assert sig.composite >= 0
    # Strong synthetic uptrend → likely Buy or Strong buy
    assert "buy" in sig.label.lower() or sig.label == "Neutral"


def test_downtrend_label_bearish(trend_down_df):
    sig = compute_signal(compute_indicators(trend_down_df))
    assert sig.composite <= 0
    assert "sell" in sig.label.lower() or sig.label == "Neutral"
