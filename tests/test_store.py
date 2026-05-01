"""Tests for core/store.py SignalStore."""

from __future__ import annotations

from core.store import SignalStore


def _fake_result(ticker="AAPL", price=100.0, label="Buy", prob_up=60.0):
    return {
        "ticker": ticker, "interval": "15m", "current_price": price,
        "updated_at": "2025-01-02T15:00:00+00:00", "mc_model": "gaussian",
        "signal": {"label": label, "confidence": 0.5, "drift_bias": 0.0001,
                   "sub_scores": {"rsi": 0.1}},
        "mc": {"prob_up": prob_up, "prob_flat": 30.0, "prob_down": 10.0,
               "median_price": price * 1.01, "expected_return": 0.5, "cvar_5": -2.0},
        "indicators": {},
    }


def test_record_and_recent(tmp_db):
    s = SignalStore(tmp_db)
    s.record(_fake_result("AAPL", 100, "Buy"))
    s.record(_fake_result("AAPL", 101, "Sell"))
    s.record(_fake_result("MSFT", 400, "Neutral"))

    rows = s.recent(limit=10)
    assert len(rows) == 3
    aapl = s.recent(ticker="AAPL")
    assert len(aapl) == 2
    assert all(r["ticker"] == "AAPL" for r in aapl)


def test_metrics(tmp_db):
    s = SignalStore(tmp_db)
    for i in range(5):
        s.record(_fake_result("AAPL", 100 + i, "Buy", prob_up=50 + i))
    m = s.metrics(ticker="AAPL")
    assert m["signals"] == 5
    assert 0 <= m["avg_prob_up"] <= 100
    assert "Buy" in m["label_counts"]


def test_record_tolerates_missing_fields(tmp_db):
    s = SignalStore(tmp_db)
    s.record({"ticker": "X", "signal": {}, "mc": {}, "current_price": 1.0})
    rows = s.recent()
    assert len(rows) == 1
