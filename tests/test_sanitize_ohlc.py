"""Tests for core.data.fetcher._sanitize_ohlc."""

import numpy as np
import pandas as pd

from core.data.fetcher import _sanitize_ohlc


def _base_frame(n=60, price=1000.0, seed=0):
    """A calm ~1000 series of 15m bars with small, realistic moves."""
    rng = np.random.default_rng(seed)
    closes = price + np.cumsum(rng.normal(0, 3.0, n))
    idx = pd.date_range("2026-06-01 13:30", periods=n, freq="15min", tz="UTC")
    opens = closes + rng.normal(0, 1.0, n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 1.5, n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 1.5, n))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": rng.integers(1000, 5000, n)},
        index=idx,
    )


def test_removes_high_spike():
    df = _base_frame()
    bad = 30
    df.loc[df.index[bad], "high"] = 1770.0  # the bug
    out = _sanitize_ohlc(df)
    local = out["close"].iloc[bad - 3:bad + 4].median()
    assert out["high"].iloc[bad] < local * 1.6, out["high"].iloc[bad]
    print(f"high spike 1770 -> {out['high'].iloc[bad]:.2f} (local ~{local:.0f})  OK")


def test_removes_low_spike():
    df = _base_frame(seed=1)
    bad = 25
    df.loc[df.index[bad], "low"] = 5.0  # downside bad tick
    out = _sanitize_ohlc(df)
    local = out["close"].iloc[bad - 3:bad + 4].median()
    assert out["low"].iloc[bad] > local * 0.5, out["low"].iloc[bad]
    print(f"low spike 5 -> {out['low'].iloc[bad]:.2f} (local ~{local:.0f})  OK")


def test_enforces_ohlc_integrity():
    df = _base_frame(seed=2)
    out = _sanitize_ohlc(df)
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (out["high"] >= out["low"]).all()
    print("OHLC integrity (high>=O/C>=low) preserved  OK")


def test_preserves_normal_bars():
    df = _base_frame(seed=3)
    out = _sanitize_ohlc(df)
    # No corruption injected -> data should be essentially unchanged.
    assert np.allclose(out["close"].to_numpy(), df["close"].to_numpy())
    assert np.allclose(out["high"].to_numpy(), df["high"].to_numpy())
    print("clean data passes through untouched  OK")


def test_preserves_legit_gap():
    """A real ~10% move/gap must survive (not be flattened as a glitch)."""
    df = _base_frame(seed=4)
    g = 40
    shift = df["close"].iloc[g] * 0.10
    df.loc[df.index[g]:, ["open", "high", "low", "close"]] += shift
    out = _sanitize_ohlc(df)
    assert out["close"].iloc[-1] > df["close"].iloc[g - 1] * 1.07
    print(f"legit +10% gap preserved (end={out['close'].iloc[-1]:.0f})  OK")


if __name__ == "__main__":
    test_removes_high_spike()
    test_removes_low_spike()
    test_enforces_ohlc_integrity()
    test_preserves_normal_bars()
    test_preserves_legit_gap()
    print("\nAll tests passed.")
