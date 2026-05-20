"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable when pytest is invoked from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _make_candles(
    n: int = 200, seed: int = 42, drift: float = 0.0002, vol: float = 0.01, start_price: float = 100.0
) -> pd.DataFrame:
    """Synthetic OHLCV with controlled drift/vol — deterministic per seed."""
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    closes = start_price * np.cumprod(1 + rets)

    # Make plausible OHLC around closes
    opens = np.concatenate([[start_price], closes[:-1]])
    spread = vol * closes
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 0.4, n)) * spread
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.4, n)) * spread
    vols = rng.integers(1_000, 100_000, n)

    idx = pd.date_range("2024-01-02", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        },
        index=idx,
    )
    return df


@pytest.fixture(scope="session")
def synth_df() -> pd.DataFrame:
    return _make_candles()


@pytest.fixture
def trend_up_df() -> pd.DataFrame:
    return _make_candles(seed=1, drift=0.005, vol=0.008)


@pytest.fixture
def trend_down_df() -> pd.DataFrame:
    return _make_candles(seed=2, drift=-0.005, vol=0.008)


@pytest.fixture
def tmp_db(tmp_path) -> str:
    return str(tmp_path / "test_signals.db")
