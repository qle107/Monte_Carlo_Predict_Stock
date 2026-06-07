"""Verification script."""

import os
import tempfile

import numpy as np
import pandas as pd

from core.conformal import BandCalibrator, warm_start_from_history

rng = np.random.default_rng(53)

db = os.path.join(tempfile.gettempdir(), "verify_warmstart.db")
for f in (db, db + "-wal", db + "-shm"):
    if os.path.exists(f):
        os.remove(f)
cal = BandCalibrator(db)

# Synthetic OHLCV: 250 bars of a noisy random walk
n = 250
rets = rng.standard_normal(n) * 0.015
close = 100.0 * np.exp(np.cumsum(rets))
high = close * (1 + np.abs(rng.standard_normal(n)) * 0.004)
low = close * (1 - np.abs(rng.standard_normal(n)) * 0.004)
open_ = np.roll(close, 1)
open_[0] = close[0]
vol = rng.integers(1_000, 50_000, n).astype(float)
df = pd.DataFrame(
    {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
    index=pd.date_range("2026-01-05", periods=n, freq="15min"),
)

print("=" * 60)
print("1) Warm start scores history and seeds the state")
print("=" * 60)
n_scored = warm_start_from_history(cal, df, "WARM", "15m", 7, "garch", n_sim=300)
cov = cal.coverage("WARM", "15m", 7)
print(f"  scored {n_scored} historical forecasts")
print(f"  state: {cov}")
assert n_scored >= 10, "warm start should score >= 10 origins on 250 bars"
assert cov["n_settled"] == n_scored
assert cov["empirical_coverage"] is not None, "coverage must be available immediately"
a = cal.target_alpha("WARM", "15m", 7)
print(f"  target_alpha now active: {a:.3f} (no waiting for live forecasts)")

print()
print("=" * 60)
print("2) Warm start never overwrites existing state")
print("=" * 60)
again = warm_start_from_history(cal, df, "WARM", "15m", 7, "garch", n_sim=300)
assert again == 0, "second warm start must be a no-op"
cal.seed_state("WARM", "15m", 7, [1, 1, 1, 1, 1])  # also a no-op now
cov2 = cal.coverage("WARM", "15m", 7)
assert cov2["n_settled"] == cov["n_settled"], "live/seeded state was overwritten"
print("  re-seed correctly refused")

print()
print("=" * 60)
print("3) Too-short history returns 0 gracefully")
print("=" * 60)
short = df.iloc[:70]
assert warm_start_from_history(cal, short, "SHORT", "15m", 7, "garch") == 0
print("  ok")

print()
print("ALL CHECKS PASSED")
