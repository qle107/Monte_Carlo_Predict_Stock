"""Verification script."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np

from core.conformal import ALPHA_TARGET, BandCalibrator

rng = np.random.default_rng(43)

db = os.path.join(tempfile.gettempdir(), "verify_conformal.db")
for f in (db, db + "-wal", db + "-shm"):
    if os.path.exists(f):
        os.remove(f)
cal = BandCalibrator(db)

T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
TICKER, IV, H = "TEST", "1h", 5  # maturity = 5 hours


def run_cycle(n, miss_rate, start, alpha_probe):
    """Issue n bands; a fraction `miss_rate` will realise outside the band."""
    t = start
    for i in range(n):
        spot = 100.0
        lo, hi = 95.0, 105.0
        cal.observe(TICKER, IV, H, t.isoformat(), spot, lo, hi)
        # settle far in the future so the band is mature
        realized = 110.0 if rng.random() < miss_rate else 100.0
        cal.settle(TICKER, IV, realized, (t + timedelta(hours=H + 1)).isoformat())
        t += timedelta(hours=1)
    return cal.target_alpha(TICKER, IV, H)


print("=" * 60)
print("1) Under-coverage -> alpha falls -> bands widen")
print("=" * 60)
# 50% miss rate vs 20% target -> ACI must push alpha down hard
a = run_cycle(80, miss_rate=0.5, start=T0, alpha_probe=True)
cov = cal.coverage(TICKER, IV, H)
print(f"  after 80 settles at ~50% miss: alpha = {a:.3f} (start 0.20)")
print(f"  coverage record: {cov}")
assert a < 0.15, "alpha should have decreased (wider bands)"
assert cov["n_settled"] == 80

print()
print("=" * 60)
print("2) Over-coverage -> alpha rises -> bands tighten")
print("=" * 60)
a2 = run_cycle(120, miss_rate=0.0, start=T0 + timedelta(days=30), alpha_probe=True)
print(f"  after 120 settles at 0% miss: alpha = {a2:.3f}")
assert a2 > a, "alpha should recover upward when bands never miss"

print()
print("=" * 60)
print("3) Cold start: nominal alpha until MIN_SETTLED")
print("=" * 60)
a3 = cal.target_alpha("FRESH", IV, H)
assert a3 == ALPHA_TARGET
print(f"  unseen ticker -> alpha = {a3} (nominal)")

print()
print("=" * 60)
print("4) Immature forecasts are not settled")
print("=" * 60)
t = T0 + timedelta(days=90)
cal.observe("IMM", IV, H, t.isoformat(), 100.0, 95.0, 105.0)
n = cal.settle("IMM", IV, 100.0, (t + timedelta(hours=1)).isoformat())  # < 5h maturity
assert n == 0, "settled an immature forecast"
n = cal.settle("IMM", IV, 100.0, (t + timedelta(hours=6)).isoformat())
assert n == 1, "mature forecast not settled"
print("  maturity gating ok")

print()
print("=" * 60)
print("5) MC engine consumes band_alpha")
print("=" * 60)
from core.montecarlo import _build_mc_result

paths = 100.0 * np.exp(np.cumsum(rng.standard_normal((20000, 6)) * 0.02, axis=1))
paths = np.hstack([np.full((20000, 1), 100.0), paths])
res_n = _build_mc_result(rng, paths, 100.0, "gaussian", 20000, band_alpha=0.20)
res_w = _build_mc_result(rng, paths, 100.0, "gaussian", 20000, band_alpha=0.05)
width_n = res_n.p90_price - res_n.p10_price
width_w = res_w.p90_price - res_w.p10_price
print(f"  band width alpha=0.20: {width_n:.2f}   alpha=0.05: {width_w:.2f}")
assert width_w > width_n * 1.2, "smaller alpha must widen the outer band"
assert res_w.band_alpha == 0.05
print("  outer band responds to calibrated alpha; band_alpha reported")

print()
print("ALL CHECKS PASSED")
