"""Verification script."""

import numpy as np

from core.analysis.montecarlo import _optimal_block_length, _simulate_bootstrap

rng = np.random.default_rng(47)
N = 400


def make_ar1(phi, n=N, sigma=0.02):
    x = np.zeros(n)
    eps = rng.standard_normal(n) * sigma
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


print("=" * 60)
print("1) Block length adapts to dependence structure")
print("=" * 60)
b_wn = np.mean([_optimal_block_length(rng.standard_normal(N) * 0.02) for _ in range(20)])
b_ar4 = np.mean([_optimal_block_length(make_ar1(0.4)) for _ in range(20)])
b_ar8 = np.mean([_optimal_block_length(make_ar1(0.8)) for _ in range(20)])
heuristic = N ** (1 / 3)
print(f"  white noise : b = {b_wn:5.1f}")
print(f"  AR(1) phi=.4: b = {b_ar4:5.1f}")
print(f"  AR(1) phi=.8: b = {b_ar8:5.1f}")
print(f"  old heuristic (all three identical): b = {heuristic:.1f}")
assert b_wn < b_ar4 < b_ar8, "block length must increase with persistence"
assert b_wn < heuristic, "white noise should get shorter blocks than N^(1/3)"
assert b_ar8 > heuristic, "strong persistence should get longer blocks than N^(1/3)"

print()
print("=" * 60)
print("2) Bootstrap preserves autocorrelation under persistence")
print("=" * 60)
ar = make_ar1(0.6)
rets = _simulate_bootstrap(rng, 2000, 100, ar, 0.0, float(np.std(ar)))
# Per-path lag-1 autocorrelation of the resampled returns
x = rets - rets.mean(axis=1, keepdims=True)
num = (x[:, :-1] * x[:, 1:]).sum(axis=1)
den = (x**2).sum(axis=1)
acf1 = float(np.mean(num / np.maximum(den, 1e-18)))
src = float(np.corrcoef(ar[:-1], ar[1:])[0, 1])
print(f"  source ACF(1) = {src:.3f}   resampled mean ACF(1) = {acf1:.3f}")
assert acf1 > 0.25, "stationary bootstrap lost the serial correlation"

print()
print("=" * 60)
print("3) Degenerate inputs fall back safely")
print("=" * 60)
assert _optimal_block_length(np.zeros(200)) >= 2.0  # zero variance
assert _optimal_block_length(rng.standard_normal(30)) >= 2.0  # too short
r = _simulate_bootstrap(rng, 500, 10, rng.standard_normal(200) * 0.02, 0.0, 0.02)
assert r.shape == (500, 10) and np.isfinite(r).all()
print("  zero-variance, short-series, and end-to-end shapes ok")

print()
print("ALL CHECKS PASSED")
