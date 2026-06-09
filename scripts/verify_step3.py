"""Verification script."""

import numpy as np

from core.analysis.montecarlo import _calibrate_garch_mle, _simulate_ensemble, _simulate_fhs

rng = np.random.default_rng(11)
SIGMA = 0.02
N_SIM, N_STEPS = 40_000, 10

# History with fat tails + vol clustering (GJR-like) so FHS has something to learn
n_hist = 300
hist = np.empty(n_hist)
s2, e_prev = SIGMA**2, 0.0
for i in range(n_hist):
    s2 = 0.1 * SIGMA**2 + (0.05 + 0.10 * (e_prev < 0)) * e_prev**2 + 0.80 * s2
    e_prev = np.sqrt(s2) * rng.standard_t(5) / np.sqrt(5 / 3)
    hist[i] = e_prev

print("=" * 60)
print("1) FHS variance targeting & mean preservation (drift=0)")
print("=" * 60)
rets = _simulate_fhs(rng, N_SIM, N_STEPS, 0.0, SIGMA, hist)
cum = rets.sum(axis=1)
ref = SIGMA * np.sqrt(N_STEPS)
print(f"  std(sum log-ret) = {cum.std():.5f}   sigma*sqrt(n) = {ref:.5f}   ratio = {cum.std() / ref:.3f}")
assert 0.75 < cum.std() / ref < 1.6, "FHS variance scale off"  # GJR paths vary, loose band

growth = np.exp(cum)
se = growth.std() / np.sqrt(N_SIM)
print(f"  E[S_T/S_0] = {growth.mean():.4f} +/- {se:.4f}   (target 1.0)")
assert abs(growth.mean() - 1.0) < 4 * se + 0.01, "mean preservation violated"

print()
print("=" * 60)
print("2) FHS resamples the FILTERED residual pool faithfully")
print("=" * 60)
# Note: GARCH filtering is SUPPOSED to absorb most raw-return kurtosis into
# the sigma_t dynamics (fat tails ~ vol clustering x thinner innovations).
# The property FHS promises is that whatever shape the standardized residual
# pool has - fat, skewed, or near-Gaussian - is preserved exactly.
# Reconstruct the pool with the same (cached) MLE fit and filter:
window = hist[-90:]
omega, alpha, gamma, beta = _calibrate_garch_mle(window)
eps_h = window - window.mean()
s2 = np.empty(len(window))
s2[0] = max(float(np.var(window)), 1e-10)
for i in range(1, len(window)):
    s2[i] = omega + (alpha + gamma * (eps_h[i - 1] < 0)) * eps_h[i - 1] ** 2 + beta * s2[i - 1]
z = eps_h / np.sqrt(np.maximum(s2, 1e-12))
z = (z - z.mean()) / z.std()
kurt_pool = float(np.mean(z**4) - 3)
skew_pool = float(np.mean(z**3))

# Step-0 FHS returns are an affine map of a pool resample (sigma is constant
# across paths at t=0), so standardized moments must match the pool's.
r0 = rets[:, 0]
zz = (r0 - r0.mean()) / r0.std()
kurt_0 = float(np.mean(zz**4) - 3)
skew_0 = float(np.mean(zz**3))

raw_k = float(np.mean(((window - window.mean()) / window.std()) ** 4) - 3)
print(f"  raw window excess kurtosis : {raw_k:.2f}  (absorbed by GARCH filter)")
print(f"  residual pool: kurt={kurt_pool:.2f} skew={skew_pool:.2f}")
print(f"  FHS step-0   : kurt={kurt_0:.2f} skew={skew_0:.2f}")
assert abs(kurt_0 - kurt_pool) < 0.5, "step-0 distribution != residual pool (kurtosis)"
assert abs(skew_0 - skew_pool) < 0.25, "step-0 distribution != residual pool (skew)"

print()
print("=" * 60)
print("3) FHS fallback below 30 returns")
print("=" * 60)
short = hist[:20]
r_fb = _simulate_fhs(rng, 1000, 5, 0.0, SIGMA, short)
assert r_fb.shape == (1000, 5) and np.isfinite(r_fb).all()
print("  shape/finite ok (Gaussian-GARCH fallback)")

print()
print("=" * 60)
print("4) Ensemble (now GARCH + FHS + jump) still correct scale")
print("=" * 60)
re_ = _simulate_ensemble(rng, N_SIM, N_STEPS, SIGMA, 0.0, hist, kurtosis_excess=2.0)
cum_e = re_.sum(axis=1)
print(f"  ratio = {cum_e.std() / ref:.3f}   (mixture: ~1.0+, old averaging bug: ~0.65)")
assert cum_e.std() / ref > 0.85

print()
print("ALL CHECKS PASSED")
