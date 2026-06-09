"""Verification script."""

import numpy as np

from core.analysis.hawkes import fit_jump_params
from core.analysis.montecarlo import _fit_hawkes_cached, _simulate_jump

rng = np.random.default_rng(23)
SIGMA = 0.02
N_SIM, N_STEPS = 60_000, 20
LAM = 0.05

# Synthetic Hawkes parameters as if fitted: baseline 0.08/bar, n=0.5, decay 0.4
HK = (0.08, 0.20, 0.40, 0.32)  # (mu, alpha, beta, lam_now = 2x stationary mean)

print("=" * 60)
print("1) Mean jump rate re-anchored to target lambda")
print("=" * 60)
# Long horizon so the initial 2x state washes out; count effective jumps by
# comparing against a no-jump diffusion: use kurtosis proxy instead - simpler:
# re-run with jump_mean strongly negative so jumps are identifiable.
r = _simulate_jump(rng, N_SIM, 100, 0.0, SIGMA, jump_intensity=LAM, jump_mean=-0.5, jump_sigma_mult=1.0, hawkes=HK)
jumps = r < -0.25  # a -0.5 jump dwarfs sigma=0.02 diffusion
rate = float(jumps.mean())
print(f"  empirical jump rate = {rate:.4f}   target = {LAM}")
assert abs(rate - LAM) < 0.008, "stationary jump rate not anchored to target"

print()
print("=" * 60)
print("2) Self-excitation: jumps cluster (aftershocks)")
print("=" * 60)
j = jumps[:, 10:]  # skip burn-in of the initial state
prev, nxt = j[:, :-1], j[:, 1:]
p_after = float(nxt[prev].mean())
p_calm = float(nxt[~prev].mean())
print(f"  P(jump | jump at t-1)    = {p_after:.4f}")
print(f"  P(jump | no jump at t-1) = {p_calm:.4f}")
print(f"  clustering ratio = {p_after / p_calm:.2f}  (constant-lambda Merton = 1.0)")
assert p_after > p_calm * 1.5, "no self-excitation detected"

# Control: constant-lambda branch must NOT cluster
rc = _simulate_jump(rng, N_SIM, 100, 0.0, SIGMA, jump_intensity=LAM, jump_mean=-0.5, jump_sigma_mult=1.0)
jc = rc < -0.25
pc_after = float(jc[:, 1:][jc[:, :-1]].mean())
pc_calm = float(jc[:, 1:][~jc[:, :-1]].mean())
print(f"  control (Merton) ratio = {pc_after / pc_calm:.2f}  (~1.0 expected)")
assert abs(pc_after / pc_calm - 1.0) < 0.15

print()
print("=" * 60)
print("3) Exact compensator: mean preservation under Hawkes")
print("=" * 60)
r2 = _simulate_jump(rng, N_SIM, N_STEPS, 0.0, SIGMA, jump_intensity=LAM, hawkes=HK)
g = np.exp(r2.sum(axis=1))
se = g.std() / np.sqrt(N_SIM)
print(f"  E[S_T/S_0] = {g.mean():.4f} +/- {se:.4f}   (target 1.0, drift=0)")
assert abs(g.mean() - 1.0) < 4 * se + 0.005, "compensator broken"

print()
print("=" * 60)
print("4) fit_jump_params on clustered synthetic returns")
print("=" * 60)
# Build a return series with clustered shocks
n_hist = 400
lam_t, exc = 0.10, 0.0
rets = np.empty(n_hist)
for i in range(n_hist):
    lam_t = 0.05 + exc
    jump = rng.random() < lam_t
    rets[i] = rng.standard_normal() * SIGMA + jump * rng.standard_normal() * 5 * SIGMA
    exc = (exc + 0.25 * jump) * np.exp(-0.5)
fit = fit_jump_params(rets)
print(f"  fitted: {None if fit is None else tuple(round(x, 4) for x in fit)}")
assert fit is not None, "fit failed on clearly clustered data"
mu_f, a_f, b_f, lam0 = fit
assert 0 < a_f / b_f < 1, "branching ratio out of range"

cached = _fit_hawkes_cached(rets)
assert cached == fit, "cache round-trip mismatch"
print("  cache round-trip ok")

print()
print("ALL CHECKS PASSED")
