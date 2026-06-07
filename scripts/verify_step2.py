"""Verification script."""

import numpy as np

from config import cfg
from core.montecarlo import _calibrate_garch_mle, _simulate_garch

rng = np.random.default_rng(7)
SIGMA = 0.02

print("=" * 60)
print("1) Unconditional variance targeting (sigma_LR preserved)")
print("=" * 60)
eps, sig = _simulate_garch(rng, 5000, 60, SIGMA, None)
uncond = float(np.mean(sig[:, -1] ** 2))
print(f"  E[sigma^2] at t=60: {uncond:.3e}   target {SIGMA**2:.3e}")
assert abs(uncond / SIGMA**2 - 1.0) < 0.10, "long-run variance drifted"

print()
print("=" * 60)
print("2) Leverage asymmetry: vol higher after negative shocks")
print("=" * 60)
# Use a clearly asymmetric parameterisation
eps, sig = _simulate_garch(rng, 20000, 3, SIGMA, None, alpha=0.05, beta=0.75, gamma=0.20)
z0 = eps[:, 1]  # standardised shock at step 1
s2_next = sig[:, 2] ** 2  # variance at step 2
v_neg = float(np.mean(s2_next[z0 < 0]))
v_pos = float(np.mean(s2_next[z0 > 0]))
print(f"  E[sigma^2 | prev shock < 0] = {v_neg:.3e}")
print(f"  E[sigma^2 | prev shock > 0] = {v_pos:.3e}")
print(f"  ratio = {v_neg / v_pos:.3f}  (must be > 1; gamma=0 would give ~1.0)")
assert v_neg > v_pos * 1.05, "leverage term not firing"

# Symmetric control: gamma=0 must show no asymmetry
eps0, sig0 = _simulate_garch(rng, 20000, 3, SIGMA, None, alpha=0.05, beta=0.75, gamma=0.0)
r0 = float(np.mean(sig0[:, 2][eps0[:, 1] < 0] ** 2) / np.mean(sig0[:, 2][eps0[:, 1] > 0] ** 2))
print(f"  control (gamma=0) ratio = {r0:.3f}  (~1.0 expected)")
assert abs(r0 - 1.0) < 0.05

print()
print("=" * 60)
print("3) MLE recovers a known gamma")
print("=" * 60)
# Simulate a GJR(1,1) series with known parameters
true = dict(omega=2e-6, alpha=0.04, gamma=0.12, beta=0.82)
n = 3000
s2 = true["omega"] / (1 - true["alpha"] - 0.5 * true["gamma"] - true["beta"])
e_prev = 0.0
r = np.empty(n)
for i in range(n):
    s2 = true["omega"] + (true["alpha"] + true["gamma"] * (e_prev < 0)) * e_prev**2 + true["beta"] * s2
    e_prev = np.sqrt(s2) * rng.standard_normal()
    r[i] = e_prev
omega_f, alpha_f, gamma_f, beta_f = _calibrate_garch_mle(r)
print(f"  true : omega=2.0e-06 alpha=0.040 gamma=0.120 beta=0.820")
print(f"  fitted: omega={omega_f:.1e} alpha={alpha_f:.3f} gamma={gamma_f:.3f} beta={beta_f:.3f}")
assert gamma_f > 0.0, "gamma must be detected as positive"
assert alpha_f + 0.5 * gamma_f + beta_f < 0.999, "stationarity violated"

print()
print(f"4) config: garch_gamma default = {cfg.garch_gamma} (env GARCH_GAMMA overrides)")
print()
print("ALL CHECKS PASSED")
