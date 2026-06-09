"""Verification script."""

import numpy as np

from core.analysis.hurst import dfa
from core.analysis.montecarlo import _compute_regime_state, _simulate_ensemble, _simulate_jump

rng = np.random.default_rng(42)
SIGMA = 0.02
N_SIM, N_STEPS = 40_000, 10

print("=" * 60)
print("1) DFA minimum-N behaviour")
print("=" * 60)
a60, se60 = dfa(rng.standard_normal(60))
a128, se128 = dfa(rng.standard_normal(128))
print(f"  N=60  -> alpha={a60:.3f}, se={se60:.3f}  (old window: fallback, se=0)")
print(f"  N=128 -> alpha={a128:.3f}, se={se128:.3f}  (real estimate, se>0)")
assert (a60, se60) == (0.5, 0.0), "N=60 should hit the fallback (documents old bug)"
assert se128 > 0, "N=128 must produce a real regression"

# Regime state on a persistent (trending) series must not be stuck neutral
ar = np.zeros(256)
eps = rng.standard_normal(256) * SIGMA
for i in range(1, 256):
    ar[i] = 0.45 * ar[i - 1] + eps[i]  # AR(1) -> persistent, alpha > 0.55
regime, H, *_ = _compute_regime_state(ar)
print(f"  AR(1) phi=0.45, N=256 -> regime={regime!r}, alpha={H:.3f}")
assert H != 0.5, "regime DFA must actually run now"

print()
print("=" * 60)
print("2) Ensemble mixture: variance no longer shrunken")
print("=" * 60)
hist = rng.standard_normal(300) * SIGMA
rets = _simulate_ensemble(rng, N_SIM, N_STEPS, SIGMA, 0.0, hist, kurtosis_excess=1.0)
cum = rets.sum(axis=1)
emp_std = cum.std()
# Mixture std should be ~ sigma*sqrt(n_steps) or larger (jump component adds var).
# The old averaging implementation gave ~0.65 * sigma * sqrt(n).
ref = SIGMA * np.sqrt(N_STEPS)
print(f"  std(sum log-ret) = {emp_std:.5f}   reference sigma*sqrt(n) = {ref:.5f}")
print(f"  ratio = {emp_std / ref:.3f}   (old buggy code: ~0.65, fixed: >= ~0.95)")
assert emp_std / ref > 0.85, "variance still shrunken -> mixture fix failed"

mean_growth = np.exp(cum).mean()
se = np.exp(cum).std() / np.sqrt(N_SIM)
print(f"  E[S_T/S_0] = {mean_growth:.4f} +/- {se:.4f}   (target 1.0, drift=0)")
assert abs(mean_growth - 1.0) < 4 * se + 0.01, "mean preservation violated"

print()
print("=" * 60)
print("3) Jump model mean preservation (compensator)")
print("=" * 60)
rj = _simulate_jump(rng, N_SIM, N_STEPS, 0.0, SIGMA, jump_intensity=0.05)
g = np.exp(rj.sum(axis=1))
print(f"  E[S_T/S_0] = {g.mean():.4f} +/- {g.std() / np.sqrt(N_SIM):.4f}   (target 1.0)")
assert abs(g.mean() - 1.0) < 4 * g.std() / np.sqrt(N_SIM) + 0.01

print()
print("ALL CHECKS PASSED")
