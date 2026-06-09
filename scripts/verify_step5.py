"""Verification script."""

import numpy as np

from core.analysis.hurst import dfa
from core.analysis.montecarlo import _compute_regime_state

rng = np.random.default_rng(31)

print("=" * 60)
print("1) White noise: insignificant alpha deviations stay neutral")
print("=" * 60)
flips = 0
trials = 200
for _ in range(trials):
    regime, H, se, *_ = _compute_regime_state(rng.standard_normal(128) * 0.02)
    if regime != "neutral":
        flips += 1
print(f"  non-neutral classifications on pure noise: {flips}/{trials}")
# Exact permutation p-value at 0.025 per side -> false-switch rate <= 5%
# two-sided by construction (binomial SE over 200 trials ~1.5%).
assert flips / trials < 0.09, "gate not suppressing noise-driven regime flips"

# Compare: how often would the UNGATED thresholds have flipped?
ungated = 0
for _ in range(trials):
    H, se = dfa(rng.standard_normal(128) * 0.02)
    if H > 0.55 or H < 0.45:
        ungated += 1
print(f"  (ungated threshold-only rule would flip: {ungated}/{trials})")

print()
print("=" * 60)
print("2) Strongly persistent series still classified as trending")
print("=" * 60)
hits = 0
for _ in range(50):
    n = 256
    ar = np.zeros(n)
    eps = rng.standard_normal(n) * 0.02
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + eps[i]  # strong persistence
    regime, H, se, *_ = _compute_regime_state(ar)
    if regime == "trending":
        hits += 1
print(f"  trending detections on AR(1) phi=0.6: {hits}/50  (alpha~0.75+, clearly significant)")
# Exact test is conservative by design; demand >=60% power against a strong
# signal while the false-positive rate above stays <=5%.
assert hits >= 30, "gate too strict - real persistence missed"

print()
print("=" * 60)
print("3) Return tuple shape & SE exposure")
print("=" * 60)
out = _compute_regime_state(rng.standard_normal(128) * 0.02)
assert len(out) == 6, "must return (regime, alpha, se, drift_m, grav_m, sigma_m)"
regime, H, se, dm, gm, sm = out
assert se > 0, "SE must be exposed for valid input"
print(f"  regime={regime!r} alpha={H:.3f} se={se:.3f} mults=({dm}, {gm}, {sm})")

print()
print("ALL CHECKS PASSED")
