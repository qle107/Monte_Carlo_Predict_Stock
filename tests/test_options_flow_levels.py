"""Verify GEX / Max Pain / Wall / Gamma Flip math with a synthetic ARM-like chain.

Reproduces a known bug scenario: spot $382.66 with a
giant legacy deep-ITM $300 call block on a far expiry. Before the fix every
level pinned to $300 and gamma flip fell to $270 (bottom of the strike window).

Run from repo root:  python tests/test_options_flow_levels.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from core.options_flow import (
    _compute_gex_profile,
    _compute_max_pain,
    _contracts_from_chains,
    _find_gamma_flip,
    _find_walls,
    _total_gex_at,
)

SPOT = 382.66


def _df(rows):
    return pd.DataFrame(rows, columns=["strike", "openInterest", "impliedVolatility"])


def build_chains():
    # Near expiry (7 DTE): liquidity clustered around spot, like the heatmap.
    near_calls = _df(
        [
            (370, 3000, 0.55), (375, 4000, 0.54), (380, 9000, 0.53),
            (385, 7000, 0.53), (390, 8000, 0.52), (395, 12000, 0.52),
            (400, 16000, 0.51), (405, 5000, 0.51), (410, 11000, 0.50),
            (415, 4000, 0.50), (420, 3000, 0.50),
        ]
    )
    near_puts = _df(
        [
            (340, 4000, 0.60), (350, 9000, 0.58), (355, 3000, 0.57),
            (360, 5000, 0.57), (365, 6000, 0.56), (370, 8000, 0.55),
            (375, 14000, 0.55), (380, 10000, 0.54), (385, 3000, 0.53),
        ]
    )
    # Far expiry (78 DTE): legacy deep-ITM $300 call block + put block at 330.
    far_calls = _df([(300, 35000, 0.65), (350, 6000, 0.55), (400, 4000, 0.50)])
    far_puts = _df([(300, 8000, 0.70), (330, 20000, 0.62), (350, 12000, 0.58)])

    return [
        ("2026-06-12", 7 / 365.0, near_calls, near_puts),
        ("2026-08-21", 78 / 365.0, far_calls, far_puts),
    ]


def main():
    chains = build_chains()
    _, _, near_calls, near_puts = chains[0]

    failures = []

    # 1) Max pain: nearest expiry only -> must sit in the near-spot cluster,
    #    NOT at the far-chain legacy $300 block.
    mp = _compute_max_pain(near_calls, near_puts)
    print(f"max_pain     = {mp:.2f}  (expect 360-410, NOT 300)")
    if not (360 <= mp <= 410):
        failures.append("max_pain outside 360-410")

    contracts = _contracts_from_chains(chains)

    # 2) Walls: aggregated OI, constrained to the correct side of spot.
    cw, pw = _find_walls(contracts, SPOT)
    print(f"call_wall    = {cw:.2f}  (expect >= spot, peak call OI -> 400)")
    print(f"put_wall     = {pw:.2f}  (expect <= spot)")
    if cw < SPOT * 0.99:
        failures.append("call_wall below spot")
    if cw != 400:
        failures.append(f"call_wall != 400 (got {cw})")
    if pw > SPOT * 1.01:
        failures.append("put_wall above spot")

    # 3) Gamma flip: price-scan zero crossing, must not collapse to the
    #    bottom of the strike window (~270-306) like the old code.
    flip = _find_gamma_flip(contracts, SPOT)
    print(f"gamma_flip   = {flip:.2f}  (expect inside (310, 420), not bottom of window)")
    if not (SPOT * 0.81 < flip < SPOT * 1.19):
        failures.append("gamma_flip pinned to scan boundary")

    # Flip must actually be a zero crossing: signs differ on either side.
    g_lo, g_hi = _total_gex_at(contracts, flip - 2), _total_gex_at(contracts, flip + 2)
    print(f"GEX(flip-2)  = {g_lo:,.0f} | GEX(flip+2) = {g_hi:,.0f}  (expect sign change)")
    if (g_lo < 0) == (g_hi < 0):
        failures.append("no sign change around gamma_flip")

    # 4) GEX profile and scaling: 0.01 per-1%-move factor applied.
    bars = _compute_gex_profile(contracts, SPOT)
    net = sum(b.gex for b in bars)
    print(f"net_gex      = {net:,.0f}  (with 0.01 factor; old code was 100x larger)")
    bar_400 = next(b for b in bars if b.strike == 400)
    # Hand-check one bar: scaled GEX must be exactly 1% of gamma*OI*100*S^2.
    from core.options_flow import _GEX_SCALE, _bs_gamma

    g = _bs_gamma(SPOT, 400, 7 / 365.0, 0.51) * 16000 + _bs_gamma(SPOT, 400, 78 / 365.0, 0.50) * 4000
    expected_400 = g * SPOT * SPOT * _GEX_SCALE
    print(f"bar@400      = {bar_400.gex:,.0f} vs hand-calc {expected_400:,.0f}")
    if abs(bar_400.gex - expected_400) > abs(expected_400) * 1e-6:
        failures.append("GEX scaling mismatch at strike 400")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
