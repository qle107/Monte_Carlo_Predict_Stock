"""Unit tests for the probability-of-profit / expected-return engine.

The probability math is validated against known closed-form cases:

  * For a long call, POP = P(S_T > breakeven) must equal N(d2) evaluated at the
    breakeven level under the same mu and sigma (exact agreement to a tolerance).
  * Probability ITM = N(d2) at the STRIKE - and must differ from POP.
  * The grid integrator's expected payoff must equal the Black-76 closed form.
  * MARKET-mode expected value (discounted) is ~0 at the option's fair value.
  * Monte Carlo (>=100k paths) agrees with the deterministic integrator.
  * Vertical-spread max gain / max loss / breakeven match their closed forms.

Run from the repo root:  python tests/test_pop_scanner.py
(or with pytest:          pytest tests/test_pop_scanner.py)
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core.scanners.pop_scanner import (
    _ZGRID,
    _ZW,
    CONTRACT_MULTIPLIER,
    Leg,
    Structure,
    evaluate_structure,
    evaluate_structure_mc,
    expected_call_payoff,
    expected_put_payoff,
    mu_from_target,
    prob_ST_above,
    scan,
)

SPOT = 100.0
T = 30.0 / 365.0
R = 0.05
SIG = 0.40


def _long_call(mid=3.10, K=100.0, comm=0.0):
    return Structure("long_call", f"Long {K:g}C",
                     [Leg("call", K, +1, mid, 0.0, SIG)], commission=comm)


def _long_put(mid=3.10, K=100.0, comm=0.0):
    return Structure("long_put", f"Long {K:g}P",
                     [Leg("put", K, +1, mid, 0.0, SIG)], commission=comm)


def test_pop_long_call_equals_Nd2_at_breakeven():
    st = _long_call()
    res = evaluate_structure(st, SPOT, T, R, SIG)
    be = st.breakevens()[0]
    pop_closed = prob_ST_above(SPOT, be, R, SIG, T) * 100.0
    assert abs(res["pop_pct"] - pop_closed) < 0.05, (res["pop_pct"], pop_closed)


def test_pop_long_put_equals_Nd2_at_breakeven():
    st = _long_put()
    res = evaluate_structure(st, SPOT, T, R, SIG)
    be = st.breakevens()[0]
    # Put profits when S_T < breakeven.
    pop_closed = (100.0 - prob_ST_above(SPOT, be, R, SIG, T) * 100.0)
    assert abs(res["pop_pct"] - pop_closed) < 0.05, (res["pop_pct"], pop_closed)


def test_prob_itm_equals_Nd2_at_strike_and_differs_from_pop():
    st = _long_call()
    res = evaluate_structure(st, SPOT, T, R, SIG)
    itm_closed = prob_ST_above(SPOT, 100.0, R, SIG, T) * 100.0
    assert abs(res["prob_itm_pct"] - itm_closed) < 0.05
    # POP requires clearing the premium, so it is strictly below prob ITM.
    assert res["prob_itm_pct"] > res["pop_pct"] + 1.0


def test_integrator_matches_black76_expected_payoff():
    s = SIG * math.sqrt(T)
    S_T = SPOT * np.exp((R - 0.5 * SIG ** 2) * T + s * _ZGRID)
    integ = float(np.dot(_ZW, np.maximum(S_T - 100.0, 0.0)))
    closed = expected_call_payoff(SPOT, 100.0, R, SIG, T)
    assert abs(integ - closed) < 1e-3, (integ, closed)

    integ_p = float(np.dot(_ZW, np.maximum(100.0 - S_T, 0.0)))
    closed_p = expected_put_payoff(SPOT, 100.0, R, SIG, T)
    assert abs(integ_p - closed_p) < 1e-3, (integ_p, closed_p)


def test_market_mode_ev_is_zero_at_fair_value():
    # Price the call at its own discounted BS expectation, then EV (discounted)
    # must be 0 and undiscounted EV must equal fair*(e^{rT}-1) (~0, small).
    disc = math.exp(-R * T)
    fair = disc * expected_call_payoff(SPOT, 100.0, R, SIG, T)
    st = _long_call(mid=fair)
    res = evaluate_structure(st, SPOT, T, R, SIG)
    ev_share = res["exp_pnl"] / CONTRACT_MULTIPLIER
    assert abs(ev_share - fair * (math.exp(R * T) - 1.0)) < 1e-3
    # Discounted EV ~ 0 (i.e. paying fair value is a zero-edge trade before costs).
    assert abs(disc * (ev_share + fair) - fair) < 1e-3


def test_monte_carlo_agrees_with_integrator():
    st = _long_call()
    res = evaluate_structure(st, SPOT, T, R, SIG)
    mc = evaluate_structure_mc(st, SPOT, T, R, SIG, paths=300_000)
    assert abs(mc["pop_pct"] - res["pop_pct"]) < 1.0
    assert abs(mc["exp_pnl"] - res["exp_pnl"]) < 5.0  # per contract $, MC noise


def test_bull_call_debit_spread_geometry():
    st = Structure(
        "bull_call_debit", "Bull call 100/105",
        [Leg("call", 100.0, +1, 3.10, 0.0, SIG),
         Leg("call", 105.0, -1, 1.20, 0.0, SIG)], commission=0.0)
    res = evaluate_structure(st, SPOT, T, R, SIG)
    width, debit = 5.0, (3.10 - 1.20)
    assert abs(res["max_loss"] - debit * 100) < 1e-6
    assert abs(res["max_gain"] - (width - debit) * 100) < 1e-6
    assert abs(st.breakevens()[0] - (100.0 + debit)) < 1e-2
    assert res["rr"] is not None and res["rr"] > 0
    # POP of a debit spread = P(S_T > breakeven).
    pop_closed = prob_ST_above(SPOT, 100.0 + debit, R, SIG, T) * 100.0
    assert abs(res["pop_pct"] - pop_closed) < 0.1


def test_bull_put_credit_spread_capital_at_risk():
    st = Structure(
        "bull_put_credit", "Bull put 100/95",
        [Leg("put", 100.0, -1, 2.80, 0.0, SIG),
         Leg("put", 95.0, +1, 1.10, 0.0, SIG)], commission=0.0)
    res = evaluate_structure(st, SPOT, T, R, SIG)
    credit = 2.80 - 1.10
    assert res["is_credit"]
    assert abs(res["net_premium"] + credit * 100) < 1e-6  # negative = credit
    assert abs(res["capital_at_risk"] - (5.0 - credit) * 100) < 1e-6
    assert abs(res["max_loss"] - (5.0 - credit) * 100) < 1e-6


def test_costs_reduce_expected_value():
    """Adding bid/ask spread + commission must lower expected P&L (no free mid)."""
    clean = evaluate_structure(_long_call(mid=3.10, comm=0.0), SPOT, T, R, SIG)
    costly = Structure("long_call", "Long 100C",
                       [Leg("call", 100.0, +1, 3.10, 0.40, SIG)], commission=0.65)
    costed = evaluate_structure(costly, SPOT, T, R, SIG)
    assert costed["exp_pnl"] < clean["exp_pnl"]
    assert costed["net_premium"] > clean["net_premium"]  # you pay more than mid


def test_mu_from_target():
    mu = mu_from_target(100.0, 110.0, T)
    assert abs(100.0 * math.exp(mu * T) - 110.0) < 1e-9


def test_scan_ranks_and_carries_both_modes():
    # Minimal synthetic chain (one expiry) - no network.
    def row(K, bid, ask, iv, itm):
        return {"strike": K, "bid": bid, "ask": ask, "last": (bid + ask) / 2,
                "volume": 100, "open_interest": 100, "implied_vol": iv,
                "in_the_money": itm}
    chain = {
        "ticker": "TEST", "expiry": "2030-01-18", "spot": 100.0,
        "calls": [row(95, 6.0, 6.3, 38, True), row(100, 3.0, 3.2, 40, False),
                  row(105, 1.2, 1.4, 42, False), row(110, 0.4, 0.6, 45, False)],
        "puts":  [row(90, 0.4, 0.6, 46, False), row(95, 1.0, 1.2, 43, False),
                  row(100, 2.7, 2.9, 40, False), row(105, 5.6, 5.9, 39, True)],
    }
    out = scan(chain, T=T, r=R, my_sigma=0.55, target_price=112.0, top_n=3)
    assert out["spot"] == 100.0
    assert out["assumptions"]["my_view"]["sigma_source"] == "override"
    assert out["assumptions"]["my_view"]["mu_source"] == "target_price"
    assert len(out["top_by_expected_return"]) <= 3
    assert len(out["top_by_pop"]) <= 3
    assert out["all"], "scan produced structures"
    # Every row carries a MARKET estimate beside the MY-VIEW one.
    for r in out["all"]:
        assert "market" in r
        assert r["mode"] == "my_view"
    # top_by_expected_return is sorted descending.
    rets = [r["exp_return_pct"] for r in out["top_by_expected_return"]]
    assert rets == sorted(rets, reverse=True)


def _demo_chain():
    def row(K, bid, ask, iv, itm):
        return {"strike": K, "bid": bid, "ask": ask, "last": (bid + ask) / 2,
                "volume": 100, "open_interest": 100, "implied_vol": iv,
                "in_the_money": itm}
    return {
        "ticker": "TEST", "expiry": "2030-01-18", "spot": 100.0,
        "calls": [row(95, 6.0, 6.3, 38, True), row(100, 3.0, 3.2, 40, False),
                  row(105, 1.2, 1.4, 42, False), row(110, 0.4, 0.6, 45, False)],
        "puts":  [row(90, 0.4, 0.6, 46, False), row(95, 1.0, 1.2, 43, False),
                  row(100, 2.7, 2.9, 40, False), row(105, 5.6, 5.9, 39, True)],
    }


def test_kelly_present_and_bounded():
    res = evaluate_structure(_long_call(), SPOT, T, R, SIG)
    assert 0.0 <= res["kelly_pct"] <= 100.0
    assert res["win_avg"] >= 0.0 and res["loss_avg"] >= 0.0
    # A bullish view on a cheap call should give a positive Kelly edge.
    bull = evaluate_structure(_long_call(mid=3.10), SPOT, T, mu=0.80, sigma=SIG)
    assert bull["kelly_pct"] > 0.0


def test_singles_only_and_bear_exclusion():
    chain = _demo_chain()
    singles = scan(chain, T=T, r=R, my_sigma=0.5, target_price=112.0,
                   include_verticals=False)
    assert singles["all"]
    assert all(r["kind"] in ("long_call", "long_put") for r in singles["all"])

    no_bear = scan(chain, T=T, r=R, my_sigma=0.5, target_price=112.0,
                   include_verticals=True, include_bear=False)
    assert all(not r["kind"].startswith("bear") for r in no_bear["all"])

    with_bear = scan(chain, T=T, r=R, my_sigma=0.5, target_price=112.0,
                     include_verticals=True, include_bear=True)
    assert any(r["kind"].startswith("bear") for r in with_bear["all"])


def test_top_by_kelly_sorted():
    out = scan(_demo_chain(), T=T, r=R, my_sigma=0.55, target_price=112.0, top_n=5)
    assert out.get("top_by_kelly")
    ks = [r["kelly_pct"] for r in out["top_by_kelly"]]
    assert ks == sorted(ks, reverse=True)
    for r in out["top_by_kelly"]:
        assert r["market"] is None or "kelly_pct" in r["market"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\nAll {passed} pop_scanner tests passed.")


if __name__ == "__main__":
    _run_all()
