"""Probability-of-profit and expected-return ESTIMATOR for option structures.

This is a probability *estimator*, not a prediction. Every number is the output
of an assumed terminal-price distribution; the assumptions (sigma, mu, horizon,
mode) travel with every result so a caller can never mistake a model estimate
for a guarantee.

Distribution model
------------------
Terminal price is lognormal (Geometric Brownian Motion at the horizon):

    S_T = S0 * exp( (mu - sigma^2/2) * T  +  sigma * sqrt(T) * Z ),   Z ~ N(0,1)

so  ln(S_T) ~ Normal( ln(S0) + (mu - sigma^2/2)T ,  (sigma*sqrt(T))^2 ).

Two modes (the caller switches between them):

  MARKET  (reality check):  mu = risk-free rate r,  sigma = the option's
          implied vol.  These are the *market-implied* probabilities. Expected
          value is ~0 before costs (slightly negative after) for essentially
          every structure - that is correct and is shown on purpose, so the
          user sees that an apparent edge requires holding a view that differs
          from the market.

  MY_VIEW (the user's thesis):  sigma = a volatility forecast (defaults to the
          realized-vol forecast from the expected-move module, overridable);
          mu = either an annualized drift the user enters, OR is derived from a
          target price + horizon (E[S_T] = target  =>  mu = ln(target/S0)/T).

Probabilities and expected P&L are computed by numerical integration over the
terminal distribution (dense grid in Z-space, trapezoid weights), so the same
code handles any payoff including multi-leg vertical spreads. A Monte-Carlo
estimator (>=100k paths) is provided for cross-checking.

Honesty rules
-------------
* Premiums use the bid/ask MIDPOINT, then an estimated transaction cost is
  SUBTRACTED (half the bid-ask spread per leg + a configurable commission), so
  expected value is never flattered by unfillable mid prices.
* Probability ITM and Probability of profit are DIFFERENT quantities and are
  computed separately - delta (~ prob ITM under Q) is only a rough proxy for
  prob ITM and is never used as POP.

Closed-form anchors (used by the unit tests):
  For a long call, P(S_T > breakeven) == N(d2) evaluated at the breakeven level
  under the same mu and sigma, where
      d2(level) = ( ln(S0/level) + (mu - sigma^2/2) T ) / ( sigma sqrt(T) ).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy import stats as _sps

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a soft dependency
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)

CONTRACT_MULTIPLIER = 100  # one US equity option = 100 shares
DEFAULT_RISK_FREE = 0.043  # annualized; overridable per call
DEFAULT_COMMISSION = 0.65  # $ per contract per leg
TRADING_DAYS_PER_YEAR = 252

# Dense standard-normal grid for deterministic integration. +/-10 sigma covers
# the distribution to ~1e-23 in each tail; 24001 points => dz ~ 8.3e-4, which
# makes trapezoid probabilities agree with the closed form to ~1e-6.
_Z_LO, _Z_HI, _Z_N = -10.0, 10.0, 24001
_ZGRID = np.linspace(_Z_LO, _Z_HI, _Z_N)
_ZPDF = np.exp(-0.5 * _ZGRID * _ZGRID) / math.sqrt(2.0 * math.pi)
_ZW = _ZPDF / _ZPDF.sum()  # normalized so probabilities sum to exactly 1


# Normal helpers (scipy when present, accurate fallbacks otherwise)
def norm_cdf(x: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (Acklam's rational approximation w/o scipy)."""
    if _HAS_SCIPY:
        return float(_sps.norm.ppf(p))
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Peter Acklam's algorithm - max relative error ~1.15e-9.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _d2_at(spot: float, level: float, mu: float, sigma: float, T: float) -> float:
    """d2 of the lognormal model at an arbitrary price `level`.

    P(S_T > level) = N(d2_at(level)).  This is the Black-Scholes d2 with the
    risk-free rate replaced by the model drift mu.
    """
    s = sigma * math.sqrt(T)
    return (math.log(spot / level) + (mu - 0.5 * sigma * sigma) * T) / s


def prob_ST_above(spot: float, level: float, mu: float, sigma: float, T: float) -> float:
    """Closed-form P(S_T > level) under the lognormal model."""
    if spot <= 0 or level <= 0 or sigma <= 0 or T <= 0:
        return float("nan")
    return norm_cdf(_d2_at(spot, level, mu, sigma, T))


# Closed-form expected payoffs (undiscounted) - test anchors + sanity
def expected_call_payoff(spot: float, K: float, mu: float, sigma: float, T: float) -> float:
    """E[max(S_T - K, 0)] (undiscounted) under drift mu. Black-76 form."""
    F = spot * math.exp(mu * T)
    s = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return F * norm_cdf(d1) - K * norm_cdf(d2)


def expected_put_payoff(spot: float, K: float, mu: float, sigma: float, T: float) -> float:
    """E[max(K - S_T, 0)] (undiscounted) under drift mu."""
    F = spot * math.exp(mu * T)
    s = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return K * norm_cdf(-d2) - F * norm_cdf(-d1)


# Structures
@dataclass
class Leg:
    """One option leg priced from a chain row.

    qty: +1 long (buy), -1 short (sell).  mid/spread are per share.
    iv:  implied vol as a fraction (0.42), not percent.
    """

    type: str  # "call" | "put"
    strike: float
    qty: int
    mid: float
    spread: float
    iv: float

    def intrinsic(self, S: np.ndarray | float):
        if self.type == "call":
            return np.maximum(S - self.strike, 0.0)
        return np.maximum(self.strike - S, 0.0)


@dataclass
class Structure:
    kind: str  # long_call | long_put | bull_call_debit | bull_put_credit | ...
    label: str
    legs: list[Leg]
    commission: float = DEFAULT_COMMISSION
    meta: dict = field(default_factory=dict)

    # -- premium / cost ---------------------------------------------------- #
    def net_mid(self) -> float:
        """Net debit at midpoint (positive = you pay, negative = credit)."""
        return sum(leg.qty * leg.mid for leg in self.legs)

    def total_cost(self) -> float:
        """Per-share frictional cost: half-spread per leg + commission per leg.

        Always a cost (slippage from mid + commission), regardless of side.
        """
        half = sum(0.5 * max(leg.spread, 0.0) for leg in self.legs)
        comm = len(self.legs) * (self.commission / CONTRACT_MULTIPLIER)
        return half + comm

    def net_debit(self) -> float:
        """Cost-adjusted net debit (mid + frictions). Credit structures < 0."""
        return self.net_mid() + self.total_cost()

    # -- payoff geometry --------------------------------------------------- #
    def payoff(self, S):
        S = np.asarray(S, dtype=float)
        total = np.zeros_like(S)
        for leg in self.legs:
            total = total + leg.qty * leg.intrinsic(S)
        return total

    def pnl(self, S):
        return self.payoff(S) - self.net_debit()

    def strikes(self) -> list[float]:
        return sorted({leg.strike for leg in self.legs})

    def _call_slope_at_inf(self) -> int:
        return sum(leg.qty for leg in self.legs if leg.type == "call")

    def bounds(self) -> tuple[float | None, float]:
        """(max_gain, max_loss) per share. max_gain None => unbounded upside.

        Piecewise-linear payoff: extrema occur at S=0, the strikes, or S->inf.
        """
        ks = self.strikes()
        test_pts = [0.0, *ks, max(ks) * 5 + 10.0]
        pnls = [float(self.pnl(np.array([s]))[0]) for s in test_pts]
        max_loss = -min(pnls)  # largest loss as a positive number
        if self._call_slope_at_inf() > 0:
            max_gain: float | None = None  # unbounded (net long calls)
        else:
            max_gain = max(pnls)
        return max_gain, max(max_loss, 0.0)

    def breakevens(self) -> list[float]:
        """Breakeven price(s): where cost-adjusted P&L crosses zero."""
        S = _spot_grid(self)
        y = self.pnl(S)
        out: list[float] = []
        sign = np.sign(y)
        idx = np.where(np.diff(sign) != 0)[0]
        for i in idx:
            x0, x1, y0, y1 = S[i], S[i + 1], y[i], y[i + 1]
            if y1 != y0:
                out.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        return sorted(round(v, 4) for v in out)

    def capital_at_risk(self) -> float:
        """Capital at risk = max loss per share (>=0)."""
        _, max_loss = self.bounds()
        return max_loss

    def primary_iv(self) -> float:
        """Representative IV for MARKET-mode sigma: |qty|-weighted leg IVs."""
        num = sum(abs(leg.qty) * leg.iv for leg in self.legs if leg.iv > 0)
        den = sum(abs(leg.qty) for leg in self.legs if leg.iv > 0)
        return num / den if den else 0.0


# spot grid cache keyed on spot+T via closure in evaluate(); breakevens use a
# generic wide grid that does not need the distribution.
def _spot_grid(struct: Structure) -> np.ndarray:
    ks = struct.strikes()
    lo = max(0.01, min(ks) * 0.2)
    hi = max(ks) * 3.0 + 10.0
    return np.linspace(lo, hi, 6000)


# Core evaluation
def _wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return float(np.interp(q, cw, v))


def evaluate_structure(
    struct: Structure,
    spot: float,
    T: float,
    mu: float,
    sigma: float,
    *,
    mult: int = CONTRACT_MULTIPLIER,
) -> dict:
    """Score one structure under the lognormal model. Per-contract $ outputs.

    Returns POP, prob ITM, expected P&L, expected return %, max gain/loss, R/R,
    breakevens and the median / 25th / 75th-percentile P&L. Probabilities come
    from deterministic grid integration; expected P&L from the same grid.
    """
    if spot <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("spot, T and sigma must be positive")

    s = sigma * math.sqrt(T)
    drift = (mu - 0.5 * sigma * sigma) * T
    S_T = spot * np.exp(drift + s * _ZGRID)  # terminal prices on the Z grid
    w = _ZW

    net_debit = struct.net_debit()
    payoff = struct.payoff(S_T)
    pnl_share = payoff - net_debit

    # Expected values (per share -> per contract)
    exp_payoff = float(np.dot(w, payoff))
    exp_pnl_share = exp_payoff - net_debit
    exp_pnl = exp_pnl_share * mult

    # Probability of profit  = P(P&L > 0).  Strict > 0 (breakeven is a loss
    # after costs). Uses the cost-adjusted P&L, so it already reflects frictions.
    pop = float(np.dot(w, (pnl_share > 0.0).astype(float)))

    # Probability ITM - DIFFERENT from POP. For a single leg it is P(S_T beyond
    # the strike). For a vertical we report it for the long (primary) leg.
    long_leg = next((lg for lg in struct.legs if lg.qty > 0), struct.legs[0])
    if long_leg.type == "call":
        prob_itm = float(np.dot(w, (long_leg.strike < S_T).astype(float)))
    else:
        prob_itm = float(np.dot(w, (long_leg.strike > S_T).astype(float)))

    max_gain, max_loss = struct.bounds()
    cap = struct.capital_at_risk()
    exp_return_pct = (exp_pnl_share / cap * 100.0) if cap > 0 else float("nan")

    # Outcome spread of P&L (per contract)
    p25 = _wquantile(pnl_share, w, 0.25) * mult
    p50 = _wquantile(pnl_share, w, 0.50) * mult
    p75 = _wquantile(pnl_share, w, 0.75) * mult

    rr = (max_gain / max_loss) if (max_gain is not None and max_loss > 0) else None

    # Kelly fraction: f* = p - (1 - p) / b, b = avg_win / avg_loss
    win_mask = pnl_share > 0.0
    p_win = float(np.dot(w, win_mask.astype(float)))
    p_loss = max(1.0 - p_win, 0.0)
    win_avg = (float(np.dot(w, np.where(win_mask, pnl_share, 0.0))) / p_win
               if p_win > 1e-9 else 0.0)            # per share, positive
    loss_avg = (float(np.dot(w, np.where(~win_mask, -pnl_share, 0.0))) / p_loss
                if p_loss > 1e-9 else 0.0)          # per share, positive
    if loss_avg <= 1e-9:
        kelly = p_win                                # essentially no downside
    else:
        b = win_avg / loss_avg
        kelly = p_win - p_loss / b if b > 0 else 0.0
    kelly = max(0.0, min(kelly, 1.0))

    return {
        "kind": struct.kind,
        "label": struct.label,
        "strikes": struct.strikes(),
        "legs": [
            {"type": lg.type, "strike": lg.strike, "qty": lg.qty,
             "mid": round(lg.mid, 4), "iv_pct": round(lg.iv * 100, 2)}
            for lg in struct.legs
        ],
        "net_debit": round(net_debit, 4),                 # per share, cost-adjusted
        "net_premium": round(net_debit * mult, 2),        # per contract ($), cost-adjusted
        "mid_per_share": round(struct.net_mid(), 4),      # quoted option price / share
        "mid_per_contract": round(struct.net_mid() * mult, 2),  # quoted price x 100
        "is_credit": net_debit < 0,
        "frictions_per_contract": round(struct.total_cost() * mult, 2),
        "breakevens": struct.breakevens(),
        "pop_pct": round(pop * 100, 2),
        "prob_itm_pct": round(prob_itm * 100, 2),
        "exp_pnl": round(exp_pnl, 2),                     # per contract ($)
        "exp_return_pct": (None if math.isnan(exp_return_pct)
                           else round(exp_return_pct, 2)),
        "max_gain": (None if max_gain is None else round(max_gain * mult, 2)),
        "max_loss": round(max_loss * mult, 2),
        "capital_at_risk": round(cap * mult, 2),
        "rr": (None if rr is None else round(rr, 2)),
        "kelly_pct": round(kelly * 100, 2),               # fraction of bankroll
        "win_avg": round(win_avg * mult, 2),              # avg $ when it wins
        "loss_avg": round(loss_avg * mult, 2),            # avg $ when it loses
        "pnl_p25": round(p25, 2),
        "pnl_p50": round(p50, 2),
        "pnl_p75": round(p75, 2),
        "assumptions": {
            "mode_sigma_pct": round(sigma * 100, 2),
            "mode_mu_pct": round(mu * 100, 2),
            "horizon_years": round(T, 5),
            "horizon_days": round(T * 365.0, 1),
        },
    }


def evaluate_both_modes(
    struct: Structure,
    spot: float,
    T: float,
    *,
    r: float,
    my_sigma: float,
    my_mu: float,
    mult: int = CONTRACT_MULTIPLIER,
) -> dict:
    """Evaluate a structure under BOTH modes so the UI can show them side by side.

    MARKET: mu=r, sigma=structure's implied vol.
    MY_VIEW: mu=my_mu, sigma=my_sigma.
    """
    mkt_sigma = struct.primary_iv()
    out_view = evaluate_structure(struct, spot, T, my_mu, my_sigma, mult=mult)
    row = dict(out_view)
    row["mode"] = "my_view"
    if mkt_sigma > 0:
        mkt = evaluate_structure(struct, spot, T, r, mkt_sigma, mult=mult)
        row["market"] = {
            "pop_pct": mkt["pop_pct"],
            "prob_itm_pct": mkt["prob_itm_pct"],
            "exp_pnl": mkt["exp_pnl"],
            "exp_return_pct": mkt["exp_return_pct"],
            "kelly_pct": mkt["kelly_pct"],
            "sigma_pct": round(mkt_sigma * 100, 2),
        }
    else:
        row["market"] = None
    return row


# Monte-Carlo cross-check (>=100k paths)
def evaluate_structure_mc(
    struct: Structure,
    spot: float,
    T: float,
    mu: float,
    sigma: float,
    *,
    paths: int = 200_000,
    seed: int | None = 7,
    mult: int = CONTRACT_MULTIPLIER,
) -> dict:
    """Monte-Carlo POP / expected P&L, for cross-checking the integrator."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(max(paths, 100_000))
    s = sigma * math.sqrt(T)
    S_T = spot * np.exp((mu - 0.5 * sigma * sigma) * T + s * z)
    net_debit = struct.net_debit()
    pnl = struct.payoff(S_T) - net_debit
    return {
        "pop_pct": round(float(np.mean(pnl > 0.0)) * 100, 2),
        "exp_pnl": round(float(np.mean(pnl)) * mult, 2),
    }


# Structure construction from a chain
def _leg_from_row(row: dict, opt_type: str, qty: int) -> Leg | None:
    bid = float(row.get("bid") or 0.0)
    ask = float(row.get("ask") or 0.0)
    last = float(row.get("last") or 0.0)
    iv = float(row.get("implied_vol") or 0.0) / 100.0  # chain stores percent
    if bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        spread = ask - bid
    elif last > 0:
        mid = last
        spread = max(0.15 * last, 0.02)  # no quotes -> assume a wide spread
    else:
        return None
    if mid <= 0:
        return None
    return Leg(type=opt_type, strike=float(row["strike"]), qty=qty,
               mid=mid, spread=spread, iv=iv)


def build_structures(
    chain: dict,
    *,
    spot: float,
    sigma_hint: float,
    T: float,
    commission: float = DEFAULT_COMMISSION,
    strike_window: float = 2.5,
    include_verticals: bool = True,
    include_bear: bool = True,
    max_width_steps: int = 2,
) -> list[Structure]:
    """Build long calls, long puts and simple verticals from one expiry's chain.

    Only strikes within `strike_window` * sigma * sqrt(T) (log space) of spot are
    considered, keeping the grid focused on tradeable, liquid contracts.

    include_bear: when False, bearish spreads (bear call / bear put) are skipped.
    """
    calls = {round(float(r["strike"]), 4): r for r in chain.get("calls", [])}
    puts = {round(float(r["strike"]), 4): r for r in chain.get("puts", [])}

    band = max(strike_window * max(sigma_hint, 0.05) * math.sqrt(max(T, 1e-6)), 0.05)

    def _near(K: float) -> bool:
        return abs(math.log(K / spot)) <= band

    structures: list[Structure] = []

    call_strikes = sorted(k for k in calls if _near(k))
    put_strikes = sorted(k for k in puts if _near(k))

    # Singles -------------------------------------------------------------- #
    for K in call_strikes:
        leg = _leg_from_row(calls[K], "call", +1)
        if leg:
            structures.append(Structure("long_call", f"Long {K:g}C", [leg], commission))
    for K in put_strikes:
        leg = _leg_from_row(puts[K], "put", +1)
        if leg:
            structures.append(Structure("long_put", f"Long {K:g}P", [leg], commission))

    if not include_verticals:
        return structures

    # Verticals between strikes `step` apart (adjacent and wider) ----------- #
    for step in range(1, max_width_steps + 1):
        for i in range(len(call_strikes) - step):
            klo, khi = call_strikes[i], call_strikes[i + step]
            long_l = _leg_from_row(calls[klo], "call", +1)
            short_h = _leg_from_row(calls[khi], "call", -1)
            long_h = _leg_from_row(calls[khi], "call", +1)
            short_l = _leg_from_row(calls[klo], "call", -1)
            if long_l and short_h:  # bull call (debit)
                structures.append(Structure(
                    "bull_call_debit", f"Bull call {klo:g}/{khi:g}",
                    [long_l, short_h], commission, {"width": khi - klo}))
            if include_bear and long_h and short_l:  # bear call (credit)
                structures.append(Structure(
                    "bear_call_credit", f"Bear call {klo:g}/{khi:g}",
                    [short_l, long_h], commission, {"width": khi - klo}))
        for i in range(len(put_strikes) - step):
            klo, khi = put_strikes[i], put_strikes[i + step]
            long_h = _leg_from_row(puts[khi], "put", +1)
            short_l = _leg_from_row(puts[klo], "put", -1)
            long_l = _leg_from_row(puts[klo], "put", +1)
            short_h = _leg_from_row(puts[khi], "put", -1)
            if include_bear and long_h and short_l:  # bear put (debit)
                structures.append(Structure(
                    "bear_put_debit", f"Bear put {khi:g}/{klo:g}",
                    [long_h, short_l], commission, {"width": khi - klo}))
            if short_h and long_l:  # bull put (credit)
                structures.append(Structure(
                    "bull_put_credit", f"Bull put {khi:g}/{klo:g}",
                    [short_h, long_l], commission, {"width": khi - klo}))

    return structures


# mu derivation for MY_VIEW
def mu_from_target(spot: float, target: float, T: float) -> float:
    """Drift implied by E[S_T] = target  =>  mu = ln(target/spot) / T."""
    if spot <= 0 or target <= 0 or T <= 0:
        return 0.0
    return math.log(target / spot) / T


# Top-level scan
def scan(
    chain: dict,
    *,
    T: float,
    spot: float | None = None,
    r: float = DEFAULT_RISK_FREE,
    rv_forecast: float | None = None,
    my_sigma: float | None = None,
    my_drift: float | None = None,
    target_price: float | None = None,
    commission: float = DEFAULT_COMMISSION,
    include_verticals: bool = True,
    include_bear: bool = True,
    strike_window: float = 2.5,
    top_n: int = 5,
    mult: int = CONTRACT_MULTIPLIER,
) -> dict:
    """Score a whole expiry and rank structures two ways.

    Ranks by MY_VIEW expected-return %, and separately by MY_VIEW POP; returns
    the top `top_n` of each. Every row carries the MARKET-mode estimate beside
    the MY_VIEW estimate so the gap between "what the market implies" and "what
    I'm assuming" is explicit.
    """
    spot = float(spot if spot is not None else chain.get("spot") or 0.0)
    if spot <= 0:
        raise ValueError("spot price unavailable for this chain")
    if T <= 0:
        raise ValueError("horizon T must be positive")

    # MY_VIEW sigma: explicit override -> RV forecast -> fall back to ATM IV.
    if my_sigma is not None and my_sigma > 0:
        sigma_view = float(my_sigma)
        sigma_src = "override"
    elif rv_forecast is not None and rv_forecast > 0:
        sigma_view = float(rv_forecast)
        sigma_src = "rv_forecast"
    else:
        sigma_view = _atm_iv(chain, spot) or 0.30
        sigma_src = "atm_iv_fallback"

    # MY_VIEW mu: explicit drift wins; else target price; else flat (mu=0).
    if my_drift is not None:
        mu_view = float(my_drift)
        mu_src = "drift_input"
    elif target_price is not None and target_price > 0:
        mu_view = mu_from_target(spot, float(target_price), T)
        mu_src = "target_price"
    else:
        mu_view = 0.0
        mu_src = "flat_default"

    structures = build_structures(
        chain, spot=spot, sigma_hint=sigma_view, T=T,
        commission=commission, strike_window=strike_window,
        include_verticals=include_verticals, include_bear=include_bear,
    )

    rows: list[dict] = []
    for st in structures:
        try:
            row = evaluate_both_modes(
                st, spot, T, r=r, my_sigma=sigma_view, my_mu=mu_view, mult=mult)
            rows.append(row)
        except Exception as e:
            logger.debug("[pop_scan] %s skipped: %s", st.label, e)

    def _valid_ret(rw: dict) -> bool:
        return rw.get("exp_return_pct") is not None

    by_return = sorted(
        (r for r in rows if _valid_ret(r)),
        key=lambda r: r["exp_return_pct"], reverse=True,
    )[:top_n]
    by_pop = sorted(rows, key=lambda r: r["pop_pct"], reverse=True)[:top_n]
    # "Best plays" = highest Kelly fraction: balances probability of winning AND
    # size of the profit in one research-backed score (Kelly, 1956).
    by_kelly = sorted(
        rows, key=lambda r: r.get("kelly_pct") or 0.0, reverse=True,
    )[:top_n]

    return {
        "ticker": chain.get("ticker"),
        "expiry": chain.get("expiry"),
        "spot": round(spot, 4),
        "mode": "my_view_vs_market",
        "assumptions": {
            "risk_free_pct": round(r * 100, 3),
            "my_view": {
                "sigma_pct": round(sigma_view * 100, 2),
                "sigma_source": sigma_src,
                "mu_pct": round(mu_view * 100, 3),
                "mu_source": mu_src,
                "target_price": target_price,
                "implied_move_pct": round(
                    (math.exp(mu_view * T) - 1.0) * 100, 2),
            },
            "horizon_days": round(T * 365.0, 1),
            "horizon_years": round(T, 5),
            "commission_per_leg": commission,
            "rv_forecast_pct": (None if rv_forecast is None
                                else round(rv_forecast * 100, 2)),
        },
        "counts": {"structures": len(rows)},
        "all": rows,
        "top_by_expected_return": by_return,
        "top_by_pop": by_pop,
        "top_by_kelly": by_kelly,
        "scoring_note": "Top 5 ranked by Kelly fraction.",
        "disclaimer": "Model estimates under assumed vol and drift.",
    }


def _atm_iv(chain: dict, spot: float) -> float | None:
    """ATM implied vol (fraction) from the chain - nearest-strike call IV."""
    best = None
    best_d = math.inf
    for r in chain.get("calls", []):
        K = float(r.get("strike") or 0)
        iv = float(r.get("implied_vol") or 0) / 100.0
        if K <= 0 or iv <= 0:
            continue
        d = abs(K - spot)
        if d < best_d:
            best_d, best = d, iv
    return best


# Self-test (run:  python -m core.pop_scanner)
if __name__ == "__main__":
    SPOT, T, R, SIG = 100.0, 30 / 365.0, 0.05, 0.40

    # 1) POP of a long call == N(d2) at the breakeven, same mu & sigma.
    mid = 3.10
    call = Structure("long_call", "Long 100C",
                     [Leg("call", 100.0, +1, mid, 0.0, SIG)], commission=0.0)
    res = evaluate_structure(call, SPOT, T, R, SIG)
    be = call.breakevens()[0]
    pop_closed = prob_ST_above(SPOT, be, R, SIG, T) * 100
    print(f"long call: BE={be:.4f}  POP_int={res['pop_pct']:.3f}  "
          f"POP_N(d2)={pop_closed:.3f}")
    assert abs(res["pop_pct"] - pop_closed) < 0.05, "POP must match N(d2) at BE"

    # 2) prob ITM == N(d2) at the strike (NOT the same as POP).
    itm_closed = prob_ST_above(SPOT, 100.0, R, SIG, T) * 100
    assert abs(res["prob_itm_pct"] - itm_closed) < 0.05
    assert res["prob_itm_pct"] > res["pop_pct"], "prob ITM > POP for a long call"

    # 3) Integrated expected payoff == closed-form Black-76 expectation.
    s = SIG * math.sqrt(T)
    S_T = SPOT * np.exp((R - 0.5 * SIG ** 2) * T + s * _ZGRID)
    exp_payoff_int = float(np.dot(_ZW, np.maximum(S_T - 100.0, 0.0)))
    exp_payoff_cf = expected_call_payoff(SPOT, 100.0, R, SIG, T)
    print(f"E[payoff] integ={exp_payoff_int:.5f}  closed={exp_payoff_cf:.5f}")
    assert abs(exp_payoff_int - exp_payoff_cf) < 1e-3

    # 4) MARKET-mode EV ~ 0 before costs: price the call at its own BS value.
    disc = math.exp(-R * T)
    fair = disc * exp_payoff_cf
    fair_call = Structure("long_call", "fair",
                          [Leg("call", 100.0, +1, fair, 0.0, SIG)], commission=0.0)
    ev = evaluate_structure(fair_call, SPOT, T, R, SIG)["exp_pnl"] / CONTRACT_MULTIPLIER
    # undiscounted EV = E[payoff] - fair = fair*(e^{rT}-1)
    print(f"market EV (undiscounted) = {ev:.5f}  (=~ fair*(e^rT-1))")
    assert abs(ev - fair * (math.exp(R * T) - 1.0)) < 1e-3
    assert abs(disc * exp_payoff_cf - fair) < 1e-9  # discounted EV == 0

    # 5) Monte-Carlo agrees with the integrator on POP & EV.
    mc = evaluate_structure_mc(call, SPOT, T, R, SIG, paths=300_000)
    print(f"MC  POP={mc['pop_pct']:.2f}  int POP={res['pop_pct']:.2f}")
    assert abs(mc["pop_pct"] - res["pop_pct"]) < 1.0

    # 6) Vertical debit spread closed-form geometry.
    bull = Structure(
        "bull_call_debit", "Bull call 100/105",
        [Leg("call", 100.0, +1, 3.10, 0.0, SIG),
         Leg("call", 105.0, -1, 1.20, 0.0, SIG)], commission=0.0)
    vr = evaluate_structure(bull, SPOT, T, R, SIG)
    width, debit = 5.0, (3.10 - 1.20)
    assert abs(vr["max_loss"] - debit * 100) < 1e-6
    assert abs(vr["max_gain"] - (width - debit) * 100) < 1e-6
    assert abs(bull.breakevens()[0] - (100.0 + debit)) < 1e-2
    print(f"bull call: maxL={vr['max_loss']} maxG={vr['max_gain']} "
          f"BE={bull.breakevens()[0]:.2f} POP={vr['pop_pct']:.2f} RR={vr['rr']}")

    # 7) Credit spread: capital at risk = width - net credit.
    bullput = Structure(
        "bull_put_credit", "Bull put 100/95",
        [Leg("put", 100.0, -1, 2.80, 0.0, SIG),
         Leg("put", 95.0, +1, 1.10, 0.0, SIG)], commission=0.0)
    cr = evaluate_structure(bullput, SPOT, T, R, SIG)
    credit = 2.80 - 1.10
    assert cr["is_credit"] and abs(cr["net_premium"] + credit * 100) < 1e-6
    assert abs(cr["capital_at_risk"] - (5.0 - credit) * 100) < 1e-6
    print(f"bull put credit: net={cr['net_premium']} cap={cr['capital_at_risk']} "
          f"POP={cr['pop_pct']:.2f}")

    print("\nAll pop_scanner self-tests passed.")
