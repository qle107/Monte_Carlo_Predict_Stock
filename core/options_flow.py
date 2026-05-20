"""
core/options_flow.py — Options flow: Max Pain + Gamma Exposure (GEX).

Fetches the options chain (yfinance, free) and computes:

  max_pain      : Strike where total option-buyer loss is maximised.
                  Price gravitates here into expiry (market-maker incentive).

  net_gex       : Total dealer net gamma exposure across all strikes.
                  Positive  → dealers are long gamma → they BUY dips, SELL rips
                              → volatility is DAMPED, price tends to pin.
                  Negative  → dealers are short gamma → they SELL dips, BUY rips
                              → volatility is AMPLIFIED, moves accelerate.

  gamma_flip    : The price level where net GEX crosses zero.
                  Above gamma flip  →  volatility suppression (positive GEX).
                  Below gamma flip  →  volatility amplification (negative GEX).

  call_wall     : Strike with highest call OI (strong resistance ceiling).
  put_wall      : Strike with highest put OI (strong support floor).

  gex_profile   : Per-strike GEX bars for charting.

Why this helps predict price reaction at demand/supply zones
------------------------------------------------------------
  • Zone at / above call_wall → bounce is harder; expect compression or failure.
  • Zone at / below put_wall  → bounce is stronger; dealers must buy there too.
  • Zone between gamma_flip and call_wall (positive GEX band) → pinning / chop.
  • Zone below gamma_flip      → directional moves; zone may not hold.
  • Max pain acts as a gravity attractor going into weekly/monthly expiry.

Black-Scholes gamma (closed-form):
  d1    = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)
  Gamma = N'(d1) / (S · σ · √T)
  GEX_k = OI_calls(k)·Γ(k)·S²·100 - OI_puts(k)·Γ(k)·S²·100
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─── Cache (avoid hammering yfinance) ────────────────────────────────────────
_cache_lock = threading.RLock()
_cache: dict = {}  # key → (result, expire_time)
_CACHE_TTL = 300.0  # 5 minutes


def _cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.monotonic() < entry[1]:
            return entry[0]
    return None


def _cache_put(key, value):
    with _cache_lock:
        _cache[key] = (value, time.monotonic() + _CACHE_TTL)


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class GEXBar:
    strike: float
    gex: float  # net GEX at this strike (calls positive, puts negative)
    call_oi: int
    put_oi: int


@dataclass
class OptionsFlow:
    ticker: str
    expiry: str  # expiry date used (YYYY-MM-DD)
    spot: float
    max_pain: float
    call_wall: float  # strike with peak call OI
    put_wall: float  # strike with peak put OI
    gamma_flip: float  # price where net GEX ≈ 0
    net_gex: float  # total dealer net GEX
    gex_positive: bool  # True = volatility-damping regime
    days_to_expiry: float
    gex_profile: list[GEXBar]  # sorted by strike
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "expiry": self.expiry,
            "spot": round(self.spot, 4),
            "max_pain": round(self.max_pain, 4),
            "call_wall": round(self.call_wall, 4),
            "put_wall": round(self.put_wall, 4),
            "gamma_flip": round(self.gamma_flip, 4),
            "net_gex": round(self.net_gex, 2),
            "gex_positive": self.gex_positive,
            "days_to_expiry": round(self.days_to_expiry, 1),
            "error": self.error,
            "gex_profile": [
                {
                    "strike": round(b.strike, 4),
                    "gex": round(b.gex, 2),
                    "call_oi": b.call_oi,
                    "put_oi": b.put_oi,
                }
                for b in self.gex_profile
            ],
        }


# ─── Black-Scholes helpers ────────────────────────────────────────────────────


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """
    Black-Scholes gamma.  Returns 0 on bad inputs.
    S = spot, K = strike, T = time to expiry in years, sigma = implied vol.
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0


# ─── Max Pain ─────────────────────────────────────────────────────────────────


def _compute_max_pain(calls_df, puts_df) -> float:
    """
    Max pain = strike that minimises total payout to option holders.

    Total payout at expiry price P:
      Σ_K [OI_call(K) · max(P−K, 0) + OI_put(K) · max(K−P, 0)] × 100

    We evaluate over all strikes in the chain.
    """
    all_strikes = sorted(set(list(calls_df["strike"].values) + list(puts_df["strike"].values)))
    if not all_strikes:
        return 0.0

    call_oi = {float(r["strike"]): int(r["openInterest"] or 0) for _, r in calls_df.iterrows()}
    put_oi = {float(r["strike"]): int(r["openInterest"] or 0) for _, r in puts_df.iterrows()}

    min_pain = float("inf")
    max_pain_strike = all_strikes[len(all_strikes) // 2]

    for P in all_strikes:
        total = 0.0
        for K in all_strikes:
            total += call_oi.get(K, 0) * max(P - K, 0.0)
            total += put_oi.get(K, 0) * max(K - P, 0.0)
        if total < min_pain:
            min_pain = total
            max_pain_strike = P

    return float(max_pain_strike)


# ─── GEX profile ─────────────────────────────────────────────────────────────


def _compute_gex_profile(
    calls_df,
    puts_df,
    spot: float,
    T: float,
    risk_free: float = 0.05,
    max_bars: int = 50,
) -> list[GEXBar]:
    """
    For each strike compute net GEX:
      GEX_net(K) = (OI_call(K) - OI_put(K)) · Γ(S, K, T, σ) · S² · 100

    We use each option's own implied vol (IV) from the chain.
    Falls back to average IV if individual IV is missing.

    Limits output to max_bars bars centered around spot price
    to reduce response size and keep chart legible.
    """
    import pandas as pd

    # Merge calls + puts on strike
    cols = ["strike", "openInterest", "impliedVolatility"]
    c = calls_df[cols].copy().rename(columns={"openInterest": "call_oi", "impliedVolatility": "call_iv"})
    p = puts_df[cols].copy().rename(columns={"openInterest": "put_oi", "impliedVolatility": "put_iv"})

    merged = pd.merge(c, p, on="strike", how="outer").fillna(0)
    merged["call_oi"] = merged["call_oi"].astype(int)
    merged["put_oi"] = merged["put_oi"].astype(int)
    merged["call_iv"] = merged["call_iv"].astype(float)
    merged["put_iv"] = merged["put_iv"].astype(float)

    # Fallback average IV
    avg_iv = float(merged[["call_iv", "put_iv"]].replace(0, float("nan")).stack().mean()) or 0.3

    bars: list[GEXBar] = []
    for _, row in merged.iterrows():
        K = float(row["strike"])
        c_iv = float(row["call_iv"]) or avg_iv
        p_iv = float(row["put_iv"]) or avg_iv
        c_oi = int(row["call_oi"])
        p_oi = int(row["put_oi"])

        g_call = _bs_gamma(spot, K, T, c_iv, risk_free)
        g_put = _bs_gamma(spot, K, T, p_iv, risk_free)

        # Dealers who sold calls are long delta → they are short gamma (negative GEX from calls)
        # Convention: call GEX positive (dealers short vol), put GEX negative
        gex = (c_oi * g_call - p_oi * g_put) * spot * spot * 100.0

        bars.append(GEXBar(strike=K, gex=gex, call_oi=c_oi, put_oi=p_oi))

    bars.sort(key=lambda b: b.strike)

    # ── Limit to most relevant strikes (centered around spot) ──────────────────
    if len(bars) > max_bars:
        # Find spot index
        spot_idx = min(range(len(bars)), key=lambda i: abs(bars[i].strike - spot))
        # Calculate window around spot
        window_half = max_bars // 2
        start = max(0, spot_idx - window_half)
        end = min(len(bars), start + max_bars)
        # Adjust start if we're near the end
        if end - start < max_bars:
            start = max(0, end - max_bars)
        bars = bars[start:end]

    return bars


def _find_gamma_flip(bars: list[GEXBar], spot: float) -> float:
    """
    Find the strike closest to where cumulative net GEX crosses zero
    (scanning from spot downward).
    """
    # Only consider strikes ≤ spot (below spot = where gamma flip matters most)
    below = [(b.strike, b.gex) for b in bars if b.strike <= spot * 1.05]
    if not below:
        return spot

    # Cumulative GEX from spot downward
    below.sort(key=lambda x: -x[0])  # high to low
    cum = 0.0
    prev_strike = spot
    for strike, gex in below:
        prev_cum = cum
        cum += gex
        if prev_cum > 0 and cum <= 0:
            # crossed zero between prev_strike and strike — interpolate
            frac = abs(prev_cum) / (abs(prev_cum) + abs(cum) + 1e-12)
            return prev_strike - frac * (prev_strike - strike)
        prev_strike = strike
    return below[-1][0]  # no flip found — return lowest strike


# ─── Public entry point ───────────────────────────────────────────────────────


def fetch_options_flow(ticker: str, spot: float | None = None) -> OptionsFlow:
    """
    Fetch options chain and compute GEX + Max Pain.
    Returns an OptionsFlow with error set if anything fails.
    Cached for 5 minutes.
    """
    key = ticker.upper()
    cached = _cache_get(key)
    if cached is not None:
        return cached

    result = _fetch(key, spot)
    _cache_put(key, result)
    return result


def _fetch(ticker: str, spot_override: float | None) -> OptionsFlow:
    _empty = OptionsFlow(
        ticker=ticker,
        expiry="",
        spot=spot_override or 0.0,
        max_pain=0.0,
        call_wall=0.0,
        put_wall=0.0,
        gamma_flip=0.0,
        net_gex=0.0,
        gex_positive=True,
        days_to_expiry=0.0,
        gex_profile=[],
        error="not_fetched",
    )
    try:
        from datetime import datetime, timezone

        import yfinance as yf

        t = yf.Ticker(ticker)

        # Get spot price
        if spot_override and spot_override > 0:
            spot = float(spot_override)
        else:
            info = t.fast_info
            spot = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None) or 0.0
            if spot <= 0:
                hist = t.history(period="1d", interval="1m")
                spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if spot <= 0:
            _empty.error = "no_spot_price"
            return _empty

        # Pick nearest expiry ≥ 7 days out (avoid pin risk noise on 0DTE)
        exps = t.options
        if not exps:
            _empty.error = "no_options_chain"
            return _empty

        now = datetime.now(timezone.utc).date()
        chosen_exp = None
        for exp in exps:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (d - now).days
            if dte >= 5:
                chosen_exp = exp
                break
        if chosen_exp is None:
            chosen_exp = exps[0]

        exp_date = datetime.strptime(chosen_exp, "%Y-%m-%d").date()
        T = max((exp_date - now).days, 1) / 365.0

        chain = t.option_chain(chosen_exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        if calls.empty or puts.empty:
            _empty.error = "empty_chain"
            return _empty

        # Focus on strikes within ±20% of spot (filter out deep OTM noise)
        lo, hi = spot * 0.80, spot * 1.20
        calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]

        if calls.empty or puts.empty:
            _empty.error = "no_near_strikes"
            return _empty

        # ── Max Pain ──────────────────────────────────────────────────────
        max_pain = _compute_max_pain(calls, puts)

        # ── Call Wall / Put Wall ──────────────────────────────────────────
        c_oi_max_idx = calls["openInterest"].fillna(0).astype(int).idxmax()
        p_oi_max_idx = puts["openInterest"].fillna(0).astype(int).idxmax()
        call_wall = float(calls.loc[c_oi_max_idx, "strike"])
        put_wall = float(puts.loc[p_oi_max_idx, "strike"])

        # ── GEX profile ───────────────────────────────────────────────────
        gex_profile = _compute_gex_profile(calls, puts, spot, T)
        net_gex = sum(b.gex for b in gex_profile)
        gamma_flip = _find_gamma_flip(gex_profile, spot)

        result = OptionsFlow(
            ticker=ticker,
            expiry=chosen_exp,
            spot=spot,
            max_pain=max_pain,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_flip=gamma_flip,
            net_gex=net_gex,
            gex_positive=(net_gex >= 0),
            days_to_expiry=T * 365,
            gex_profile=gex_profile,
            error=None,
        )
        logger.info(
            "[OptionsFlow] %s  spot=%.2f  max_pain=%.2f  call_wall=%.2f  "
            "put_wall=%.2f  gamma_flip=%.2f  net_gex=%.0f  exp=%s",
            ticker,
            spot,
            max_pain,
            call_wall,
            put_wall,
            gamma_flip,
            net_gex,
            chosen_exp,
        )
        return result

    except ImportError:
        _empty.error = "yfinance_not_installed"
        return _empty
    except Exception as exc:
        logger.warning("[OptionsFlow] %s failed: %s", ticker, exc)
        _empty.error = str(exc)[:120]
        return _empty
