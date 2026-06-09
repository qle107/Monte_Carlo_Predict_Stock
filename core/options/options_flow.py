"""Options flow: max pain, GEX, and unusual activity.

Sector labels live in core.sectors; yfinance plumbing lives in core.yf_client.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass

from core.data.sectors import HIGH_VOLUME_ETFS, sector_for
from core.data.yf_client import is_rate_limit, safe_float, safe_int, yf_call

__all__ = [
    "HIGH_VOLUME_ETFS",
    "GEXBar",
    "OptionsFlow",
    "UnusualOption",
    "fetch_options_flow",
    "scan_unusual_options",
    "scan_volume_spikes",
]

logger = logging.getLogger(__name__)

# Avoid hammering yfinance - 5 min TTL.
_FLOW_CACHE_TTL = 300.0
_flow_cache_lock = threading.RLock()
_flow_cache: dict = {}  # ticker -> (OptionsFlow, expire_monotonic)


def _flow_cache_get(key: str) -> OptionsFlow | None:
    with _flow_cache_lock:
        entry = _flow_cache.get(key)
        if entry is None:
            return None
        result, exp = entry
        if time.monotonic() > exp:
            del _flow_cache[key]
            return None
        return result


def _flow_cache_put(key: str, result: OptionsFlow) -> None:
    with _flow_cache_lock:
        now = time.monotonic()
        stale = [k for k, (_, exp) in _flow_cache.items() if now > exp]
        for k in stale:
            del _flow_cache[k]
        _flow_cache[key] = (result, now + _FLOW_CACHE_TTL)


# Dataclasses


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
    gex_positive: bool  # True means volatility-damping regime
    days_to_expiry: float
    gex_profile: list[GEXBar]  # sorted by strike
    expiries_used: list[str] | None = None  # expiries aggregated into GEX/walls
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
            "expiries_used": self.expiries_used or [],
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


# Black-Scholes helpers


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """Black-Scholes gamma. Returns 0 on bad inputs.
    S=spot, K=strike, T=time to expiry in years, sigma=implied vol."""
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0


# Max Pain


def _compute_max_pain(calls_df, puts_df) -> float:
    """
    Max pain = strike that minimizes total payout to option holders.
    Standard convention: computed on a SINGLE (nearest) expiry chain.
    Evaluates total payout across all strikes in the chain:
      sum_K [OI_call(K) * max(P-K, 0) + OI_put(K) * max(K-P, 0)] * 100
    """
    all_strikes = sorted(set(list(calls_df["strike"].values) + list(puts_df["strike"].values)))
    if not all_strikes:
        return 0.0

    # Sum (not overwrite) OI per strike - chains can carry duplicate strike rows.
    call_oi: dict[float, int] = {}
    for _, r in calls_df.iterrows():
        k = float(r["strike"])
        call_oi[k] = call_oi.get(k, 0) + safe_int(r["openInterest"])
    put_oi: dict[float, int] = {}
    for _, r in puts_df.iterrows():
        k = float(r["strike"])
        put_oi[k] = put_oi.get(k, 0) + safe_int(r["openInterest"])

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


# GEX profile

# GEX scaling: gamma * OI * 100 (contract multiplier) * S^2 * 0.01.
# The 0.01 expresses GEX as dollar delta change per 1% spot move - the
# industry convention (SpotGamma/MenthorQ style). Without it values are
# inflated 100x vs reference platforms.
_GEX_SCALE = 100.0 * 0.01


def _contracts_from_chains(
    chains: list[tuple[str, float, object, object]],
) -> list[dict]:
    """
    Flatten (expiry, T, calls_df, puts_df) chains into per-contract dicts:
      {strike, type, oi, iv, T}
    Contracts with zero/NaN IV get the cross-chain average IV (fallback 0.3).
    """
    contracts: list[dict] = []
    for _exp, T, calls_df, puts_df in chains:
        for side, df in (("call", calls_df), ("put", puts_df)):
            for _, row in df.iterrows():
                oi = safe_int(row.get("openInterest"))
                if oi <= 0:
                    continue
                contracts.append(
                    {
                        "strike": float(row["strike"]),
                        "type": side,
                        "oi": oi,
                        "iv": safe_float(row.get("impliedVolatility")),
                        "T": T,
                    }
                )

    ivs = [c["iv"] for c in contracts if c["iv"] > 0]
    avg_iv = (sum(ivs) / len(ivs)) if ivs else 0.3
    for c in contracts:
        if c["iv"] <= 0:
            c["iv"] = avg_iv
    return contracts


def _total_gex_at(contracts: list[dict], price: float, risk_free: float = 0.05) -> float:
    """Total dealer net GEX if spot were at `price` (gamma re-evaluated at price)."""
    total = 0.0
    for c in contracts:
        g = _bs_gamma(price, c["strike"], c["T"], c["iv"], risk_free)
        sign = 1.0 if c["type"] == "call" else -1.0
        total += sign * c["oi"] * g
    return total * price * price * _GEX_SCALE


def _compute_gex_profile(
    contracts: list[dict],
    spot: float,
    risk_free: float = 0.05,
    max_bars: int = 50,
) -> list[GEXBar]:
    """
    Net GEX per strike, aggregated across expiries:
      GEX_net(K) = sum_contracts(K) [ +-OI * gamma(S, K, T, IV) ] * S^2 * 100 * 0.01

    Convention: call GEX positive (dealers long gamma), put GEX negative.
    Output is capped at max_bars strikes centered around spot to keep charts legible.
    """
    by_strike: dict[float, dict] = {}
    for c in contracts:
        K = c["strike"]
        slot = by_strike.setdefault(K, {"gex": 0.0, "call_oi": 0, "put_oi": 0})
        g = _bs_gamma(spot, K, c["T"], c["iv"], risk_free)
        if c["type"] == "call":
            slot["gex"] += c["oi"] * g * spot * spot * _GEX_SCALE
            slot["call_oi"] += c["oi"]
        else:
            slot["gex"] -= c["oi"] * g * spot * spot * _GEX_SCALE
            slot["put_oi"] += c["oi"]

    bars = [
        GEXBar(strike=K, gex=v["gex"], call_oi=v["call_oi"], put_oi=v["put_oi"])
        for K, v in by_strike.items()
    ]
    bars.sort(key=lambda b: b.strike)

    # Limit to most relevant strikes (centered around spot)
    if len(bars) > max_bars:
        spot_idx = min(range(len(bars)), key=lambda i: abs(bars[i].strike - spot))
        window_half = max_bars // 2
        start = max(0, spot_idx - window_half)
        end = min(len(bars), start + max_bars)
        if end - start < max_bars:
            start = max(0, end - max_bars)
        bars = bars[start:end]

    return bars


def _find_gamma_flip(contracts: list[dict], spot: float, n_steps: int = 120) -> float:
    """
    Gamma flip = price level where total dealer net GEX crosses zero.

    Correct method: re-evaluate the FULL net GEX (gamma recomputed at each
    hypothetical spot level) across a price grid (+-20% around spot) and find
    the zero crossing nearest to spot. The old per-strike cumulative walk
    almost never detected a crossing and fell back to the lowest strike in
    the window (e.g. reporting 270 when the true flip was ~380).
    """
    if not contracts:
        return spot

    lo, hi = spot * 0.80, spot * 1.20
    step = (hi - lo) / n_steps
    levels = [lo + i * step for i in range(n_steps + 1)]
    vals = [_total_gex_at(contracts, p) for p in levels]

    crossings: list[float] = []
    for i in range(len(levels) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if v0 == 0.0:
            crossings.append(levels[i])
        elif (v0 < 0.0) != (v1 < 0.0):
            frac = abs(v0) / (abs(v0) + abs(v1) + 1e-12)
            crossings.append(levels[i] + frac * step)

    if crossings:
        return min(crossings, key=lambda p: abs(p - spot))

    # No flip inside +-20%: regime is one-sided here. Report the scan boundary
    # on the side the flip must lie (below if all-positive, above if all-negative).
    return lo if vals[len(vals) // 2] > 0 else hi


def _find_walls(contracts: list[dict], spot: float) -> tuple[float, float]:
    """
    Call wall = strike with heaviest call OI AT/ABOVE spot (overhead resistance).
    Put wall  = strike with heaviest put  OI AT/BELOW spot (support floor).
    OI is aggregated across expiries so one thin/legacy chain can't pin both
    walls to a single stale deep-ITM block. Walls are never allowed on the
    wrong side of spot - if no OI exists on the proper side, fall back to the
    strike nearest spot on that side, else spot itself.
    """
    call_oi: dict[float, int] = {}
    put_oi: dict[float, int] = {}
    for c in contracts:
        if c["type"] == "call":
            call_oi[c["strike"]] = call_oi.get(c["strike"], 0) + c["oi"]
        else:
            put_oi[c["strike"]] = put_oi.get(c["strike"], 0) + c["oi"]

    def _wall(oi_map: dict[float, int], above: bool) -> float:
        side = {
            k: v
            for k, v in oi_map.items()
            if (k >= spot * 0.99 if above else k <= spot * 1.01) and v > 0
        }
        if side:
            return max(side, key=lambda k: side[k])
        # No OI on the proper side: nearest strike on that side, else spot.
        strikes = [k for k in oi_map if (k >= spot if above else k <= spot)]
        return min(strikes, key=lambda k: abs(k - spot)) if strikes else spot

    return _wall(call_oi, above=True), _wall(put_oi, above=False)


# Public entry point


def fetch_options_flow(ticker: str, spot: float | None = None) -> OptionsFlow:
    """
    Fetch options chain and compute GEX + Max Pain.
    Returns an OptionsFlow with error set if anything fails.
    Cached for 5 minutes.
    """
    key = ticker.upper()
    cached = _flow_cache_get(key)
    if cached is not None:
        return cached

    result = _fetch(key, spot)
    _flow_cache_put(key, result)
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

        # Fetch up to 5 near-term expiries (0-90 DTE) and AGGREGATE them.
        # Using a single "most liquid" chain pinned every level (max pain,
        # walls, flip) to whatever stale deep-ITM block dominated that one
        # chain - e.g. all levels reading $300 with spot at $382.
        exps = t.options
        if not exps:
            _empty.error = "no_options_chain"
            return _empty

        now = datetime.now(timezone.utc).date()

        selected: list[tuple[str, int]] = []  # (expiry, dte)
        for exp in exps:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (d - now).days
            if dte < 0:
                continue
            if dte <= 90:
                selected.append((exp, dte))
            if len(selected) >= 5:
                break
        if not selected:
            # Fallback: first future expiry regardless of the 90-day cap
            for exp in exps:
                try:
                    d = datetime.strptime(exp, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if (d - now).days >= 0:
                    selected = [(exp, (d - now).days)]
                    break
        if not selected:
            _empty.error = "no_future_expiry"
            return _empty

        # Focus on strikes within ±30% of spot (wider than ±20% to catch illiquid/small-cap chains)
        lo, hi = spot * 0.70, spot * 1.30

        chains: list[tuple[str, float, object, object]] = []  # (expiry, T, calls, puts)
        for exp, dte in selected:
            try:
                ch = t.option_chain(exp)
            except Exception:
                continue
            c = ch.calls[(ch.calls["strike"] >= lo) & (ch.calls["strike"] <= hi)].copy()
            p = ch.puts[(ch.puts["strike"] >= lo) & (ch.puts["strike"] <= hi)].copy()
            if c.empty and p.empty:
                continue
            T = max(dte, 1) / 365.0
            chains.append((exp, T, c, p))

        if not chains:
            _empty.error = "no_near_strikes"
            return _empty

        # Max pain: standard convention = nearest expiry chain only.
        nearest_exp, nearest_T, nearest_calls, nearest_puts = chains[0]
        if nearest_calls.empty or nearest_puts.empty:
            mp_calls, mp_puts = next(
                ((c, p) for _, _, c, p in chains if not c.empty and not p.empty),
                (nearest_calls, nearest_puts),
            )
        else:
            mp_calls, mp_puts = nearest_calls, nearest_puts
        max_pain = _compute_max_pain(mp_calls, mp_puts)

        # Aggregate every contract across the selected expiries.
        contracts = _contracts_from_chains(chains)
        if not contracts:
            _empty.error = "no_open_interest"
            return _empty

        call_wall, put_wall = _find_walls(contracts, spot)
        gex_profile = _compute_gex_profile(contracts, spot)
        net_gex = sum(b.gex for b in gex_profile)
        gamma_flip = _find_gamma_flip(contracts, spot)

        result = OptionsFlow(
            ticker=ticker,
            expiry=nearest_exp,
            spot=spot,
            max_pain=max_pain,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_flip=gamma_flip,
            net_gex=net_gex,
            gex_positive=(net_gex >= 0),
            days_to_expiry=nearest_T * 365,
            gex_profile=gex_profile,
            expiries_used=[e for e, _, _, _ in chains],
            error=None,
        )
        logger.info(
            "[OptionsFlow] %s  spot=%.2f  max_pain=%.2f  call_wall=%.2f  "
            "put_wall=%.2f  gamma_flip=%.2f  net_gex=%.0f  exps=%s",
            ticker,
            spot,
            max_pain,
            call_wall,
            put_wall,
            gamma_flip,
            net_gex,
            ",".join(e for e, _, _, _ in chains),
        )
        return result

    except ImportError:
        _empty.error = "yfinance_not_installed"
        return _empty
    except Exception as exc:
        logger.warning("[OptionsFlow] %s failed: %s", ticker, exc)
        _empty.error = str(exc)[:120]
        return _empty


@dataclass
class UnusualOption:
    """A single unusual options contract."""

    ticker: str
    expiry: str
    strike: float
    option_type: str  # "call" | "put"
    volume: int
    open_interest: int
    vol_oi_ratio: float
    implied_vol: float
    avg_chain_iv: float  # average IV across the chain for context
    in_the_money: bool
    premium_per_contract: float  # mid-price * 100
    total_premium: float  # volume * premium_per_contract
    unusual_score: float  # 0-1 composite
    flags: list  # e.g. ["high_vol_oi", "iv_spike", "otm_sweep"]
    sentiment: str  # "bullish" | "bearish" | "mixed"
    spot: float
    days_to_expiry: float
    percent_change: float = 0.0  # option contract % price change today
    sector: str = "Other"  # sector label from core.data.sectors
    trade_style: str = "block"  # "sweep" | "block" (snapshot approximation)
    exec_side: str = "mid"  # "ask" | "bid" | "mid" (lit-execution lean estimate)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "expiry": self.expiry,
            "strike": round(self.strike, 4),
            "option_type": self.option_type,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "vol_oi_ratio": round(self.vol_oi_ratio, 2),
            "implied_vol": round(self.implied_vol * 100, 2),  # as %
            "avg_chain_iv": round(self.avg_chain_iv * 100, 2),
            "in_the_money": self.in_the_money,
            "percent_change": round(self.percent_change, 2),
            "sector": self.sector,
            "premium_per_contract": round(self.premium_per_contract, 2),
            "total_premium": round(self.total_premium, 2),
            "unusual_score": round(self.unusual_score, 4),
            "flags": self.flags,
            "sentiment": self.sentiment,
            "spot": round(self.spot, 4),
            "days_to_expiry": round(self.days_to_expiry, 1),
            "trade_style": self.trade_style,
            "exec_side": self.exec_side,
        }


def _estimate_exec_side(last: float, bid: float, ask: float) -> str:
    """
    Estimate whether a print leaned buyer-initiated ("ask") or seller-initiated
    ("bid") from the contract snapshot (last vs bid/ask). yfinance gives no true
    trade-condition tape, so this is a lit-execution lean, not an exchange flag.

      last at/above ask          -> "ask"
      last at/below bid          -> "bid"
      last above bid/ask midpoint-> "ask"  (lean buy)
      last below midpoint        -> "bid"  (lean sell)
      otherwise / unknown spread -> "mid"
    """
    if bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if last >= ask * 0.999:
            return "ask"
        if last <= bid * 1.001:
            return "bid"
        if last > mid:
            return "ask"
        if last < mid:
            return "bid"
        return "mid"
    return "mid"


def _classify_trade_style(vol_oi: float, vol_oi_threshold: float) -> str:
    """
    Approximate sweep vs block from a chain snapshot.

    A *sweep* is aggressive, opening, new-money flow: traded volume far exceeds
    standing open interest (high vol/OI). A *block* is a large single negotiated
    print that trades against existing OI (moderate vol/OI). True sweep/block
    tagging needs the trade tape (multi-exchange / single-print conditions),
    which yfinance does not expose; this uses vol/OI as the proxy.
    """
    return "sweep" if vol_oi >= vol_oi_threshold else "block"


def _scan_ticker_unusual(
    ticker: str,
    min_volume: int = 10,
    min_oi: int = 10,
    vol_oi_threshold: float = 3.0,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    max_dte: int = 60,
    min_premium: float = 0.0,
    new_positions_only: bool = False,
    min_sweep_premium: float = 50_000.0,
    min_block_premium: float = 100_000.0,
    exclude_bid_side: bool = True,
) -> list[UnusualOption] | None:
    """
    Fetch options chain for one ticker and return unusual contracts.

    Returns a list of UnusualOption hits (empty list = nothing unusual found),
    or None if the ticker appears delisted / has no valid market data.

    Flow filters (snapshot approximations - see _classify_trade_style /
    _estimate_exec_side for the data caveats):
      - sweeps kept only if total premium >= min_sweep_premium
      - blocks kept only if total premium >= min_block_premium
      - bid-side (seller-initiated lean) prints dropped when exclude_bid_side
    """
    import statistics
    from datetime import datetime, timezone

    import yfinance as yf

    results: list[UnusualOption] = []
    try:
        t = yf.Ticker(ticker.upper())

        # Spot price (guarded by semaphore + retry)
        info = yf_call(lambda: t.fast_info)
        spot = float(getattr(info, "last_price", None) or getattr(info, "regular_market_price", None) or 0.0)
        if spot <= 0:
            # No price at all - check options to distinguish delisted vs. closed market
            try:
                exps = yf_call(lambda: t.options)
            except Exception:
                exps = []
            if not exps:
                logger.info(
                    "[UnusualOptions] %s: no price + no options chain, likely delisted/invalid", ticker
                )
                return None  # signal "delisted" to caller
            return []  # has options but no live price - data gap, not delisted

        now_date = datetime.now(timezone.utc).date()

        # Collect expiries within max_dte (guarded by semaphore)
        all_exps = yf_call(lambda: t.options) or []
        near_exps = []
        for exp in all_exps:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (d - now_date).days
                if 0 <= dte <= max_dte:
                    near_exps.append((exp, dte))
            except ValueError:
                pass

        if not near_exps:
            return []

        # Aggregate all contracts across near-term expiries
        all_contracts: list[dict] = []
        for exp, dte in near_exps:
            try:
                chain = yf_call(t.option_chain, exp)
            except Exception as exc:
                if is_rate_limit(exc):
                    logger.warning("[UnusualOptions] %s rate-limited on expiry %s (giving up)", ticker, exp)
                    break  # stop fetching more expiries for this ticker rather than silently skip all
                continue

            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    vol = safe_int(row.get("volume"))
                    oi = safe_int(row.get("openInterest"))
                    iv = safe_float(row.get("impliedVolatility"))
                    strike = safe_float(row.get("strike"))
                    bid = safe_float(row.get("bid"))
                    ask = safe_float(row.get("ask"))
                    last = safe_float(row.get("lastPrice"))
                    # Use bid/ask midpoint when available, fall back to lastPrice
                    mid = (bid + ask) / 2.0 if (bid > 0 or ask > 0) else last
                    itm = bool(row.get("inTheMoney", False))
                    pct_chg = safe_float(row.get("percentChange"))

                    if vol < min_volume or oi < min_oi or strike <= 0:
                        continue

                    total_prem = vol * mid * 100

                    # Skip contracts below the absolute premium floor early
                    if min_premium > 0 and total_prem < min_premium:
                        continue

                    # New-positions filter: only contracts where vol > OI
                    if new_positions_only and vol <= oi:
                        continue

                    # Execution-side lean: drop bid-side (seller-initiated) prints
                    exec_side = _estimate_exec_side(last, bid, ask)
                    if exclude_bid_side and exec_side == "bid":
                        continue

                    # Sweep vs block + per-style premium thresholds
                    vol_oi = vol / max(oi, 1)
                    trade_style = _classify_trade_style(vol_oi, vol_oi_threshold)
                    style_floor = min_sweep_premium if trade_style == "sweep" else min_block_premium
                    if total_prem < style_floor:
                        continue

                    all_contracts.append(
                        {
                            "ticker": ticker.upper(),
                            "expiry": exp,
                            "dte": dte,
                            "strike": strike,
                            "type": side,
                            "volume": vol,
                            "oi": oi,
                            "iv": iv,
                            "mid": mid,
                            "itm": itm,
                            "pct_chg": pct_chg,
                            "trade_style": trade_style,
                            "exec_side": exec_side,
                        }
                    )

        if not all_contracts:
            return []

        # Chain-wide IV stats for z-score
        ivs = [c["iv"] for c in all_contracts if c["iv"] > 0]
        avg_iv = statistics.mean(ivs) if ivs else 0.3
        std_iv = statistics.stdev(ivs) if len(ivs) > 1 else 0.05

        # Total call + put volume for CP divergence
        total_call_vol = sum(c["volume"] for c in all_contracts if c["type"] == "call")
        total_put_vol = sum(c["volume"] for c in all_contracts if c["type"] == "put")
        cp_ratio = total_call_vol / max(total_put_vol, 1)

        # Premium percentile threshold (top-30% = notable)
        premiums = sorted(c["volume"] * c["mid"] * 100 for c in all_contracts)
        prem_threshold = premiums[int(len(premiums) * 0.70)] if premiums else 0

        # Signal weights (sum to 1.0).
        # cp_divergence is chain-level (same for every contract on the ticker)
        # so it stays at 0.05 - prevents all contracts scoring 100% just because
        # the chain is call/put skewed.
        _SIG_W = {
            "high_vol_oi": 0.35,  # contract-specific: new money flowing in
            "iv_spike": 0.25,  # contract-specific: elevated IV conviction
            "otm_sweep": 0.20,  # contract-specific: directional speculation
            "large_premium": 0.15,  # contract-specific: notional size
            "cp_divergence": 0.05,  # chain-level tie-breaker only
        }

        for c in all_contracts:
            vol_oi = c["volume"] / max(c["oi"], 1)
            iv = c["iv"]
            mid = c["mid"]
            total_prem = c["volume"] * mid * 100
            otm_dist = abs(c["strike"] / spot - 1.0)
            is_otm = not c["itm"]

            flags: list[str] = []
            signal_scores: dict[str, float] = {}

            if vol_oi >= vol_oi_threshold:
                flags.append("high_vol_oi")
                signal_scores["high_vol_oi"] = min(vol_oi / (vol_oi_threshold * 3), 1.0)

            if std_iv > 0 and (iv - avg_iv) / std_iv >= iv_spike_z:
                flags.append("iv_spike")
                signal_scores["iv_spike"] = min((iv - avg_iv) / (std_iv * 3), 1.0)

            if is_otm and otm_dist >= otm_pct and c["volume"] >= min_volume * 2:
                flags.append("otm_sweep")
                signal_scores["otm_sweep"] = min(otm_dist / 0.20, 1.0)

            if total_prem >= prem_threshold and total_prem > 0:
                flags.append("large_premium")
                max_prem = max(premiums[-1], 1)
                signal_scores["large_premium"] = min(total_prem / max_prem, 1.0)

            if cp_ratio > 3.0 or cp_ratio < 0.33:
                flags.append("cp_divergence")
                div = max(cp_ratio, 1.0 / max(cp_ratio, 0.01))
                signal_scores["cp_divergence"] = min((div - 1) / 5.0, 1.0)

            if not flags:
                continue

            # Weighted composite - denominator is always the full weight sum (1.0),
            # so a contract with only cp_divergence maxes at 0.05, not 1.0.
            composite = sum(signal_scores[f] * _SIG_W[f] for f in signal_scores)

            # Sentiment
            if c["type"] == "call":
                sentiment = "bullish"
            elif c["type"] == "put":
                sentiment = "bearish"
            else:
                sentiment = "mixed"
            # Override: if OTM puts dominate (cp_ratio < 0.5), even calls are mixed
            if "cp_divergence" in flags and cp_ratio < 0.5:
                sentiment = "bearish" if c["type"] == "put" else "mixed"

            results.append(
                UnusualOption(
                    ticker=c["ticker"],
                    expiry=c["expiry"],
                    strike=c["strike"],
                    option_type=c["type"],
                    volume=c["volume"],
                    open_interest=c["oi"],
                    vol_oi_ratio=round(vol_oi, 2),
                    implied_vol=iv,
                    avg_chain_iv=avg_iv,
                    in_the_money=c["itm"],
                    premium_per_contract=round(mid * 100, 2),
                    total_premium=round(total_prem, 2),
                    unusual_score=round(composite, 4),
                    flags=flags,
                    sentiment=sentiment,
                    spot=spot,
                    days_to_expiry=float(c["dte"]),
                    percent_change=c.get("pct_chg", 0.0),
                    sector=sector_for(ticker),
                    trade_style=c.get("trade_style", "block"),
                    exec_side=c.get("exec_side", "mid"),
                )
            )

    except Exception as exc:
        logger.warning("[UnusualOptions] %s scan failed: %s", ticker, exc)

    # Sort by composite score descending, cap per ticker
    results.sort(key=lambda x: x.unusual_score, reverse=True)
    return results[:20]


# Cooperative cancellation for in-flight unusual-options scans.
# POST /api/options/unusual/cancel sets this; queued workers then no-op so the
# scan stops hammering yfinance (useful when rate-limited). Cleared at the
# start of every new scan.
_scan_cancel = threading.Event()


def cancel_unusual_scan() -> None:
    """Signal any in-flight unusual-options scan to skip its remaining tickers."""
    _scan_cancel.set()


def scan_unusual_options(
    tickers: list[str],
    min_volume: int = 10,
    min_oi: int = 10,
    vol_oi_threshold: float = 3.0,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    max_dte: int = 60,
    max_concurrent: int = 6,
    top_n: int = 50,
    min_premium: float = 0.0,
    new_positions_only: bool = False,
    min_sweep_premium: float = 50_000.0,
    min_block_premium: float = 100_000.0,
    exclude_bid_side: bool = True,
    exclude_high_volume_etfs: bool = True,
) -> dict:
    """
    Scan a list of tickers for unusual options activity. Returns top-N hits
    ranked by composite unusual score.

    Flow filters (snapshot approximations):
      - sweeps kept only if premium >= min_sweep_premium (default $50K)
      - blocks kept only if premium >= min_block_premium (default $100K)
      - exclude_bid_side drops seller-initiated (bid-side) prints
      - exclude_high_volume_etfs removes index/sector/leverage/vol ETFs whose
        tape is dominated by hedging rather than conviction flow

    Returns a dict with keys: hits (list), summary (dict), scanned_at (ISO-8601).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone

    # De-dup (watchlists overlap) and normalize
    tickers = list(dict.fromkeys(t.upper().strip() for t in tickers if t.strip()))

    excluded_etfs: list[str] = []
    if exclude_high_volume_etfs:
        excluded_etfs = [t for t in tickers if t in HIGH_VOLUME_ETFS]
        tickers = [t for t in tickers if t not in HIGH_VOLUME_ETFS]

    if not tickers:
        return {"hits": [], "summary": {}, "scanned_at": datetime.now(timezone.utc).isoformat()}

    all_hits: list[UnusualOption] = []
    tickers_with_hits: set[str] = set()
    delisted_tickers: set[str] = set()

    logger.info("[UnusualOptions] scanning %d tickers (max_concurrent=%d)", len(tickers), max_concurrent)
    _scan_cancel.clear()

    # Stagger worker start times so threads don't all hit yfinance at t=0.
    # Each worker sleeps a random jitter before its first network call.
    _stagger_lock = threading.Lock()
    _stagger_counter = [0]

    def _worker(tkr: str) -> list[UnusualOption] | None:
        # Cancelled mid-scan: skip remaining tickers without touching yfinance.
        if _scan_cancel.is_set():
            return []
        # Staggered start: worker N sleeps N * 0.15 s (+ jitter) before touching yfinance
        with _stagger_lock:
            idx = _stagger_counter[0]
            _stagger_counter[0] += 1
        time.sleep(idx * 0.15 + random.uniform(0.0, 0.25))
        return _scan_ticker_unusual(
            tkr,
            min_volume=min_volume,
            min_oi=min_oi,
            vol_oi_threshold=vol_oi_threshold,
            iv_spike_z=iv_spike_z,
            otm_pct=otm_pct,
            max_dte=max_dte,
            min_premium=min_premium,
            new_positions_only=new_positions_only,
            min_sweep_premium=min_sweep_premium,
            min_block_premium=min_block_premium,
            exclude_bid_side=exclude_bid_side,
        )

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_map = {pool.submit(_worker, tkr): tkr for tkr in tickers}
        for fut in as_completed(future_map):
            tkr = future_map[fut]
            try:
                hits = fut.result()
                if hits is None:
                    # None = delisted / no valid market data
                    delisted_tickers.add(tkr)
                elif hits:
                    all_hits.extend(hits)
                    tickers_with_hits.add(tkr)
            except Exception as exc:
                if is_rate_limit(exc):
                    logger.warning("[UnusualOptions] worker %s rate-limited (all retries exhausted)", tkr)
                else:
                    logger.warning("[UnusualOptions] worker %s raised: %s", tkr, exc)

    # Global sort and cap
    all_hits.sort(key=lambda x: x.unusual_score, reverse=True)
    top_hits = all_hits[:top_n]

    bullish = sum(1 for h in top_hits if h.sentiment == "bullish")
    bearish = sum(1 for h in top_hits if h.sentiment == "bearish")
    mixed = sum(1 for h in top_hits if h.sentiment == "mixed")
    sweeps = sum(1 for h in top_hits if h.trade_style == "sweep")
    blocks = sum(1 for h in top_hits if h.trade_style == "block")

    if delisted_tickers:
        logger.info(
            "[UnusualOptions] %d tickers appear delisted/invalid: %s",
            len(delisted_tickers),
            ", ".join(sorted(delisted_tickers)),
        )

    if _scan_cancel.is_set():
        logger.info("[UnusualOptions] scan cancelled by user - remaining tickers skipped")

    logger.info(
        "[UnusualOptions] scan complete: %d hits from %d/%d tickers  bull=%d bear=%d mixed=%d  delisted=%d",
        len(top_hits),
        len(tickers_with_hits),
        len(tickers),
        bullish,
        bearish,
        mixed,
        len(delisted_tickers),
    )

    return {
        "hits": [h.to_dict() for h in top_hits],
        "summary": {
            "tickers_scanned": len(tickers),
            "tickers_with_hits": len(tickers_with_hits),
            "total_hits": len(top_hits),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "mixed_count": mixed,
            "sweep_count": sweeps,
            "block_count": blocks,
            "delisted_count": len(delisted_tickers),
            "delisted_tickers": sorted(delisted_tickers),
            "cancelled": _scan_cancel.is_set(),
            "excluded_etf_count": len(excluded_etfs),
            "filters": {
                "min_sweep_premium": min_sweep_premium,
                "min_block_premium": min_block_premium,
                "exclude_bid_side": exclude_bid_side,
                "exclude_high_volume_etfs": exclude_high_volume_etfs,
            },
        },
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


# Volume spike scanner


def scan_volume_spikes(
    tickers: list[str],
    min_vol_ratio: float = 2.0,
    top_n: int = 50,
    lookback_days: int = 20,
    batch_size: int = 100,
    min_avg_volume: int = 50_000,
) -> list[dict]:
    """
    Scan tickers for unusual single-day volume spikes.

    For each ticker computes: vol_ratio = today_volume / avg(volume over lookback_days).
    Returns the top_n tickers where vol_ratio >= min_vol_ratio, sorted descending.
    Batches downloads (batch_size tickers at a time) to avoid timeouts.
    Tickers with avg volume below min_avg_volume are skipped as illiquid.

    Each result dict contains: ticker, price, today_volume, avg_volume, vol_ratio, sector.
    """
    import pandas as pd
    import yfinance as yf

    results: list[dict] = []
    unique_tickers = list(dict.fromkeys(t.upper().strip() for t in tickers if t.strip()))

    period_str = f"{lookback_days + 8}d"  # extra buffer for weekends/holidays

    for batch_start in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[batch_start : batch_start + batch_size]
        try:
            raw = yf.download(
                batch,
                period=period_str,
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="column",
            )
        except Exception as exc:
            logger.warning("[vol_spike] batch download failed (start=%d): %s", batch_start, exc)
            continue

        if raw is None or raw.empty:
            continue

        # Extract Volume & Close
        try:
            if len(batch) == 1:
                # Single-ticker download: flat column names
                ticker = batch[0]
                vol_series_map = {ticker: raw.get("Volume", pd.Series(dtype=float)).dropna()}
                close_series_map = {ticker: raw.get("Close", pd.Series(dtype=float)).dropna()}
            elif isinstance(raw.columns, pd.MultiIndex):
                # Multi-ticker: MultiIndex (field, ticker)
                vol_df = raw.get("Volume", pd.DataFrame())
                close_df = raw.get("Close", pd.DataFrame())
                vol_series_map = {
                    t: (
                        vol_df[t].dropna()
                        if isinstance(vol_df, pd.DataFrame) and t in vol_df.columns
                        else pd.Series(dtype=float)
                    )
                    for t in batch
                }
                close_series_map = {
                    t: (
                        close_df[t].dropna()
                        if isinstance(close_df, pd.DataFrame) and t in close_df.columns
                        else pd.Series(dtype=float)
                    )
                    for t in batch
                }
            else:
                # Unexpected shape - skip batch
                logger.debug("[vol_spike] unexpected column shape in batch %d", batch_start)
                continue
        except Exception as exc:
            logger.warning("[vol_spike] column extraction failed (batch %d): %s", batch_start, exc)
            continue

        for ticker in batch:
            try:
                vol_s = vol_series_map.get(ticker, pd.Series(dtype=float))
                close_s = close_series_map.get(ticker, pd.Series(dtype=float))

                if len(vol_s) < 5:
                    continue

                today_vol = float(vol_s.iloc[-1])
                hist = vol_s.iloc[max(0, len(vol_s) - lookback_days - 1) : -1]
                if len(hist) == 0:
                    continue
                avg_vol = float(hist.mean())

                # Skip illiquid / data-absent tickers
                if avg_vol < min_avg_volume or today_vol <= 0:
                    continue

                ratio = today_vol / avg_vol
                if ratio < min_vol_ratio:
                    continue

                price = round(float(close_s.iloc[-1]), 4) if not close_s.empty else 0.0
                results.append(
                    {
                        "ticker": ticker,
                        "price": price,
                        "today_volume": int(today_vol),
                        "avg_volume": int(avg_vol),
                        "vol_ratio": round(ratio, 2),
                        "sector": sector_for(ticker),
                    }
                )
            except Exception:
                continue

    results.sort(key=lambda x: x["vol_ratio"], reverse=True)
    logger.info(
        "[vol_spike] scan complete: %d/%d tickers exceed %.1fx avg vol",
        len(results),
        len(unique_tickers),
        min_vol_ratio,
    )
    return results[:top_n]
