"""GEX heatmap: net gamma exposure per strike x expiration.

Reuses the Black-Scholes gamma and GEX scaling conventions from
core.options.options_flow (call GEX positive, put GEX negative,
dollar-delta per 1% spot move).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from core.data.yf_client import safe_float, safe_int, yf_call

from .options_flow import _GEX_SCALE, _bs_gamma, _find_gamma_flip

logger = logging.getLogger(__name__)

__all__ = ["fetch_gex_heatmap"]

# Same 5-minute TTL convention as the flow cache.
_CACHE_TTL = 300.0
_cache_lock = threading.RLock()
_cache: dict = {}  # ticker -> (result, expire_monotonic)


def _cache_get(key: str) -> dict | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        result, exp = entry
        if time.monotonic() > exp:
            del _cache[key]
            return None
        return result


def _cache_put(key: str, result: dict) -> None:
    with _cache_lock:
        now = time.monotonic()
        for k in [k for k, (_, exp) in _cache.items() if now > exp]:
            del _cache[k]
        _cache[key] = (result, now + _CACHE_TTL)


def fetch_gex_heatmap(ticker: str) -> dict:
    """
    Net GEX per strike x expiration for the heatmap tab, across ALL listed
    future expiries (one yfinance chain request per expiry) and ALL strikes
    with open interest.

    GEX(K, exp) = sum_side [ +-OI * gamma(S, K, T, IV) ] * S^2 * 100 * 0.01
    (call positive / put negative, dollars of dealer delta per 1% spot move).

    Returns {ticker, spot, expiries, strikes, rows, net_gex, gamma_flip,
    max_pos, max_neg, accel_zone, error}. `rows` is sorted by strike
    descending and each row's `cells` aligns with `expiries`; a cell is None
    when the strike has no open interest at that expiry. `accel_zone` is the
    contiguous band of net-negative-GEX strikes below spot (where short-gamma
    dealer hedging amplifies a sell-off), or None. Summary stats (net_gex,
    gamma_flip, max_pos, max_neg, accel_zone) use near-term expiries only
    (<= 90 DTE); the grid shows all expiries. Cached for 5 minutes.
    """
    key = ticker.upper()
    cached = _cache_get(key)
    if cached is not None:
        return cached
    result = _fetch(ticker.upper())
    if not result.get("error"):
        _cache_put(key, result)
    return result


def _fetch(ticker: str) -> dict:
    empty = {
        "ticker": ticker,
        "spot": 0.0,
        "expiries": [],
        "strikes": [],
        "rows": [],
        "net_gex": 0.0,
        "gamma_flip": 0.0,
        "max_pos": None,
        "max_neg": None,
        "accel_zone": None,
        "error": None,
    }
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        info = t.fast_info
        spot = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None) or 0.0
        if not spot or spot <= 0:
            hist = yf_call(t.history, period="1d", interval="1m")
            spot = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else 0.0
        if spot <= 0:
            empty["error"] = "no_spot_price"
            return empty
        spot = float(spot)

        exps = t.options
        if not exps:
            empty["error"] = "no_options_chain"
            return empty

        today = datetime.now(timezone.utc).date()
        future: list[tuple[str, int]] = []
        for exp in exps:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (d - today).days
            if dte >= 0:
                future.append((exp, dte))
        if not future:
            empty["error"] = "no_future_expiry"
            return empty

        # cell_gex[expiry][strike] = net GEX; contracts feed the gamma-flip calc.
        cell_gex: dict[str, dict[float, float]] = {}
        contracts: list[dict] = []
        pending_iv: list[tuple[str, float, float, int, float]] = []  # rows with no IV
        iv_sum, iv_n = 0.0, 0

        for exp, dte in future:
            try:
                ch = yf_call(t.option_chain, exp)
            except Exception:
                continue
            T = max(dte, 1) / 365.0
            slot = cell_gex.setdefault(exp, {})
            for side, df in (("call", ch.calls), ("put", ch.puts)):
                sign = 1.0 if side == "call" else -1.0
                for _, row in df.iterrows():
                    K = safe_float(row.get("strike"))
                    if K <= 0:
                        continue
                    oi = safe_int(row.get("openInterest"))
                    if oi <= 0:
                        continue
                    iv = safe_float(row.get("impliedVolatility"))
                    if iv <= 0:
                        pending_iv.append((exp, K, sign, oi, T))
                        continue
                    iv_sum += iv
                    iv_n += 1
                    g = _bs_gamma(spot, K, T, iv)
                    slot[K] = slot.get(K, 0.0) + sign * oi * g * spot * spot * _GEX_SCALE
                    contracts.append({"strike": K, "type": side, "oi": oi, "iv": iv, "T": T})

        # Contracts with missing IV get the cross-chain average (fallback 0.3),
        # same convention as _contracts_from_chains.
        avg_iv = (iv_sum / iv_n) if iv_n else 0.3
        for exp, K, sign, oi, T in pending_iv:
            g = _bs_gamma(spot, K, T, avg_iv)
            slot = cell_gex.setdefault(exp, {})
            slot[K] = slot.get(K, 0.0) + sign * oi * g * spot * spot * _GEX_SCALE
            contracts.append(
                {"strike": K, "type": "call" if sign > 0 else "put", "oi": oi, "iv": avg_iv, "T": T}
            )

        cell_gex = {e: s for e, s in cell_gex.items() if s}
        if not cell_gex:
            empty["error"] = "no_open_interest"
            return empty

        expiries = [e for e, _ in future if e in cell_gex]

        # Union of ALL strikes carrying open interest, highest first.
        strikes_desc = sorted({K for s in cell_gex.values() for K in s}, reverse=True)

        rows = [
            {
                "strike": K,
                "cells": [
                    (round(cell_gex[e][K], 2) if K in cell_gex[e] else None) for e in expiries
                ],
            }
            for K in strikes_desc
        ]

        # Summary stats (net GEX, flip, max +-GEX, accel zone) use NEAR-TERM
        # expiries only (<= 90 DTE), matching the flow module's convention and
        # reference GEX platforms - otherwise long-dated LEAPS put OI drags
        # the flip and max -GEX to far-OTM strikes. The heatmap grid itself
        # still shows every expiry.
        dte_by_exp = dict(future)
        near_exps = [e for e in expiries if dte_by_exp.get(e, 9999) <= 90] or expiries
        per_strike: dict[float, float] = {}
        for e in near_exps:
            for K, v in cell_gex[e].items():
                per_strike[K] = per_strike.get(K, 0.0) + v
        net_gex = sum(per_strike.values())
        max_pos_k = max(per_strike, key=lambda k: per_strike[k])
        max_neg_k = min(per_strike, key=lambda k: per_strike[k])
        near_contracts = [c for c in contracts if c["T"] * 365.0 <= 90.5] or contracts
        gamma_flip = _find_gamma_flip(near_contracts, spot)

        # Acceleration zone: contiguous band of net-negative-GEX strikes BELOW
        # SPOT. Dealers are short gamma there, so hedging is pro-cyclical and
        # amplifies a sell-off through the band. Small positive blips (under
        # 2% of the largest absolute strike GEX) don't break the band; a
        # significant positive-GEX strike (dealer support) ends it.
        eps = 0.02 * max(abs(v) for v in per_strike.values())
        run: list[float] = []
        started = False
        for K in sorted(per_strike, reverse=True):
            if K >= spot:
                continue
            v = per_strike[K]
            if v < 0:
                run.append(K)
                started = True
            elif started and v > eps:
                break
        accel_zone = None
        if run:
            accel_zone = {
                "high": max(run),
                "low": min(run),
                "gex": round(sum(per_strike[k] for k in run), 2),
            }

        result = {
            "ticker": ticker,
            "spot": round(spot, 4),
            "expiries": expiries,
            "strikes": strikes_desc,
            "rows": rows,
            "net_gex": round(net_gex, 2),
            "gamma_flip": round(gamma_flip, 4),
            "max_pos": {"strike": max_pos_k, "gex": round(per_strike[max_pos_k], 2)},
            "max_neg": {"strike": max_neg_k, "gex": round(per_strike[max_neg_k], 2)},
            "accel_zone": accel_zone,
            "error": None,
        }
        logger.info(
            "[GEXHeatmap] %s spot=%.2f expiries=%d strikes=%d net_gex=%.0f",
            ticker, spot, len(expiries), len(strikes_desc), net_gex,
        )
        return result

    except ImportError:
        empty["error"] = "yfinance_not_installed"
        return empty
    except Exception as exc:
        logger.warning("[GEXHeatmap] %s failed: %s", ticker, exc)
        empty["error"] = str(exc)[:120]
        return empty
