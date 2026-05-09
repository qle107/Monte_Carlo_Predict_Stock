"""
core/scanner.py — Multi-stock breakout / breakdown scanner.

Scans a list of tickers concurrently using the existing indicator →
regime → signal pipeline.  Returns a ranked list of scan results,
sorted by breakout/breakdown score.

Breakout score  > 0  → breaking out upward
Breakdown score < 0  → breaking down

Each result also includes a condensed summary of the key signals so
the dashboard can display them without additional API calls.

Usage
─────
    from core.scanner import scan_tickers
    results = await scan_tickers(
        tickers=["AAPL", "NVDA", "SPY", ...],
        interval="1d",
        lookback=60,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np

from .fetcher     import fetch_candles
from .indicators  import compute_indicators
from .montecarlo  import run as run_mc
from .regime      import detect_regime
from .signal      import compute_signal
from config import cfg

logger = logging.getLogger(__name__)


# ─── Predefined watchlists ───────────────────────────────────────────────────

WATCHLISTS: Dict[str, List[str]] = {
    "sp500_large": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
        "UNH", "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG", "HD", "COST",
        "MRK", "ABBV", "CVX", "CRM", "AMD", "PEP", "KO", "ADBE", "TMO",
        "ACN", "MCD", "BAC", "NFLX", "CSCO", "DHR", "LIN", "ABT", "WMT",
        "TXN", "INTC", "NEE", "DIS", "QCOM", "PM", "AMGN", "UPS", "INTU",
        "RTX", "CAT", "AMAT", "BKNG", "GS",
    ],
    "tech": [
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "AVGO", "AMAT",
        "MU", "KLAC", "LRCX", "ASML", "TSM", "ARM", "SMCI", "MRVL",
        "ADBE", "CRM", "ORCL", "SAP", "SNOW", "PLTR", "CRWD", "PANW",
        "ZS", "DDOG", "NET", "MDB", "SHOP", "SQ", "PYPL", "COIN",
    ],
    "etfs": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "GLD", "SLV", "TLT",
        "HYG", "LQD", "EEM", "EFA", "XLE", "XLF", "XLK", "XLV", "XLI",
        "XLU", "XLP", "XLB", "XLRE", "VNQ", "ARKK", "SQQQ", "TQQQ",
    ],
    "biotech": [
        "MRNA", "BNTX", "GILD", "BIIB", "REGN", "VRTX", "ALNY", "SGEN",
        "RARE", "ACAD", "INCY", "EXEL", "BMRN", "SRPT", "BEAM",
    ],
    "momentum": [
        "NVDA", "META", "TSLA", "PLTR", "SMCI", "ARM", "MSTR", "COIN",
        "CRWD", "NET", "DDOG", "SNOW", "SHOP", "UBER", "LYFT", "RBLX",
        "SOFI", "HOOD", "IONQ", "RGTI", "QUBT",
    ],
}

DEFAULT_WATCHLIST = "sp500_large"


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    ticker:         str
    price:          float
    score:          float       # signed: + breakout up, - breakdown
    direction:      str         # "breakout_up" | "breakdown" | "trending_up" | "trending_down" | "neutral"
    strength:       str         # "strong" | "moderate" | "weak"
    regime:         str
    signal_label:   str
    confidence:     float       # 0..1
    prob_up:        Optional[float]   # from MC if computed
    prob_down:      Optional[float]
    rsi:            float
    adx:            float
    macd_hist:      float
    bb_position:    float
    obv_slope:      float
    atr_pct:        float
    vol_regime:     str
    donchian_pos:   float
    hurst:          float
    trend_score:    float
    potential_up:   float
    potential_down: float
    rsi_divergence: float
    ema200_dist:    float
    price_vs_52w:   float
    reasoning:      str
    error:          Optional[str] = None
    elapsed_ms:     float = 0.0


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _compute_scan_score(reg, sig, ind) -> float:
    """
    Composite breakout/breakdown score in [-1, 1].
    Combines regime, signal, and key indicator reads.
    Positive → breakout up; Negative → breakdown.
    """
    score = 0.0

    # 1. Regime direction (dominant, 40% weight)
    regime_map = {
        "breakout_up":      +1.0,
        "strong_uptrend":   +0.75,
        "weak_uptrend":     +0.40,
        "breakout_down":    -1.0,
        "strong_downtrend": -0.75,
        "weak_downtrend":   -0.40,
        "range_bound":       0.0,
        "choppy":            0.0,
    }
    score += regime_map.get(reg.regime, 0.0) * 0.40

    # 2. Signal composite (30% weight)
    score += float(np.clip(sig.composite * sig.confidence, -1.0, 1.0)) * 0.30

    # 3. Donchian position (breakout proximity, 15% weight)
    score += float(np.clip(reg.donchian_pos, -1.0, 1.0)) * 0.15

    # 4. OBV slope — volume confirmation (10% weight)
    score += float(np.clip(ind.obv_slope / 1.0, -1.0, 1.0)) * 0.10

    # 5. RSI extremes — confirmation or fade (5% weight)
    rsi_score = 0.0
    if ind.rsi > 70:   rsi_score = -0.3   # overbought → mild fade
    elif ind.rsi < 30: rsi_score =  0.3   # oversold → mild bounce
    score += rsi_score * 0.05

    return float(np.clip(score, -1.0, 1.0))


def _classify_direction(score: float, regime: str) -> tuple[str, str]:
    """Returns (direction, strength)."""
    abs_s = abs(score)

    if regime in ("breakout_up", "breakout_down"):
        strength = "strong" if abs_s > 0.5 else "moderate"
    elif abs_s > 0.55:
        strength = "strong"
    elif abs_s > 0.30:
        strength = "moderate"
    else:
        strength = "weak"

    if score > 0.20:
        if regime == "breakout_up":
            direction = "breakout_up"
        elif "uptrend" in regime:
            direction = "trending_up"
        else:
            direction = "bullish"
    elif score < -0.20:
        if regime == "breakout_down":
            direction = "breakdown"
        elif "downtrend" in regime:
            direction = "trending_down"
        else:
            direction = "bearish"
    else:
        direction = "neutral"

    return direction, strength


# ─── Single-ticker scan ───────────────────────────────────────────────────────

async def _scan_one(
    ticker: str,
    interval: str,
    lookback: int,
    extended: bool,
    loop: asyncio.AbstractEventLoop,
) -> ScanResult:
    t0 = time.monotonic()
    try:
        df = await loop.run_in_executor(
            None, fetch_candles, ticker, interval, lookback, extended
        )
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(
            None, detect_regime, df, ind.adx, ind.obv_slope
        )
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)

        price = float(df["close"].iloc[-1])
        score = _compute_scan_score(reg, sig, ind)
        direction, strength = _classify_direction(score, reg.regime)

        elapsed = (time.monotonic() - t0) * 1000

        return ScanResult(
            ticker         = ticker,
            price          = round(price, 4),
            score          = round(score, 4),
            direction      = direction,
            strength       = strength,
            regime         = reg.regime,
            signal_label   = sig.label,
            confidence     = round(sig.confidence, 3),
            prob_up        = None,
            prob_down      = None,
            rsi            = ind.rsi,
            adx            = ind.adx,
            macd_hist      = ind.macd_hist,
            bb_position    = ind.bb_position,
            obv_slope      = ind.obv_slope,
            atr_pct        = ind.atr_pct,
            vol_regime     = ind.vol_regime,
            donchian_pos   = reg.donchian_pos,
            hurst          = reg.hurst,
            trend_score    = reg.trend_score,
            potential_up   = reg.potential_up,
            potential_down = reg.potential_down,
            rsi_divergence = ind.rsi_divergence,
            ema200_dist    = ind.ema200_dist,
            price_vs_52w   = ind.price_vs_52w,
            reasoning      = sig.reasoning,
            elapsed_ms     = round(elapsed, 1),
        )

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("[scanner] %s failed: %s", ticker, e)
        return ScanResult(
            ticker=ticker, price=0.0, score=0.0, direction="error", strength="weak",
            regime="unknown", signal_label="Error", confidence=0.0,
            prob_up=None, prob_down=None, rsi=50.0, adx=0.0, macd_hist=0.0,
            bb_position=0.0, obv_slope=0.0, atr_pct=0.0, vol_regime="unknown",
            donchian_pos=0.0, hurst=0.5, trend_score=0.0,
            potential_up=33.3, potential_down=33.3,
            rsi_divergence=0.0, ema200_dist=0.0, price_vs_52w=0.0,
            reasoning="", error=str(e), elapsed_ms=round(elapsed, 1),
        )


# ─── Public: scan watchlist ───────────────────────────────────────────────────

async def _run_mc_on_result(
    result: ScanResult,
    interval: str,
    lookback: int,
    extended: bool,
    loop: asyncio.AbstractEventLoop,
    n_sim: int = 500,
    n_forward: int = 10,
    mc_model: str = "garch",
) -> ScanResult:
    """
    Re-fetch candles for one ScanResult and run a fast MC (n_sim=500) to
    produce accurate prob_up / prob_down estimates. Mutates in place.
    Returns the updated ScanResult.
    """
    try:
        df  = await loop.run_in_executor(
            None, fetch_candles, result.ticker, interval, lookback, extended
        )
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(None, detect_regime, df, ind.adx, ind.obv_slope)
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)
        price = float(df["close"].iloc[-1])

        mc = await loop.run_in_executor(
            None, run_mc,
            price, sig, n_sim, n_forward, mc_model,
            ind.returns, ind.kurtosis,
        )
        result.prob_up   = round(float(mc.prob_up),   1)
        result.prob_down = round(float(mc.prob_down), 1)
    except Exception as e:
        logger.debug("[scanner MC] %s failed: %s", result.ticker, e)
    return result


async def scan_tickers(
    tickers:         List[str],
    interval:        str  = "1d",
    lookback:        int  = 60,
    extended:        bool = False,
    max_concurrent:  int  = None,   # defaults to cfg.scan_max_concurrent
    min_score_abs:   float = None,  # defaults to cfg.scan_min_score
    mc_top_n:        int  = 5,      # run fast MC on top N results (0 to disable)
    mc_n_sim:        int  = 500,    # simulations for top-N MC pass
    mc_n_forward:    int  = 10,     # forward candles for top-N MC
    mc_model:        str  = "garch",
) -> Dict[str, Any]:
    """
    Scan tickers concurrently. Returns a structured report with:
      - breakouts:  top bullish signals (score > 0)
      - breakdowns: top bearish signals (score < 0)
      - all:        full ranked list

    If mc_top_n > 0, a fast Monte Carlo is run on the top mc_top_n results
    (by absolute score) to produce accurate prob_up / prob_down estimates.
    """
    if max_concurrent is None:
        max_concurrent = cfg.scan_max_concurrent
    if min_score_abs is None:
        min_score_abs = cfg.scan_min_score
    loop = asyncio.get_running_loop()
    sem  = asyncio.Semaphore(max_concurrent)

    async def _throttled(t: str) -> ScanResult:
        async with sem:
            return await _scan_one(t, interval, lookback, extended, loop)

    t_start = time.monotonic()
    tasks   = [asyncio.create_task(_throttled(t)) for t in tickers]
    results = await asyncio.gather(*tasks)

    # Filter errors
    valid   = [r for r in results if r.direction != "error"]
    errors  = [r for r in results if r.direction == "error"]

    # Apply minimum score filter
    if min_score_abs > 0:
        valid = [r for r in valid if abs(r.score) >= min_score_abs]

    # Sort by score descending
    ranked = sorted(valid, key=lambda r: r.score, reverse=True)

    # ── Fast MC on top-N results ──────────────────────────────────────────
    if mc_top_n > 0 and ranked:
        # Pick top N by |score| (covers both strong breakouts and breakdowns)
        by_abs = sorted(ranked, key=lambda r: abs(r.score), reverse=True)
        top_n  = by_abs[:mc_top_n]
        mc_sem = asyncio.Semaphore(min(mc_top_n, 3))   # limit concurrency for MC

        async def _mc_throttled(r: ScanResult) -> ScanResult:
            async with mc_sem:
                return await _run_mc_on_result(
                    r, interval, lookback, extended, loop,
                    n_sim=mc_n_sim, n_forward=mc_n_forward, mc_model=mc_model,
                )

        mc_tasks = [asyncio.create_task(_mc_throttled(r)) for r in top_n]
        await asyncio.gather(*mc_tasks)
        # Results are mutated in place — ranked already references the same objects

    elapsed = round((time.monotonic() - t_start) * 1000)

    breakouts  = [r for r in ranked if r.score >  0.20]
    breakdowns = [r for r in ranked if r.score < -0.20]
    neutral    = [r for r in ranked if abs(r.score) <= 0.20]

    def _to_dict(r: ScanResult) -> dict:
        return asdict(r)

    return {
        "scanned":    len(tickers),
        "succeeded":  len(valid),
        "failed":     len(errors),
        "elapsed_ms": elapsed,
        "interval":   interval,
        "lookback":   lookback,
        "breakouts":  [_to_dict(r) for r in breakouts],
        "breakdowns": [_to_dict(r) for r in breakdowns],
        "neutral":    [_to_dict(r) for r in neutral],
        "all":        [_to_dict(r) for r in ranked],
        "errors":     [{"ticker": r.ticker, "error": r.error} for r in errors],
    }


def get_watchlist(name: str) -> List[str]:
    """Return a named watchlist (case-insensitive). Falls back to sp500_large."""
    return WATCHLISTS.get(name.lower(), WATCHLISTS[DEFAULT_WATCHLIST])
