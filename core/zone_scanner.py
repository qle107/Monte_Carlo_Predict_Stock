"""
core/zone_scanner.py — Demand/Supply Zone + EMA Strategy Scanner
─────────────────────────────────────────────────────────────────
Strategy logic
──────────────
For each ticker:

1.  Fetch OHLCV and compute EMA 20 / 50 / 200.
2.  Detect demand and supply zones (from core.zones).
3.  Classify the EMA stack:
      "bull_stack"   — price > EMA20 > EMA50 > EMA200  (full bull alignment)
      "bear_stack"   — price < EMA20 < EMA50 < EMA200  (full bear alignment)
      "above_200"    — price > EMA200 only (partial)
      "below_200"    — price < EMA200 (bearish bias)
      "mixed"        — neither pattern
4.  Find actionable setups (4 scenarios):
      LONG  A: price at/near demand zone  AND bull_stack / above_200
      LONG  B: price breaks above supply  AND bull_stack / above_200
      SHORT A: price at/near supply zone  AND bear_stack / below_200
      SHORT B: price breaks below demand  AND bear_stack / below_200
5.  Score each setup (0–1):
      • Zone strength  (0.4 weight)
      • EMA alignment  (0.4 weight) — full stack vs partial
      • RSI position   (0.1 weight) — not overbought for longs / not oversold for shorts
      • Volume confirm (0.1 weight) — OBV slope sign
6.  Compute MC-based trade setup (reuses trade_setup_from_analysis or ATR estimate).
7.  Return sorted by setup_score desc.

ZoneScanResult fields
─────────────────────
  ticker, price, interval, setup_score, setup_type, side
  ema20, ema50, ema200, ema_stack
  nearest_zone_level, nearest_zone_strength, nearest_zone_type
  zone_tp1, zone_tp2, zone_sl, zone_rr
  mc_tp1, mc_tp2      — populated only when full MC is run (Load ↗)
  rsi, adx, obv_slope, atr_pct
  trade_setup         — full TradeSetup dict (ATR-estimated TPs for scanner speed)
  reason              — plain-English summary
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from .fetcher     import fetch_candles
from .indicators  import compute_indicators
from .regime      import detect_regime
from .signal      import compute_signal
from .zones       import detect_zones, ZoneResult, Zone
from .trade_setup import compute_trade_setup, to_dict as ts_to_dict

logger = logging.getLogger(__name__)


# ─── EMA helpers ─────────────────────────────────────────────────────────────

def _ema_value(closes: np.ndarray, period: int) -> float:
    if len(closes) < period:
        return float(closes[-1]) if len(closes) else 0.0
    s = pd.Series(closes.astype(float))
    return float(s.ewm(span=period, adjust=False).mean().iloc[-1])


def _classify_ema_stack(price: float, ema20: float, ema50: float, ema200: float) -> str:
    """
    Returns one of:
      "bull_stack"  — price > EMA20 > EMA50 > EMA200
      "bear_stack"  — price < EMA20 < EMA50 < EMA200
      "above_200"   — price > EMA200 but EMAs not fully aligned bullish
      "below_200"   — price < EMA200 but EMAs not fully aligned bearish
      "mixed"       — neither
    """
    bull = price > ema20 > ema50 > ema200
    bear = price < ema20 < ema50 < ema200
    if bull:
        return "bull_stack"
    if bear:
        return "bear_stack"
    if price > ema200:
        return "above_200"
    return "below_200"


# ─── Setup scoring ────────────────────────────────────────────────────────────

def _ema_score(ema_stack: str, side: str) -> float:
    """0–1 EMA alignment score for the given trade side."""
    if side == "long":
        return {"bull_stack": 1.0, "above_200": 0.6, "mixed": 0.3, "below_200": 0.1, "bear_stack": 0.0}[ema_stack]
    else:
        return {"bear_stack": 1.0, "below_200": 0.6, "mixed": 0.3, "above_200": 0.1, "bull_stack": 0.0}[ema_stack]


def _score_setup(
    zone_strength: float,
    ema_stack:     str,
    side:          str,
    rsi:           float,
    obv_slope:     float,
) -> float:
    # Zone quality (40%)
    z_score = zone_strength * 0.40

    # EMA alignment (40%)
    e_score = _ema_score(ema_stack, side) * 0.40

    # RSI position (10%) — longs want RSI 30–65, shorts want RSI 40–75
    if side == "long":
        rsi_ok = 1.0 if 30 <= rsi <= 65 else (0.5 if rsi < 30 else 0.2)
    else:
        rsi_ok = 1.0 if 40 <= rsi <= 75 else (0.5 if rsi > 75 else 0.2)
    r_score = rsi_ok * 0.10

    # OBV volume confirm (10%)
    obv_ok = 1.0 if (side == "long" and obv_slope > 0) or (side == "short" and obv_slope < 0) else 0.3
    v_score = obv_ok * 0.10

    return round(z_score + e_score + r_score + v_score, 3)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ZoneScanResult:
    ticker:        str
    price:         float
    interval:      str
    setup_score:   float       # 0–1 overall quality
    setup_type:    str         # "demand_bounce"|"supply_break"|"supply_bounce"|"demand_break"|"none"
    side:          str         # "long"|"short"|"none"

    ema20:         float
    ema50:         float
    ema200:        float
    ema_stack:     str

    nearest_zone_level:    Optional[float]
    nearest_zone_strength: Optional[float]
    nearest_zone_type:     Optional[str]   # "demand"|"supply"

    zone_tp1:      Optional[float]
    zone_tp2:      Optional[float]
    zone_sl:       Optional[float]
    zone_tp1_dist: Optional[float]
    zone_tp2_dist: Optional[float]
    zone_sl_dist:  Optional[float]
    zone_rr:       Optional[float]

    # ATR-estimated MC targets (for scanner speed — no full MC run)
    mc_tp1:        Optional[float]
    mc_tp2:        Optional[float]

    rsi:           float
    adx:           float
    obv_slope:     float
    atr_pct:       float

    trade_setup:   dict        # full TradeSetup dict (ATR-estimated)
    reason:        str
    error:         Optional[str] = None
    elapsed_ms:    float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Single ticker zone scan ──────────────────────────────────────────────────

async def _zone_scan_one(
    ticker:   str,
    interval: str,
    lookback: int,
    extended: bool,
    loop:     asyncio.AbstractEventLoop,
) -> ZoneScanResult:
    t0 = time.monotonic()

    _empty = ZoneScanResult(
        ticker=ticker, price=0.0, interval=interval,
        setup_score=0.0, setup_type="none", side="none",
        ema20=0.0, ema50=0.0, ema200=0.0, ema_stack="mixed",
        nearest_zone_level=None, nearest_zone_strength=None, nearest_zone_type=None,
        zone_tp1=None, zone_tp2=None, zone_sl=None,
        zone_tp1_dist=None, zone_tp2_dist=None, zone_sl_dist=None, zone_rr=None,
        mc_tp1=None, mc_tp2=None,
        rsi=50.0, adx=0.0, obv_slope=0.0, atr_pct=0.015,
        trade_setup={"valid": False, "side": "none", "reason": ""},
        reason="", elapsed_ms=0.0,
    )

    try:
        # ── 1. Fetch + indicators ────────────────────────────────────
        df  = await loop.run_in_executor(None, fetch_candles, ticker, interval, lookback, extended)
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(None, detect_regime, df, ind.adx, ind.obv_slope)
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)

        closes = df["close"].to_numpy(float)
        price  = float(closes[-1])

        # ── 2. EMA 20 / 50 / 200 ────────────────────────────────────
        ema20  = _ema_value(closes, 20)
        ema50  = _ema_value(closes, 50)
        ema200 = _ema_value(closes, 200)
        ema_stack = _classify_ema_stack(price, ema20, ema50, ema200)

        # ── 3. Zone detection ────────────────────────────────────────
        zones = await loop.run_in_executor(None, detect_zones, df)
        atr   = zones.atr if zones.atr > 0 else price * 0.015
        touch_band = atr * 0.6

        # ── 4. Find best setup ───────────────────────────────────────
        setup_type  = "none"
        side        = "none"
        trig_zone: Optional[Zone] = None
        zone_tp1 = zone_tp2 = zone_sl = None
        zone_tp1_dist = zone_tp2_dist = zone_sl_dist = zone_rr = None

        nd = zones.nearest_demand
        ns = zones.nearest_supply
        ctx = zones.price_context

        # Priority: full stack > partial > nothing
        long_ok  = ema_stack in ("bull_stack", "above_200")
        short_ok = ema_stack in ("bear_stack", "below_200")

        # LONG A: demand bounce
        if long_ok and nd and abs(price - nd.level) <= touch_band:
            setup_type = "demand_bounce"
            side = "long"
            trig_zone = nd
            # TP = supply zones above
            sups = sorted([z for z in zones.supply_zones if z.level > price], key=lambda z: z.level)
            zone_tp1 = round(sups[0].level, 4) if len(sups) > 0 else round(price + atr * 3, 4)
            zone_tp2 = round(sups[1].level, 4) if len(sups) > 1 else round(zone_tp1 * 1.02, 4)
            zone_sl  = round(nd.low - atr * 0.25, 4)

        # LONG B: supply break (price just crossed above supply → flip to demand)
        elif long_ok and ns and ns.level <= price * 1.01 and ns.level >= price * 0.98:
            setup_type = "supply_break"
            side = "long"
            trig_zone = ns
            sups = sorted([z for z in zones.supply_zones if z.level > ns.level + atr], key=lambda z: z.level)
            zone_tp1 = round(sups[0].level, 4) if len(sups) > 0 else round(price + atr * 3, 4)
            zone_tp2 = round(sups[1].level, 4) if len(sups) > 1 else round(zone_tp1 * 1.02, 4)
            zone_sl  = round(ns.low - atr * 0.25, 4)

        # SHORT A: supply bounce
        elif short_ok and ns and abs(price - ns.level) <= touch_band:
            setup_type = "supply_bounce"
            side = "short"
            trig_zone = ns
            dems = sorted([z for z in zones.demand_zones if z.level < price], key=lambda z: z.level, reverse=True)
            zone_tp1 = round(dems[0].level, 4) if len(dems) > 0 else round(price - atr * 3, 4)
            zone_tp2 = round(dems[1].level, 4) if len(dems) > 1 else round(zone_tp1 * 0.98, 4)
            zone_sl  = round(ns.high + atr * 0.25, 4)

        # SHORT B: demand break (price just broke below demand)
        elif short_ok and nd and nd.level >= price * 0.99 and nd.level <= price * 1.02:
            setup_type = "demand_break"
            side = "short"
            trig_zone = nd
            dems = sorted([z for z in zones.demand_zones if z.level < nd.level - atr], key=lambda z: z.level, reverse=True)
            zone_tp1 = round(dems[0].level, 4) if len(dems) > 0 else round(price - atr * 3, 4)
            zone_tp2 = round(dems[1].level, 4) if len(dems) > 1 else round(zone_tp1 * 0.98, 4)
            zone_sl  = round(nd.high + atr * 0.25, 4)

        # ── 5. Compute zone distances + R:R ─────────────────────────
        if zone_tp1 is not None and zone_sl is not None:
            if side == "long":
                zone_tp1_dist = round((zone_tp1 - price) / price * 100, 2)
                zone_tp2_dist = round((zone_tp2 - price) / price * 100, 2) if zone_tp2 else None
                zone_sl_dist  = round((price - zone_sl)  / price * 100, 2)
            else:
                zone_tp1_dist = round((price - zone_tp1) / price * 100, 2)
                zone_tp2_dist = round((price - zone_tp2) / price * 100, 2) if zone_tp2 else None
                zone_sl_dist  = round((zone_sl - price)  / price * 100, 2)
            risk   = abs(price - zone_sl)
            reward = abs(zone_tp1 - price)
            zone_rr = round(reward / risk, 2) if risk > 0 else None

        # ── 6. Setup score ───────────────────────────────────────────
        zone_strength = trig_zone.strength if trig_zone else 0.0
        setup_score   = _score_setup(zone_strength, ema_stack, side, ind.rsi, ind.obv_slope) \
                        if side != "none" else 0.0

        # ── 7. ATR-estimated MC TP (no full MC run for speed) ────────
        atr_dollar = price * ind.atr_pct
        if side == "long":
            mc_tp1 = round(price + atr_dollar * 1.5, 4)
            mc_tp2 = round(price + atr_dollar * 2.5, 4)
        elif side == "short":
            mc_tp1 = round(price - atr_dollar * 1.5, 4)
            mc_tp2 = round(price - atr_dollar * 2.5, 4)
        else:
            mc_tp1 = mc_tp2 = None

        # ── 8. Trade setup (ATR-estimated, no full MC) ───────────────
        ts_score = abs(sig.composite) if side != "none" else 0.0
        if "down" in setup_type or side == "short":
            ts_score = -abs(ts_score)

        ts = compute_trade_setup(
            ticker     = ticker,
            price      = price,
            score      = ts_score if ts_score != 0.0 else (0.25 if side == "long" else -0.25 if side == "short" else 0.0),
            direction  = setup_type if setup_type != "none" else reg.regime,
            regime     = reg.regime,
            confidence = sig.confidence,
            rsi        = ind.rsi,
            adx        = ind.adx,
            atr_pct    = ind.atr_pct,
            prob_up    = 0.0,
            prob_down  = 0.0,
            # Pass zone levels as MC percentiles — zones ARE the targets
            mc_p10     = zone_tp2 if side == "short" else None,
            mc_p25     = zone_tp1 if side == "short" else None,
            mc_p75     = zone_tp1 if side == "long"  else None,
            mc_p90     = zone_tp2 if side == "long"  else None,
            interval   = interval,
            df         = df,
        )
        trade_setup_dict = ts_to_dict(ts)

        # ── 9. Reason string ─────────────────────────────────────────
        setup_labels = {
            "demand_bounce": "↑ Demand Bounce",
            "supply_break":  "↑ Supply Break",
            "supply_bounce": "↓ Supply Reject",
            "demand_break":  "↓ Demand Break",
        }
        ema_labels = {
            "bull_stack": "Full Bull Stack",
            "bear_stack": "Full Bear Stack",
            "above_200":  "Above EMA200",
            "below_200":  "Below EMA200",
            "mixed":      "Mixed EMAs",
        }
        if side != "none":
            reason = (
                f"{setup_labels.get(setup_type, setup_type)} · "
                f"{ema_labels.get(ema_stack, ema_stack)} · "
                f"Zone str {zone_strength:.0%} · "
                f"Setup score {setup_score:.0%} · "
                f"RSI {ind.rsi:.0f} · ADX {ind.adx:.0f}"
            )
        else:
            reason = f"No zone setup — {ema_labels.get(ema_stack,'?')} · price not near any zone"

        elapsed = (time.monotonic() - t0) * 1000

        return ZoneScanResult(
            ticker        = ticker,
            price         = round(price, 4),
            interval      = interval,
            setup_score   = setup_score,
            setup_type    = setup_type,
            side          = side,
            ema20         = round(ema20,  4),
            ema50         = round(ema50,  4),
            ema200        = round(ema200, 4),
            ema_stack     = ema_stack,
            nearest_zone_level    = trig_zone.level    if trig_zone else None,
            nearest_zone_strength = trig_zone.strength if trig_zone else None,
            nearest_zone_type     = trig_zone.zone_type if trig_zone else None,
            zone_tp1      = zone_tp1,
            zone_tp2      = zone_tp2,
            zone_sl       = zone_sl,
            zone_tp1_dist = zone_tp1_dist,
            zone_tp2_dist = zone_tp2_dist,
            zone_sl_dist  = zone_sl_dist,
            zone_rr       = zone_rr,
            mc_tp1        = mc_tp1,
            mc_tp2        = mc_tp2,
            rsi           = round(ind.rsi, 1),
            adx           = round(ind.adx, 1),
            obv_slope     = round(ind.obv_slope, 4),
            atr_pct       = round(ind.atr_pct, 4),
            trade_setup   = trade_setup_dict,
            reason        = reason,
            elapsed_ms    = round(elapsed, 1),
        )

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("[zone_scanner] %s failed: %s", ticker, e)
        err = _empty
        err.ticker     = ticker
        err.error      = str(e)
        err.reason     = f"Error: {e}"
        err.elapsed_ms = round(elapsed, 1)
        return err


# ─── Public: scan watchlist for zone setups ──────────────────────────────────

async def zone_scan_tickers(
    tickers:       List[str],
    interval:      str  = "1d",
    lookback:      int  = 100,   # need more bars for EMA200
    extended:      bool = False,
    max_concurrent: int = 8,
    min_score:     float = 0.0,
) -> dict:
    """
    Scan tickers for demand/supply zone + EMA setups.

    Returns dict with keys:
      longs, shorts, no_setup, all, meta
    """
    loop = asyncio.get_event_loop()
    sem  = asyncio.Semaphore(max_concurrent)

    async def _throttled(t: str) -> ZoneScanResult:
        async with sem:
            return await _zone_scan_one(t, interval, lookback, extended, loop)

    results = await asyncio.gather(*[_throttled(t) for t in tickers])

    longs   = []
    shorts  = []
    no_setup = []

    for r in results:
        if r.error:
            continue
        d = r.to_dict()
        if r.side == "long"  and r.setup_score >= min_score:
            longs.append(d)
        elif r.side == "short" and r.setup_score >= min_score:
            shorts.append(d)
        else:
            no_setup.append(d)

    # Sort by setup_score desc
    longs.sort(  key=lambda x: x["setup_score"], reverse=True)
    shorts.sort( key=lambda x: x["setup_score"], reverse=True)
    no_setup.sort(key=lambda x: x["setup_score"], reverse=True)
    all_results = sorted(longs + shorts + no_setup, key=lambda x: x["setup_score"], reverse=True)

    return {
        "longs":    longs,
        "shorts":   shorts,
        "no_setup": no_setup,
        "all":      all_results,
        "meta": {
            "total":    len(results),
            "longs":    len(longs),
            "shorts":   len(shorts),
            "no_setup": len(no_setup),
            "interval": interval,
        },
    }
