"""Demand/supply zone and EMA strategy scanner."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import cfg
from core.analysis.indicators import compute_indicators
from core.analysis.regime import detect_regime
from core.analysis.signal import compute_signal
from core.analysis.trade_setup import compute_trade_setup
from core.analysis.trade_setup import to_dict as ts_to_dict
from core.analysis.zones import Zone, detect_zones
from core.data.fetcher import fetch_candles

logger = logging.getLogger(__name__)


def _ema_series(closes: np.ndarray, period: int) -> pd.Series:
    s = pd.Series(closes.astype(float))
    return s.ewm(span=period, adjust=False).mean()


def _ema_value(closes: np.ndarray, period: int) -> float:
    if len(closes) < period:
        return float(closes[-1]) if len(closes) else 0.0
    return float(_ema_series(closes, period).iloc[-1])


def _ema_cross(closes: np.ndarray, fast: int, slow: int, lookback: int = 5) -> str:
    """
    Detect recent EMA cross between `fast` and `slow` EMAs within last `lookback` bars.
    Returns: "cross_up" | "cross_down" | "none"
    """
    if len(closes) < slow + lookback:
        return "none"
    fast_s = _ema_series(closes, fast)
    slow_s = _ema_series(closes, slow)
    # Check recent bars for a cross
    for i in range(-1, -(lookback + 1), -1):
        try:
            curr_above = fast_s.iloc[i] > slow_s.iloc[i]
            prev_above = fast_s.iloc[i - 1] > slow_s.iloc[i - 1]
        except IndexError:
            break
        if curr_above and not prev_above:
            return "cross_up"
        if not curr_above and prev_above:
            return "cross_down"
    return "none"


def _classify_ema_stack(price: float, ema20: float, ema50: float, ema200: float) -> str:
    """
    Returns one of:
      "bull_stack"  - price > EMA20 > EMA50 > EMA200
      "bear_stack"  - price < EMA20 < EMA50 < EMA200
      "above_200"   - price > EMA200 but EMAs not fully aligned bullish
      "below_200"   - price < EMA200 but EMAs not fully aligned bearish
      "mixed"       - neither
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


def _ema_score(ema_stack: str, side: str) -> float:
    """0-1 EMA alignment score for the given trade side."""
    if side == "long":
        return {"bull_stack": 1.0, "above_200": 0.6, "mixed": 0.3, "below_200": 0.1, "bear_stack": 0.0}[
            ema_stack
        ]
    else:
        return {"bear_stack": 1.0, "below_200": 0.6, "mixed": 0.3, "above_200": 0.1, "bull_stack": 0.0}[
            ema_stack
        ]


def _score_setup(
    zone_strength: float,
    ema_stack: str,
    side: str,
    rsi: float,
    obv_slope: float,
) -> float:
    # Zone quality (40%)
    z_score = zone_strength * 0.40

    # EMA alignment (40%)
    e_score = _ema_score(ema_stack, side) * 0.40

    # RSI position (10%) - longs want RSI 30-65, shorts want RSI 40-75
    if side == "long":
        rsi_ok = 1.0 if 30 <= rsi <= 65 else (0.5 if rsi < 30 else 0.2)
    else:
        rsi_ok = 1.0 if 40 <= rsi <= 75 else (0.5 if rsi > 75 else 0.2)
    r_score = rsi_ok * 0.10

    # OBV volume confirm (10%)
    obv_ok = 1.0 if (side == "long" and obv_slope > 0) or (side == "short" and obv_slope < 0) else 0.3
    v_score = obv_ok * 0.10

    return round(z_score + e_score + r_score + v_score, 3)


@dataclass
class ZoneScanResult:
    ticker: str
    price: float
    interval: str
    setup_score: float  # 0-1 overall quality
    setup_type: str  # "demand_bounce"|"demand_support"|"supply_break"|"supply_bounce"|"demand_break"|"none"
    side: str  # "long"|"short"|"none"

    # EMA values
    ema20: float
    ema50: float
    ema200: float
    ema_stack: str  # "bull_stack"|"bear_stack"|"above_200"|"below_200"|"mixed"

    # EMA cross signals (within last 5 bars)
    cross_20_50: str  # "cross_up"|"cross_down"|"none"
    cross_50_200: str  # "cross_up"|"cross_down"|"none"
    cross_20_200: str  # "cross_up"|"cross_down"|"none"

    # Price vs each EMA (%)
    dist_ema20: float  # (price - ema20) / price * 100
    dist_ema50: float
    dist_ema200: float

    # Nearest zones (structured for display)
    nearest_zone_level: float | None
    nearest_zone_strength: float | None
    nearest_zone_type: str | None  # "demand"|"supply"

    # All demand zones sorted nearest-to-price first (up to 3)
    demand_zones: list  # list of {level, strength, fresh, touches, label}
    supply_zones: list  # list of {level, strength, fresh, touches, label}

    zone_tp1: float | None
    zone_tp2: float | None
    zone_sl: float | None
    zone_tp1_dist: float | None
    zone_tp2_dist: float | None
    zone_sl_dist: float | None
    zone_rr: float | None

    # ATR-estimated MC targets (for scanner speed - no full MC run)
    mc_tp1: float | None
    mc_tp2: float | None

    rsi: float
    adx: float
    obv_slope: float
    atr_pct: float

    trade_setup: dict  # full TradeSetup dict (ATR-estimated)
    reason: str
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


async def _zone_scan_one(
    ticker: str,
    interval: str,
    lookback: int,
    extended: bool,
    loop: asyncio.AbstractEventLoop,
) -> ZoneScanResult:
    t0 = time.monotonic()

    _empty = ZoneScanResult(
        ticker=ticker,
        price=0.0,
        interval=interval,
        setup_score=0.0,
        setup_type="none",
        side="none",
        ema20=0.0,
        ema50=0.0,
        ema200=0.0,
        ema_stack="mixed",
        cross_20_50="none",
        cross_50_200="none",
        cross_20_200="none",
        dist_ema20=0.0,
        dist_ema50=0.0,
        dist_ema200=0.0,
        nearest_zone_level=None,
        nearest_zone_strength=None,
        nearest_zone_type=None,
        demand_zones=[],
        supply_zones=[],
        zone_tp1=None,
        zone_tp2=None,
        zone_sl=None,
        zone_tp1_dist=None,
        zone_tp2_dist=None,
        zone_sl_dist=None,
        zone_rr=None,
        mc_tp1=None,
        mc_tp2=None,
        rsi=50.0,
        adx=0.0,
        obv_slope=0.0,
        atr_pct=0.015,
        trade_setup={"valid": False, "side": "none", "reason": ""},
        reason="",
        elapsed_ms=0.0,
    )

    try:
        df = await loop.run_in_executor(None, fetch_candles, ticker, interval, lookback, extended)
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(None, detect_regime, df, ind.adx, ind.obv_slope)
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)

        closes = df["close"].to_numpy(float)
        price = float(closes[-1])

        ema20 = _ema_value(closes, 20)
        ema50 = _ema_value(closes, 50)
        ema200 = _ema_value(closes, 200)
        ema_stack = _classify_ema_stack(price, ema20, ema50, ema200)

        cross_20_50 = _ema_cross(closes, 20, 50, lookback=5)
        cross_50_200 = _ema_cross(closes, 50, 200, lookback=5)
        cross_20_200 = _ema_cross(closes, 20, 200, lookback=5)

        dist_ema20 = round((price - ema20) / price * 100, 2) if price else 0.0
        dist_ema50 = round((price - ema50) / price * 100, 2) if price else 0.0
        dist_ema200 = round((price - ema200) / price * 100, 2) if price else 0.0

        zones = await loop.run_in_executor(None, detect_zones, df)
        atr = zones.atr if zones.atr > 0 else price * 0.015
        touch_band = atr * 0.6

        # Build structured zone lists for display (nearest to price first, up to 3)
        def _zone_label(z, price: float) -> str:
            """Strong / Demand 1 / Demand 2 label based on strength."""
            if z.strength >= 0.70:
                return "Strong"
            return None  # caller assigns index

        def _build_zone_list(raw_zones, price: float, zone_kind: str) -> list:
            """Sort by proximity to price, label by strength rank."""
            if zone_kind == "demand":
                # Demand zones below price - sort by level desc (nearest first)
                candidates = sorted(raw_zones, key=lambda z: abs(price - z.level))[:3]
            else:
                # Supply zones above price - sort by level asc (nearest first)
                candidates = sorted(raw_zones, key=lambda z: abs(price - z.level))[:3]
            result = []
            idx = 1
            for z in candidates:
                if z.strength >= 0.70:
                    lbl = f"Strong {'Support' if zone_kind == 'demand' else 'Resistance'}"
                else:
                    lbl = f"{'Support' if zone_kind == 'demand' else 'Resistance'} {idx}"
                    idx += 1
                result.append(
                    {
                        "level": round(z.level, 4),
                        "low": round(z.low, 4),
                        "high": round(z.high, 4),
                        "strength": round(z.strength, 3),
                        "touches": z.touches,
                        "fresh": z.fresh,
                        "label": lbl,
                        "dist_pct": round((price - z.level) / price * 100, 2)
                        if zone_kind == "demand"
                        else round((z.level - price) / price * 100, 2),
                    }
                )
            return result

        demand_zones_list = _build_zone_list(zones.demand_zones, price, "demand")
        supply_zones_list = _build_zone_list(zones.supply_zones, price, "supply")

        setup_type = "none"
        side = "none"
        trig_zone: Zone | None = None
        zone_tp1 = zone_tp2 = zone_sl = None
        zone_tp1_dist = zone_tp2_dist = zone_sl_dist = zone_rr = None

        nd = zones.nearest_demand
        ns = zones.nearest_supply

        # Priority: full stack > partial > nothing
        long_ok = ema_stack in ("bull_stack", "above_200")
        short_ok = ema_stack in ("bear_stack", "below_200")

        # Helper: supply TPs above price
        def _supply_tps(above_level: float):
            sups = sorted([z for z in zones.supply_zones if z.level > above_level], key=lambda z: z.level)
            tp1 = round(sups[0].level, 4) if sups else round(price + atr * 3, 4)
            tp2 = round(sups[1].level, 4) if len(sups) > 1 else round(tp1 * 1.02, 4)
            return tp1, tp2

        # LONG A: demand bounce - price AT demand zone, bullish EMA stack
        # Requires price still ABOVE zone center (not broken below).
        if long_ok and nd and price >= nd.level and (price - nd.level) <= touch_band:
            setup_type = "demand_bounce"
            side = "long"
            trig_zone = nd
            zone_tp1, zone_tp2 = _supply_tps(price)
            zone_sl = round(nd.low - atr * 0.25, 4)

        # LONG A2: demand support - price is ABOVE the zone center (zone not broken)
        # and within 2×ATR of demand. Fires regardless of EMA direction because
        # the key fact is that price has NOT broken below the zone.
        # Also catches the touch_band case when EMA is not fully bullish.
        elif nd and price >= nd.level and (price - nd.level) <= atr * 2.0:
            setup_type = "demand_support"
            side = "long"
            trig_zone = nd
            zone_tp1, zone_tp2 = _supply_tps(price)
            zone_sl = round(nd.low - atr * 0.25, 4)

        # LONG B: supply break (price just crossed above supply -> flip to demand)
        elif long_ok and ns and ns.level <= price * 1.01 and ns.level >= price * 0.98:
            setup_type = "supply_break"
            side = "long"
            trig_zone = ns
            sups = sorted([z for z in zones.supply_zones if z.level > ns.level + atr], key=lambda z: z.level)
            zone_tp1 = round(sups[0].level, 4) if sups else round(price + atr * 3, 4)
            zone_tp2 = round(sups[1].level, 4) if len(sups) > 1 else round(zone_tp1 * 1.02, 4)
            zone_sl = round(ns.low - atr * 0.25, 4)

        # SHORT A: supply bounce
        elif short_ok and ns and abs(price - ns.level) <= touch_band:
            setup_type = "supply_bounce"
            side = "short"
            trig_zone = ns
            dems = sorted(
                [z for z in zones.demand_zones if z.level < price], key=lambda z: z.level, reverse=True
            )
            zone_tp1 = round(dems[0].level, 4) if dems else round(price - atr * 3, 4)
            zone_tp2 = round(dems[1].level, 4) if len(dems) > 1 else round(zone_tp1 * 0.98, 4)
            zone_sl = round(ns.high + atr * 0.25, 4)

        # SHORT B: demand break - price has actually broken BELOW the demand zone centre.
        # nd.level > price  means the zone centre is now above current price = confirmed break.
        # Guard: zone must be within 2% above price (recent break, not a distant one).
        elif short_ok and nd and nd.level > price and nd.level <= price * 1.02:
            setup_type = "demand_break"
            side = "short"
            trig_zone = nd
            dems = sorted(
                [z for z in zones.demand_zones if z.level < nd.level - atr],
                key=lambda z: z.level,
                reverse=True,
            )
            zone_tp1 = round(dems[0].level, 4) if dems else round(price - atr * 3, 4)
            zone_tp2 = round(dems[1].level, 4) if len(dems) > 1 else round(zone_tp1 * 0.98, 4)
            zone_sl = round(nd.high + atr * 0.25, 4)

        if zone_tp1 is not None and zone_sl is not None:
            if side == "long":
                zone_tp1_dist = round((zone_tp1 - price) / price * 100, 2)
                zone_tp2_dist = round((zone_tp2 - price) / price * 100, 2) if zone_tp2 else None
                zone_sl_dist = round((price - zone_sl) / price * 100, 2)
            else:
                zone_tp1_dist = round((price - zone_tp1) / price * 100, 2)
                zone_tp2_dist = round((price - zone_tp2) / price * 100, 2) if zone_tp2 else None
                zone_sl_dist = round((zone_sl - price) / price * 100, 2)
            risk = abs(price - zone_sl)
            reward = abs(zone_tp1 - price)
            zone_rr = round(reward / risk, 2) if risk > 0 else None

        zone_strength = trig_zone.strength if trig_zone else 0.0
        setup_score = (
            _score_setup(zone_strength, ema_stack, side, ind.rsi, ind.obv_slope) if side != "none" else 0.0
        )

        atr_dollar = price * ind.atr_pct
        if side == "long":
            mc_tp1 = round(price + atr_dollar * 1.5, 4)
            mc_tp2 = round(price + atr_dollar * 2.5, 4)
        elif side == "short":
            mc_tp1 = round(price - atr_dollar * 1.5, 4)
            mc_tp2 = round(price - atr_dollar * 2.5, 4)
        else:
            mc_tp1 = mc_tp2 = None

        ts_score = abs(sig.composite) if side != "none" else 0.0
        if "down" in setup_type or side == "short":
            ts_score = -abs(ts_score)

        ts = compute_trade_setup(
            ticker=ticker,
            price=price,
            score=ts_score
            if ts_score != 0.0
            else (0.25 if side == "long" else -0.25 if side == "short" else 0.0),
            direction=setup_type if setup_type != "none" else reg.regime,
            regime=reg.regime,
            confidence=sig.confidence,
            rsi=ind.rsi,
            adx=ind.adx,
            atr_pct=ind.atr_pct,
            prob_up=0.0,
            prob_down=0.0,
            # Pass zone levels as MC percentiles - zones ARE the targets
            mc_p10=zone_tp2 if side == "short" else None,
            mc_p25=zone_tp1 if side == "short" else None,
            mc_p75=zone_tp1 if side == "long" else None,
            mc_p90=zone_tp2 if side == "long" else None,
            interval=interval,
            df=df,
        )
        trade_setup_dict = ts_to_dict(ts)

        setup_labels = {
            "demand_bounce": "↑ Demand Bounce",
            "demand_support": "↑ Demand Support",
            "supply_break": "↑ Supply Break",
            "supply_bounce": "↓ Supply Reject",
            "demand_break": "↓ Demand Break",
        }
        ema_labels = {
            "bull_stack": "Full Bull Stack",
            "bear_stack": "Full Bear Stack",
            "above_200": "Above EMA200",
            "below_200": "Below EMA200",
            "mixed": "Mixed EMAs",
        }
        if side != "none":
            reason = (
                f"{setup_labels.get(setup_type, setup_type)}, "
                f"{ema_labels.get(ema_stack, ema_stack)}, "
                f"zone str {zone_strength:.0%}, "
                f"setup score {setup_score:.0%}, "
                f"RSI {ind.rsi:.0f}, ADX {ind.adx:.0f}"
            )
        else:
            reason = f"No zone setup; {ema_labels.get(ema_stack, '?')}, price not near any zone"

        elapsed = (time.monotonic() - t0) * 1000

        return ZoneScanResult(
            ticker=ticker,
            price=round(price, 4),
            interval=interval,
            setup_score=setup_score,
            setup_type=setup_type,
            side=side,
            ema20=round(ema20, 4),
            ema50=round(ema50, 4),
            ema200=round(ema200, 4),
            ema_stack=ema_stack,
            cross_20_50=cross_20_50,
            cross_50_200=cross_50_200,
            cross_20_200=cross_20_200,
            dist_ema20=dist_ema20,
            dist_ema50=dist_ema50,
            dist_ema200=dist_ema200,
            nearest_zone_level=trig_zone.level if trig_zone else None,
            nearest_zone_strength=trig_zone.strength if trig_zone else None,
            nearest_zone_type=trig_zone.zone_type if trig_zone else None,
            demand_zones=demand_zones_list,
            supply_zones=supply_zones_list,
            zone_tp1=zone_tp1,
            zone_tp2=zone_tp2,
            zone_sl=zone_sl,
            zone_tp1_dist=zone_tp1_dist,
            zone_tp2_dist=zone_tp2_dist,
            zone_sl_dist=zone_sl_dist,
            zone_rr=zone_rr,
            mc_tp1=mc_tp1,
            mc_tp2=mc_tp2,
            rsi=round(ind.rsi, 1),
            adx=round(ind.adx, 1),
            obv_slope=round(ind.obv_slope, 4),
            atr_pct=round(ind.atr_pct, 4),
            trade_setup=trade_setup_dict,
            reason=reason,
            elapsed_ms=round(elapsed, 1),
        )

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("[zone_scanner] %s failed: %s", ticker, e)
        err = _empty
        err.ticker = ticker
        err.error = str(e)
        err.reason = f"Error: {e}"
        err.elapsed_ms = round(elapsed, 1)
        return err


async def zone_scan_tickers(
    tickers: list[str],
    interval: str = "1d",
    lookback: int = 100,  # need more bars for EMA200
    extended: bool = False,
    max_concurrent: int | None = None,  # defaults to cfg.scan_max_concurrent
    min_score: float = 0.0,
) -> dict:
    """
    Scan tickers for demand/supply zone + EMA setups.

    Returns dict with keys:
      longs, shorts, no_setup, all, meta
    """
    if max_concurrent is None:
        max_concurrent = cfg.scan_max_concurrent
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)

    async def _throttled(t: str) -> ZoneScanResult:
        async with sem:
            return await _zone_scan_one(t, interval, lookback, extended, loop)

    results = await asyncio.gather(*[_throttled(t) for t in tickers])

    longs = []
    shorts = []
    no_setup = []

    for r in results:
        if r.error:
            continue
        d = r.to_dict()
        if r.side == "long" and r.setup_score >= min_score:
            longs.append(d)
        elif r.side == "short" and r.setup_score >= min_score:
            shorts.append(d)
        else:
            no_setup.append(d)

    # Sort by setup_score desc
    longs.sort(key=lambda x: x["setup_score"], reverse=True)
    shorts.sort(key=lambda x: x["setup_score"], reverse=True)
    no_setup.sort(key=lambda x: x["setup_score"], reverse=True)
    all_results = sorted(longs + shorts + no_setup, key=lambda x: x["setup_score"], reverse=True)

    return {
        "longs": longs,
        "shorts": shorts,
        "no_setup": no_setup,
        "all": all_results,
        "meta": {
            "total": len(results),
            "longs": len(longs),
            "shorts": len(shorts),
            "no_setup": len(no_setup),
            "interval": interval,
        },
    }
