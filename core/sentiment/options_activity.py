"""Options flow snapshot and options-activity sentiment scoring (yfinance, blocking)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import yfinance as yf

from core.data.yf_client import yf_call

logger = logging.getLogger(__name__)

# Options flow  (yfinance - blocking -> run in executor)


def _scan_unusual_activity(
    calls_df,
    puts_df,
    exp_str: str,
    today,
    min_vol: int = 500,
    vol_oi_mult: float = 1.5,
) -> list[dict]:
    """
    Identify contracts where volume > open_interest × vol_oi_mult AND volume > min_vol.

    Returns a list of dicts shaped for the Unusual Activity table:
    {expiry, dte, strike, type, volume, oi, vol_oi, premium, flow, pct_change,
     iv, in_money, leaps}.

    `leaps` is True when DTE > 60 - surfaced as a separate group in the UI.

    yfinance has no aggressor side; UI shows a flow disclaimer.
    """
    out: list[dict] = []
    try:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = max((exp_dt - today).days, 0)
    except Exception:
        dte = 0
    is_leaps = dte > 60

    for type_str, df in (("call", calls_df), ("put", puts_df)):
        if df is None or df.empty:
            continue
        d = df.copy()
        d["volume"] = d["volume"].fillna(0).astype(int)
        d["openInterest"] = d["openInterest"].fillna(0).astype(int)
        mask = (d["volume"] > min_vol) & (d["volume"] > d["openInterest"] * vol_oi_mult)
        rows = d[mask]
        for _, row in rows.iterrows():
            vol = int(row["volume"])
            oi = int(row["openInterest"])
            lp = float(row.get("lastPrice", 0) or 0)
            chg = float(row.get("percentChange", 0) or 0)
            iv = float(row.get("impliedVolatility", 0) or 0)
            ratio = round(vol / max(oi, 1), 2)
            premium = round(lp * 100, 2)  # cost per contract (USD)
            flow = round(lp * 100 * vol, 2)  # notional flow (USD)
            out.append(
                {
                    "expiry": exp_str,
                    "dte": dte,
                    "strike": float(row["strike"]),
                    "type": type_str,
                    "volume": vol,
                    "oi": oi,
                    "vol_oi": ratio,
                    "premium": premium,
                    "flow": flow,
                    "pct_change": round(chg, 2),
                    "iv": round(iv * 100, 1),
                    "in_money": bool(row.get("inTheMoney", False)),
                    "leaps": is_leaps,
                }
            )
    return out


def _options_flow_sync(ticker: str) -> dict:
    """
    Fetch the nearest 3 expiry dates' call + put volume / open interest.
    Also identifies the top call and put strikes by volume, plus a separate
    pass that scans up to 8 expirations for **unusual activity** (high
    volume relative to open interest - see _scan_unusual_activity).

    PCR < 0.7  -> call-heavy (bullish crowd positioning)
    0.7-1.0    -> neutral
    PCR > 1.0  -> put-heavy  (bearish / hedging)
    """
    try:
        t = yf.Ticker(ticker)
        expirations = yf_call(lambda: t.options)
        if not expirations:
            return {"available": False, "reason": "no options data"}

        n_exp = min(3, len(expirations))
        n_unusual_exp = min(8, len(expirations))  # deeper scan for LEAPS
        today_date = datetime.now(timezone.utc).date()
        total_call_vol = total_put_vol = 0
        total_call_oi = total_put_oi = 0
        chains_summary = []
        hot_calls: list[dict] = []
        hot_puts: list[dict] = []
        unusual_activity: list[dict] = []
        scanned_for_unusual: set = set()

        for exp in expirations[:n_exp]:
            try:
                chain = yf_call(t.option_chain, exp)
                cv = int(chain.calls["volume"].fillna(0).sum())
                pv = int(chain.puts["volume"].fillna(0).sum())
                co = int(chain.calls["openInterest"].fillna(0).sum())
                po = int(chain.puts["openInterest"].fillna(0).sum())

                total_call_vol += cv
                total_put_vol += pv
                total_call_oi += co
                total_put_oi += po

                chains_summary.append(
                    {
                        "expiry": exp,
                        "call_vol": cv,
                        "put_vol": pv,
                        "call_oi": co,
                        "put_oi": po,
                        "pcr_vol": round(pv / cv, 3) if cv else None,
                    }
                )

                # Top 5 call strikes by volume for this expiry
                calls_df = chain.calls.copy()
                calls_df["volume"] = calls_df["volume"].fillna(0)
                calls_df = calls_df[calls_df["volume"] > 0].nlargest(5, "volume")
                for _, row in calls_df.iterrows():
                    lp = float(row.get("lastPrice", 0) or 0)
                    chg = float(row.get("percentChange", 0) or 0)
                    hot_calls.append(
                        {
                            "expiry": exp,
                            "strike": float(row["strike"]),
                            "volume": int(row["volume"]),
                            "oi": int(row.get("openInterest", 0) or 0),
                            "iv": round(float(row.get("impliedVolatility", 0) or 0) * 100, 1),
                            "type": "call",
                            "in_money": bool(row.get("inTheMoney", False)),
                            "last_price": round(lp, 2),
                            "pct_change": round(chg, 2),
                        }
                    )

                # Top 5 put strikes by volume
                puts_df = chain.puts.copy()
                puts_df["volume"] = puts_df["volume"].fillna(0)
                puts_df_top = puts_df[puts_df["volume"] > 0].nlargest(5, "volume")
                for _, row in puts_df_top.iterrows():
                    lp = float(row.get("lastPrice", 0) or 0)
                    chg = float(row.get("percentChange", 0) or 0)
                    hot_puts.append(
                        {
                            "expiry": exp,
                            "strike": float(row["strike"]),
                            "volume": int(row["volume"]),
                            "oi": int(row.get("openInterest", 0) or 0),
                            "iv": round(float(row.get("impliedVolatility", 0) or 0) * 100, 1),
                            "type": "put",
                            "in_money": bool(row.get("inTheMoney", False)),
                            "last_price": round(lp, 2),
                            "pct_change": round(chg, 2),
                        }
                    )

                unusual_activity.extend(_scan_unusual_activity(chain.calls, chain.puts, exp, today_date))
                scanned_for_unusual.add(exp)

            except Exception:
                pass

        # Fetch up to 5 more expirations beyond the first 3 to catch LEAPS-type
        # unusual flow. Each chain fetch is ~0.5-1 s; the 5-min sentiment cache
        # amortises the cost.
        for exp in expirations[:n_unusual_exp]:
            if exp in scanned_for_unusual:
                continue
            try:
                chain = yf_call(t.option_chain, exp)
                unusual_activity.extend(_scan_unusual_activity(chain.calls, chain.puts, exp, today_date))
            except Exception:
                pass

        # Sort by vol/OI ratio descending; cap to keep the payload small
        unusual_activity.sort(key=lambda x: x["vol_oi"], reverse=True)
        unusual_activity = unusual_activity[:50]

        total_vol = total_call_vol + total_put_vol
        pcr_vol = round(total_put_vol / total_call_vol, 3) if total_call_vol else None
        pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi else None

        flow_bias = (
            "call_heavy"
            if pcr_vol is not None and pcr_vol < 0.70
            else "put_heavy"
            if pcr_vol is not None and pcr_vol > 1.00
            else "neutral"
            if pcr_vol is not None
            else "unknown"
        )

        call_pct = round(total_call_vol / total_vol * 100) if total_vol else 0
        put_pct = round(total_put_vol / total_vol * 100) if total_vol else 0

        # Keep top-10 hottest strikes across all expiries
        hot_calls.sort(key=lambda x: x["volume"], reverse=True)
        hot_puts.sort(key=lambda x: x["volume"], reverse=True)

        # Get current spot price for ATM/OTM classification downstream
        try:
            info = yf_call(lambda: t.fast_info)
            spot_price = float(
                getattr(info, "last_price", None) or getattr(info, "regular_market_price", None) or 0.0
            )
        except Exception:
            spot_price = 0.0

        return {
            "available": True,
            "expirations_used": n_exp,
            "spot_price": round(spot_price, 2),
            "call_volume": total_call_vol,
            "put_volume": total_put_vol,
            "call_oi": total_call_oi,
            "put_oi": total_put_oi,
            "pcr_volume": pcr_vol,
            "pcr_oi": pcr_oi,
            "flow_bias": flow_bias,
            "call_pct": call_pct,
            "put_pct": put_pct,
            "chains": chains_summary,
            "hot_calls": hot_calls[:10],
            "hot_puts": hot_puts[:10],
            # Unusual Activity: contracts with vol > OI×1.5 AND vol > 500.
            # Raw exchange volume - NOT aggressor-side; the UI shows a disclaimer.
            "unusual_activity": unusual_activity,
            "flow_direction_available": False,  # yfinance never has aggressor side
        }

    except Exception as exc:
        logger.debug("Options flow %s failed: %s", ticker, exc)
        return {"available": False, "reason": str(exc)}


# Options-activity sentiment scorer


def _score_options_activity(options_data: dict) -> tuple[float, str, dict]:
    """Score options flow from ATM/OTM volume, expiry, and PCR. Returns (score, label, details)."""
    from datetime import date, timedelta

    if not options_data.get("available"):
        return 0.0, "neutral", {}

    spot = float(options_data.get("spot_price", 0.0))
    if spot <= 0:
        return 0.0, "neutral", {}

    hot_calls = options_data.get("hot_calls", [])
    hot_puts = options_data.get("hot_puts", [])
    if not hot_calls and not hot_puts:
        return 0.0, "neutral", {}

    atm_lo = spot * 0.985  # ATM band: ±1.5% of spot
    atm_hi = spot * 1.015
    otm_call_lo = spot * 1.0  # OTM calls: 0-12% above spot
    otm_call_hi = spot * 1.12
    otm_put_lo = spot * 0.88  # OTM puts:  0-12% below spot
    otm_put_hi = spot * 1.0

    today = date.today()
    days_to_fri = (4 - today.weekday()) % 7 or 7  # always >= 1
    next_fri = today + timedelta(days=days_to_fri)
    nw_start = (next_fri - timedelta(days=1)).isoformat()  # Thursday
    nw_end = (next_fri + timedelta(days=3)).isoformat()  # following Monday

    atm_call_vol = sum(c["volume"] for c in hot_calls if atm_lo <= c["strike"] <= atm_hi)
    atm_put_vol = sum(p["volume"] for p in hot_puts if atm_lo <= p["strike"] <= atm_hi)

    otm_call_vol = sum(c["volume"] for c in hot_calls if otm_call_lo < c["strike"] <= otm_call_hi)
    otm_put_vol = sum(p["volume"] for p in hot_puts if otm_put_lo <= p["strike"] < otm_put_hi)

    # Distinct OTM call strikes - "call ladder" breadth indicator
    otm_call_strikes = sorted(
        {c["strike"] for c in hot_calls if otm_call_lo < c["strike"] <= otm_call_hi and c["volume"] > 0}
    )
    n_ladder = len(otm_call_strikes)

    # Next-week volume concentration
    nw_call_vol = sum(c["volume"] for c in hot_calls if nw_start <= (c.get("expiry") or "") <= nw_end)
    nw_put_vol = sum(p["volume"] for p in hot_puts if nw_start <= (p.get("expiry") or "") <= nw_end)

    pcr = options_data.get("pcr_volume")  # None if unavailable

    score = 0.0

    atm_total = atm_call_vol + atm_put_vol
    if atm_total > 0:
        score += ((atm_call_vol - atm_put_vol) / atm_total) * 0.35

    otm_total = otm_call_vol + otm_put_vol
    if otm_total > 0:
        score += ((otm_call_vol - otm_put_vol) / otm_total) * 0.30

    if n_ladder >= 4:
        score += 0.12
    elif n_ladder == 3:
        score += 0.08
    elif n_ladder == 2:
        score += 0.04

    nw_total = nw_call_vol + nw_put_vol
    if nw_total > 0:
        score += ((nw_call_vol - nw_put_vol) / nw_total) * 0.20

    if pcr is not None:
        if pcr < 0.35:
            score += 0.15  # extremely call-heavy
        elif pcr < 0.60:
            score += 0.10
        elif pcr < 0.80:
            score += 0.05
        elif pcr > 1.80:
            score -= 0.15  # extremely put-heavy
        elif pcr > 1.20:
            score -= 0.10
        elif pcr > 1.00:
            score -= 0.05

    score = round(max(-1.0, min(1.0, score)), 3)

    if score > 0.35:
        label = "strongly_bullish"
    elif score > 0.12:
        label = "bullish"
    elif score < -0.35:
        label = "strongly_bearish"
    elif score < -0.12:
        label = "bearish"
    else:
        label = "neutral"

    details = {
        "spot": round(spot, 2),
        "atm_call_vol": atm_call_vol,
        "atm_put_vol": atm_put_vol,
        "otm_call_vol": otm_call_vol,
        "otm_put_vol": otm_put_vol,
        "otm_call_strikes": otm_call_strikes,
        "call_ladder_count": n_ladder,
        "next_week_expiry": next_fri.isoformat(),
        "next_week_call_vol": nw_call_vol,
        "next_week_put_vol": nw_put_vol,
        "pcr": pcr,
        "score": score,
        "label": label,
    }
    return score, label, details
