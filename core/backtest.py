"""
core/backtest.py — walk-forward backtesting.

Given a long candle DataFrame, slide a window across history. At every step:
  1. Compute indicators + signal on history-up-to-i
  2. Run Monte Carlo with the chosen model
  3. Compare predicted prob_up vs the realised next-N-bar return

Reports
───────
  hit_rate            % of calls (Buy / Sell) that finished in the right direction
  brier_score         Mean squared error of prob_up vs realised up (0–1, lower is better)
  log_loss            Binary cross-entropy of prob_up vs realised up (lower is better)
  expected_vs_real    Correlation between MC expected_return and realised return
  calibration         Bucketed reliability: avg prob_up vs realised up rate, in 5 bins
  sharpe_ratio        Annualised Sharpe ratio of strategy returns (net of costs)
  max_drawdown        Maximum peak-to-trough drawdown of cumulative returns
  avg_win             Average return on winning trades (%)
  avg_loss            Average return on losing trades (%)
  win_loss_ratio      avg_win / |avg_loss|
  profit_factor       sum of wins / sum of |losses| (gross P&L factor)
  max_consec_losses   Maximum consecutive losing trades
  signals             Per-step list (compact) for plotting / inspection
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .indicators import compute_indicators
from .montecarlo import run as run_mc
from .regime import detect_regime
from .signal import compute_signal
from config import cfg


def _safe_log(p: float) -> float:
    return math.log(max(min(p, 1 - 1e-9), 1e-9))


def _max_drawdown(cum_returns: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown of a cumulative P&L series (decimal).
    Returns a positive number: 0.15 = 15% drawdown.
    """
    if len(cum_returns) == 0:
        return 0.0
    peak = cum_returns[0]
    max_dd = 0.0
    for r in cum_returns:
        if r > peak:
            peak = r
        dd = (peak - r) / max(abs(peak), 1e-12)
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _sharpe(trade_returns: List[float], periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio from a list of per-trade net returns (decimal).
    Assumes each trade occupies n_forward bars; annualises by trades-per-year.
    """
    if len(trade_returns) < 4:
        return 0.0
    arr = np.asarray(trade_returns, dtype=float)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    if std < 1e-12:
        return 0.0
    # Scale to annual: sqrt(periods_per_year / 1) where 1 trade ≈ 1 period unit
    return float(mean / std * np.sqrt(periods_per_year))


def _max_consec_losses(hits: List[bool]) -> int:
    """Return the maximum consecutive False values in a boolean list."""
    max_run = cur_run = 0
    for h in hits:
        if not h:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def walk_forward(
    df: pd.DataFrame,
    n_forward:    int = 10,
    n_sim:        int = 200,
    mc_model:     str = "garch",
    min_history:  int = 50,
    step:         int = 1,
) -> Dict:
    """
    df          : long candle DataFrame (UTC index)
    n_forward   : how many bars ahead to look for the realised outcome
    n_sim       : MC simulations per step
    mc_model    : MC innovation model
    min_history : require at least this many bars before issuing a signal
    step        : stride (1 = every bar). For 1m bars use step=4 to speed up.

    Transaction costs applied to each trade:
      commission = cfg.backtest_commission (fraction of trade value, each side)
      slippage   = cfg.backtest_slippage   (fraction, each side)
    Total cost per round-trip = 2 × (commission + slippage).
    """
    commission = float(cfg.backtest_commission)
    slippage   = float(cfg.backtest_slippage)
    band_pct   = float(cfg.backtest_band_pct)
    round_trip_cost = 2.0 * (commission + slippage)

    closes = df["close"].to_numpy(float)
    n      = len(df)

    if n < min_history + n_forward + 5:
        return {
            "ok":               False,
            "error":            f"need ≥ {min_history + n_forward + 5} bars, got {n}",
            "n_evaluated":      0,
            "hit_rate":         None,
            "brier_score":      None,
            "log_loss":         None,
            "expected_vs_real": None,
            "sharpe_ratio":     None,
            "max_drawdown":     None,
            "avg_win":          None,
            "avg_loss":         None,
            "win_loss_ratio":   None,
            "profit_factor":    None,
            "max_consec_losses": None,
            "calibration":      [],
            "signals":          [],
        }

    last_eval = n - n_forward - 1   # last index where we have a future to compare
    indices   = list(range(min_history, last_eval + 1, max(1, step)))

    rows:    List[dict]  = []
    correct: int         = 0
    called:  int         = 0
    briers:  List[float] = []
    logls:   List[float] = []
    pred_returns: List[float] = []
    real_returns: List[float] = []

    # For trade-level stats — only track directional calls (Buy / Sell)
    trade_net_rets: List[float] = []  # net of costs
    trade_hits:     List[bool]  = []

    for i in indices:
        sub = df.iloc[: i + 1]
        if len(sub) < min_history:
            continue
        try:
            ind = compute_indicators(sub)
            reg = detect_regime(sub, adx=ind.adx, obv_slope=ind.obv_slope)
            sig = compute_signal(ind, regime=reg)
            entry = float(sub["close"].iloc[-1])
            if entry <= 0 or not math.isfinite(entry):
                continue
            mc = run_mc(
                entry, sig,
                n_simulations   = n_sim,
                n_candles       = n_forward,
                model           = mc_model,
                recent_returns  = ind.returns,
                kurtosis_excess = ind.kurtosis,
            )
        except Exception:
            continue

        # Realised next-N-bar return
        future_close = float(closes[i + n_forward])
        if entry <= 0 or not math.isfinite(future_close):
            continue
        real_ret = future_close / entry - 1.0
        pred_ret = mc.expected_return / 100.0

        prob_up_dec = mc.prob_up / 100.0
        realised_up = 1.0 if real_ret > band_pct else 0.0

        briers.append((prob_up_dec - realised_up) ** 2)
        logls.append(-(realised_up * _safe_log(prob_up_dec) +
                       (1 - realised_up) * _safe_log(1 - prob_up_dec)))
        pred_returns.append(pred_ret)
        real_returns.append(real_ret)

        # Directional call tracking — Buy/Sell only
        label = sig.label
        is_call = "Buy" in label or "Sell" in label
        if is_call:
            called += 1
            up_call = "Buy" in label
            hit = (up_call and real_ret > 0) or (not up_call and real_ret < 0)
            if hit:
                correct += 1

            # Net return for trade = gross return direction - round-trip cost
            gross_ret = real_ret if up_call else -real_ret
            net_ret   = gross_ret - round_trip_cost
            trade_net_rets.append(net_ret)
            trade_hits.append(hit)

        rows.append({
            "ts":       sub.index[-1].isoformat(),
            "price":    round(entry, 4),
            "label":    label,
            "conf":     sig.confidence,
            "prob_up":  mc.prob_up,
            "exp_ret":  mc.expected_return,
            "real_ret": round(real_ret * 100, 3),
            "hit":      bool(
                ("Buy" in label and real_ret > 0) or
                ("Sell" in label and real_ret < 0)
            ) if is_call else None,
        })

    if not rows:
        return {
            "ok": False,
            "error": "no valid evaluation points",
            "n_evaluated": 0,
            "hit_rate": None,
            "brier_score": None,
            "log_loss": None,
            "expected_vs_real": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "avg_win": None,
            "avg_loss": None,
            "win_loss_ratio": None,
            "profit_factor": None,
            "max_consec_losses": None,
            "calibration": [],
            "signals": [],
        }

    # ── Calibration buckets on prob_up ────────────────────────────────────
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    calibration = []
    for lo, hi in buckets:
        mask = [(lo <= r["prob_up"] / 100.0 < hi) for r in rows]
        idxs = [j for j, m in enumerate(mask) if m]
        if not idxs:
            calibration.append({"bin": f"{lo:.1f}-{min(hi,1.0):.1f}", "n": 0,
                                 "avg_pred": None, "real_up_rate": None})
            continue
        avg_pred = float(np.mean([rows[j]["prob_up"] / 100.0 for j in idxs]))
        real_up_rate = float(np.mean([1.0 if rows[j]["real_ret"] > 0.3 else 0.0 for j in idxs]))
        calibration.append({
            "bin":          f"{lo:.1f}-{min(hi,1.0):.1f}",
            "n":            len(idxs),
            "avg_pred":     round(avg_pred, 3),
            "real_up_rate": round(real_up_rate, 3),
        })

    pred_arr = np.array(pred_returns)
    real_arr = np.array(real_returns)
    if pred_arr.std() > 1e-12 and real_arr.std() > 1e-12:
        corr = float(np.corrcoef(pred_arr, real_arr)[0, 1])
    else:
        corr = 0.0

    # ── Trade-level statistics ────────────────────────────────────────────
    wins  = [r for r in trade_net_rets if r > 0]
    losses = [r for r in trade_net_rets if r <= 0]

    avg_win  = float(np.mean(wins))   * 100 if wins   else None
    avg_loss = float(np.mean(losses)) * 100 if losses else None

    win_loss_ratio = (
        abs(avg_win / avg_loss) if (avg_win is not None and avg_loss is not None and avg_loss != 0)
        else None
    )
    profit_factor = (
        sum(wins) / abs(sum(losses))
        if wins and losses and abs(sum(losses)) > 0
        else None
    )

    # Sharpe on net trade returns (annualised assuming 252 trade-periods per year)
    sharpe = _sharpe(trade_net_rets, periods_per_year=252) if trade_net_rets else None

    # Max drawdown on cumulative net P&L of trades
    if trade_net_rets:
        cum = np.cumsum(np.asarray(trade_net_rets))
        max_dd = _max_drawdown(cum)
    else:
        max_dd = None

    max_consec = _max_consec_losses(trade_hits) if trade_hits else None

    return {
        "ok":                True,
        "n_evaluated":       len(rows),
        "n_called":          called,
        "commission":        commission,
        "slippage":          slippage,
        "round_trip_cost":   round(round_trip_cost * 100, 4),  # as %
        "hit_rate":          round(correct / called * 100, 2) if called else None,
        "brier_score":       round(float(np.mean(briers)), 4),
        "log_loss":          round(float(np.mean(logls)),  4),
        "expected_vs_real":  round(corr, 4),
        "sharpe_ratio":      round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown":      round(max_dd * 100, 2) if max_dd is not None else None,  # as %
        "avg_win":           round(avg_win,  3) if avg_win  is not None else None,
        "avg_loss":          round(avg_loss, 3) if avg_loss is not None else None,
        "win_loss_ratio":    round(win_loss_ratio, 3) if win_loss_ratio is not None else None,
        "profit_factor":     round(profit_factor, 3) if profit_factor is not None else None,
        "max_consec_losses": max_consec,
        "mean_prob_up":      round(float(np.mean([r["prob_up"] for r in rows])), 2),
        "real_up_rate":      round(float(np.mean([1.0 if r["real_ret"] > 0.3 else 0.0 for r in rows])) * 100, 2),
        "calibration":       calibration,
        "signals":           rows[-300:],   # keep payload small
    }
