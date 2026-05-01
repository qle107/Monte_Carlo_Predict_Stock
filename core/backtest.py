"""
core/backtest.py — walk-forward backtesting.

Given a long candle DataFrame, slide a window across history. At every step:
  1. Compute indicators + signal on history-up-to-i
  2. Run Monte Carlo with the chosen model
  3. Compare predicted prob_up vs the realised next-N-bar return

Reports
───────
  hit_rate           % of calls (Buy / Sell) that finished in the right direction
  brier_score        Mean squared error of prob_up vs realised up (0–1, lower is better)
  log_loss           Binary cross-entropy of prob_up vs realised up (lower is better)
  expected_vs_real   Correlation between MC expected_return and realised return
  calibration        Bucketed reliability: avg prob_up vs realised up rate, in 5 bins
  signals            Per-step list (compact) for plotting / inspection
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .indicators import compute_indicators
from .montecarlo import run as run_mc
from .signal import compute_signal


def _safe_log(p: float) -> float:
    return math.log(max(min(p, 1 - 1e-9), 1e-9))


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
    """
    closes = df["close"].to_numpy(float)
    n      = len(df)

    if n < min_history + n_forward + 5:
        return {
            "ok":              False,
            "error":           f"need ≥ {min_history + n_forward + 5} bars, got {n}",
            "n_evaluated":     0,
            "hit_rate":        None,
            "brier_score":     None,
            "log_loss":        None,
            "expected_vs_real": None,
            "calibration":     [],
            "signals":         [],
        }

    last_eval = n - n_forward - 1  # last index where we have a future to compare
    indices   = list(range(min_history, last_eval + 1, max(1, step)))

    rows:    List[dict]  = []
    correct: int         = 0
    called:  int         = 0
    briers:  List[float] = []
    logls:   List[float] = []
    pred_returns: List[float] = []
    real_returns: List[float] = []

    for i in indices:
        sub = df.iloc[: i + 1]
        if len(sub) < min_history:
            continue
        try:
            ind = compute_indicators(sub)
            sig = compute_signal(ind)
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
        # Realised "up" = ended above flat band
        band_pct    = 0.003
        realised_up = 1.0 if real_ret > band_pct else 0.0

        briers.append((prob_up_dec - realised_up) ** 2)
        logls.append(-(realised_up * _safe_log(prob_up_dec) +
                       (1 - realised_up) * _safe_log(1 - prob_up_dec)))
        pred_returns.append(pred_ret)
        real_returns.append(real_ret)

        # Was the directional call correct?  Buy/Sell only — neutrals don't score.
        label = sig.label
        is_call = "Buy" in label or "Sell" in label
        if is_call:
            called += 1
            up_call = "Buy" in label
            if (up_call and real_ret > 0) or (not up_call and real_ret < 0):
                correct += 1

        rows.append({
            "ts":        sub.index[-1].isoformat(),
            "price":     round(entry, 4),
            "label":     label,
            "conf":      sig.confidence,
            "prob_up":   mc.prob_up,
            "exp_ret":   mc.expected_return,
            "real_ret":  round(real_ret * 100, 3),
            "hit":       bool(
                ("Buy" in label and real_ret > 0) or
                ("Sell" in label and real_ret < 0)
            ),
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
            "calibration": [],
            "signals": [],
        }

    # Calibration buckets on prob_up
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

    return {
        "ok":               True,
        "n_evaluated":      len(rows),
        "n_called":         called,
        "hit_rate":         round(correct / called * 100, 2) if called else None,
        "brier_score":      round(float(np.mean(briers)), 4),
        "log_loss":         round(float(np.mean(logls)),  4),
        "expected_vs_real": round(corr, 4),
        "mean_prob_up":     round(float(np.mean([r["prob_up"] for r in rows])), 2),
        "real_up_rate":     round(float(np.mean([1.0 if r["real_ret"] > 0.3 else 0.0 for r in rows])) * 100, 2),
        "calibration":      calibration,
        "signals":          rows[-300:],   # keep payload small
    }
