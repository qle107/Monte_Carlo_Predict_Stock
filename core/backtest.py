"""Walk-forward backtesting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import cfg

from .indicators import compute_indicators
from .montecarlo import run as run_mc
from .regime import detect_regime
from .signal import compute_signal

# Map a candle interval to bars-per-trading-year.
# Equity markets: ~252 trading days × hours of trade × (60 / interval_min).
_BARS_PER_YEAR = {
    "1m": 252 * 6.5 * 60,
    "2m": 252 * 6.5 * 30,
    "5m": 252 * 6.5 * 12,
    "15m": 252 * 6.5 * 4,
    "30m": 252 * 6.5 * 2,
    "1h": 252 * 6.5,
    "2h": 252 * 3.25,
    "4h": 252 * 1.625,
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


def _safe_log(p: float) -> float:
    return math.log(max(min(p, 1 - 1e-9), 1e-9))


def _max_drawdown_equity(equity: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown of an equity curve (decimal).
    `equity` must be strictly positive (cumulative wealth, not cumulative
    return). Returns a positive number - 0.15 = 15 % drawdown.
    """
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    dd = (running_peak - equity) / running_peak
    return float(dd.max())


def _sharpe_annualised(trade_returns: list[float], trade_duration_bars: int, bars_per_year: float) -> float:
    """
    Annualised Sharpe ratio computed from a list of per-trade net returns.
    We scale by sqrt(trades-per-year), where one trade spans
    `trade_duration_bars` bars of the timeframe.
    """
    if len(trade_returns) < 4:
        return 0.0
    arr = np.asarray(trade_returns, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std < 1e-12 or trade_duration_bars <= 0 or bars_per_year <= 0:
        return 0.0
    trades_per_year = bars_per_year / float(trade_duration_bars)
    return float(mean / std * math.sqrt(trades_per_year))


def _max_consec_losses(hits: list[bool]) -> int:
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
    n_forward: int = 10,
    n_sim: int = 200,
    mc_model: str = "garch",
    min_history: int = 50,
    step: int = 1,
    interval: str = "15m",
    allow_overlap: bool = False,
) -> dict:
    """
    df             long candle DataFrame (UTC index)
    n_forward      how many bars ahead to look for the realised outcome
    n_sim          MC simulations per step
    mc_model       MC innovation model
    min_history    require at least this many bars before issuing a signal
    step           stride between candidate signals (1 = every bar)
    interval       timeframe code - drives Sharpe annualisation
    allow_overlap  if False (default) a directional call blocks new calls
                   for `n_forward` bars, eliminating duplicate counting

    Transaction costs applied to each trade:
      commission = cfg.backtest_commission (fraction of trade value, each side)
      slippage   = cfg.backtest_slippage   (fraction, each side)
    Total cost per round-trip = 2 × (commission + slippage).
    """
    commission = float(cfg.backtest_commission)
    slippage = float(cfg.backtest_slippage)
    band_pct = float(cfg.backtest_band_pct)
    round_trip_cost = 2.0 * (commission + slippage)
    bars_per_year = _BARS_PER_YEAR.get(interval, 252.0)

    closes = df["close"].to_numpy(float)
    n = len(df)

    def _empty(err: str) -> dict:
        return {
            "ok": False,
            "error": err,
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

    if n < min_history + n_forward + 5:
        return _empty(f"need ≥ {min_history + n_forward + 5} bars, got {n}")

    last_eval = n - n_forward - 1  # last index where we have a future to compare
    indices = list(range(min_history, last_eval + 1, max(1, step)))

    rows: list[dict] = []
    correct: int = 0
    called: int = 0
    briers: list[float] = []
    logls: list[float] = []
    pred_returns: list[float] = []
    real_returns: list[float] = []

    # Trade-level stats - only directional (Buy/Sell) calls
    trade_net_rets: list[float] = []
    trade_hits: list[bool] = []
    next_eligible_bar: int = -1  # for non-overlapping trades

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
                entry,
                sig,
                n_simulations=n_sim,
                n_candles=n_forward,
                model=mc_model,
                recent_returns=ind.returns,
                kurtosis_excess=ind.kurtosis,
            )
        except Exception:
            continue

        # Realised next-N-bar return
        future_close = float(closes[i + n_forward])
        if not math.isfinite(future_close):
            continue
        real_ret = future_close / entry - 1.0
        pred_ret = mc.expected_return / 100.0

        prob_up_dec = mc.prob_up / 100.0
        realised_up = 1.0 if real_ret > band_pct else 0.0

        briers.append((prob_up_dec - realised_up) ** 2)
        logls.append(-(realised_up * _safe_log(prob_up_dec) + (1 - realised_up) * _safe_log(1 - prob_up_dec)))
        pred_returns.append(pred_ret)
        real_returns.append(real_ret)

        # Directional call tracking - Buy/Sell only
        label = sig.label
        is_call = ("Buy" in label) or ("Sell" in label)
        up_call = "Buy" in label
        # A "hit" is a move past the same band_pct used everywhere else, in
        # the called direction.
        hit = (up_call and real_ret > band_pct) or (not up_call and real_ret < -band_pct)

        # Only record the trade when overlap rules allow it.
        recorded_trade = False
        if is_call and (allow_overlap or i >= next_eligible_bar):
            called += 1
            if hit:
                correct += 1
            gross_ret = real_ret if up_call else -real_ret
            net_ret = gross_ret - round_trip_cost
            trade_net_rets.append(net_ret)
            trade_hits.append(hit)
            recorded_trade = True
            next_eligible_bar = i + n_forward  # block overlap

        rows.append(
            {
                "ts": sub.index[-1].isoformat(),
                "price": round(entry, 4),
                "label": label,
                "conf": sig.confidence,
                "prob_up": mc.prob_up,
                "exp_ret": mc.expected_return,
                "real_ret": round(real_ret * 100, 3),
                "hit": bool(hit) if is_call else None,
                "traded": recorded_trade,
            }
        )

    if not rows:
        return _empty("no valid evaluation points")

    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    calibration = []
    for lo, hi in buckets:
        idxs = [j for j, r in enumerate(rows) if lo <= r["prob_up"] / 100.0 < hi]
        if not idxs:
            calibration.append(
                {"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": 0, "avg_pred": None, "real_up_rate": None}
            )
            continue
        avg_pred = float(np.mean([rows[j]["prob_up"] / 100.0 for j in idxs]))
        real_up_rate = float(np.mean([1.0 if rows[j]["real_ret"] / 100.0 > band_pct else 0.0 for j in idxs]))
        calibration.append(
            {
                "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "n": len(idxs),
                "avg_pred": round(avg_pred, 3),
                "real_up_rate": round(real_up_rate, 3),
            }
        )

    pred_arr = np.array(pred_returns)
    real_arr = np.array(real_returns)
    if pred_arr.std() > 1e-12 and real_arr.std() > 1e-12:
        corr = float(np.corrcoef(pred_arr, real_arr)[0, 1])
    else:
        corr = 0.0

    wins = [r for r in trade_net_rets if r > 0]
    losses = [r for r in trade_net_rets if r <= 0]

    avg_win = float(np.mean(wins)) * 100 if wins else None
    avg_loss = float(np.mean(losses)) * 100 if losses else None

    win_loss_ratio = (
        abs(avg_win / avg_loss) if (avg_win is not None and avg_loss is not None and avg_loss != 0) else None
    )
    sum_losses = abs(sum(losses))
    if wins and sum_losses > 0:
        profit_factor = sum(wins) / sum_losses
    elif wins and not losses:
        profit_factor = float("inf")
    else:
        profit_factor = None

    sharpe: float | None = (
        _sharpe_annualised(trade_net_rets, n_forward, bars_per_year) if trade_net_rets else None
    )

    # Max drawdown on the equity curve (1 + cum_returns), not raw cumsum
    if trade_net_rets:
        equity = np.cumprod(1.0 + np.asarray(trade_net_rets))
        max_dd = _max_drawdown_equity(equity)
    else:
        max_dd = None

    max_consec = _max_consec_losses(trade_hits) if trade_hits else None

    return {
        "ok": True,
        "n_evaluated": len(rows),
        "n_called": called,
        "commission": commission,
        "slippage": slippage,
        "round_trip_cost": round(round_trip_cost * 100, 4),
        "interval": interval,
        "allow_overlap": allow_overlap,
        "hit_rate": round(correct / called * 100, 2) if called else None,
        "brier_score": round(float(np.mean(briers)), 4),
        "log_loss": round(float(np.mean(logls)), 4),
        "expected_vs_real": round(corr, 4),
        "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown": round(max_dd * 100, 2) if max_dd is not None else None,
        "avg_win": round(avg_win, 3) if avg_win is not None else None,
        "avg_loss": round(avg_loss, 3) if avg_loss is not None else None,
        "win_loss_ratio": round(win_loss_ratio, 3) if win_loss_ratio is not None else None,
        "profit_factor": (
            round(profit_factor, 3)
            if (profit_factor is not None and math.isfinite(profit_factor))
            else profit_factor
        ),
        "max_consec_losses": max_consec,
        "mean_prob_up": round(float(np.mean([r["prob_up"] for r in rows])), 2),
        "real_up_rate": round(
            float(np.mean([1.0 if r["real_ret"] / 100.0 > band_pct else 0.0 for r in rows])) * 100, 2
        ),
        "calibration": calibration,
        "signals": rows[-300:],
    }
