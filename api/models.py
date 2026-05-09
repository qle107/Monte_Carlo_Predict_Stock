"""
api/models.py — Pydantic request/response models.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, field_validator

from config import VALID_INTERVALS, VALID_MC_MODELS

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class ConfigUpdate(BaseModel):
    # ── Core ─────────────────────────────────────────────────────────────
    ticker:       Optional[str]   = None
    interval:     Optional[str]   = None
    n_sim:        Optional[int]   = None
    n_forward:    Optional[int]   = None
    lookback:     Optional[int]   = None
    chart_bars:   Optional[int]   = None
    poll_seconds: Optional[int]   = None
    extended:     Optional[bool]  = None
    mc_model:     Optional[str]   = None

    # ── Monte Carlo model parameters ─────────────────────────────────────
    garch_alpha:      Optional[float] = None
    garch_beta:       Optional[float] = None
    mc_clip:          Optional[float] = None
    jump_intensity:   Optional[float] = None
    jump_sigma_mult:  Optional[float] = None

    # ── Trade setup gates ─────────────────────────────────────────────────
    min_score:        Optional[float] = None
    min_adx:          Optional[float] = None
    min_conf:         Optional[float] = None
    min_mc_prob:      Optional[float] = None
    min_rr:           Optional[float] = None
    min_score_choppy: Optional[float] = None
    rsi_overbought:   Optional[float] = None
    rsi_oversold:     Optional[float] = None
    sl_max_pct:       Optional[float] = None

    # ── Indicator periods ─────────────────────────────────────────────────
    rsi_period:       Optional[int]   = None
    ema_fast:         Optional[int]   = None
    ema_slow:         Optional[int]   = None
    ema_long:         Optional[int]   = None
    macd_fast:        Optional[int]   = None
    macd_slow:        Optional[int]   = None
    macd_signal:      Optional[int]   = None
    bb_period:        Optional[int]   = None
    bb_k:             Optional[float] = None
    atr_period:       Optional[int]   = None
    adx_period:       Optional[int]   = None
    obv_period:       Optional[int]   = None
    slope_period:     Optional[int]   = None
    mom_period:       Optional[int]   = None
    vwap_period:      Optional[int]   = None
    rsi_div_lookback: Optional[int]   = None

    # ── Zone detection ────────────────────────────────────────────────────
    zone_pivot_window:  Optional[int]   = None
    zone_cluster_atr:   Optional[float] = None
    zone_touch_atr:     Optional[float] = None
    zone_break_atr:     Optional[float] = None
    zone_max_demand:    Optional[int]   = None
    zone_max_supply:    Optional[int]   = None
    zone_width_atr:     Optional[float] = None

    # ── Backtest ──────────────────────────────────────────────────────────
    backtest_band_pct:   Optional[float] = None
    backtest_commission: Optional[float] = None
    backtest_slippage:   Optional[float] = None

    # ── Scanner ───────────────────────────────────────────────────────────
    scan_min_score:      Optional[float] = None
    scan_max_concurrent: Optional[int]   = None

    # ── Regime ───────────────────────────────────────────────────────────
    regime_hurst_lags:  Optional[int] = None
    regime_donchian_n:  Optional[int] = None
    regime_pivot_wing:  Optional[int] = None

    # ── Signal weights ────────────────────────────────────────────────────
    signal_base_weights: Optional[str] = None
    gap_threshold:       Optional[float] = None

    # ── Validators ───────────────────────────────────────────────────────
    # Use `v is not None` (not `if v`) so a value of 0 is still validated
    # rather than silently passing through.

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, v):
        if v is None:
            return v
        v = v.upper().strip()
        if not _TICKER_RE.match(v):
            raise ValueError("ticker must be 1-10 chars, A-Z 0-9 . -")
        return v

    @field_validator("interval")
    @classmethod
    def valid_interval(cls, v):
        if v is not None and v not in VALID_INTERVALS:
            raise ValueError(f"interval must be one of {VALID_INTERVALS}")
        return v

    @field_validator("mc_model")
    @classmethod
    def valid_mc_model(cls, v):
        if v is not None and v not in VALID_MC_MODELS:
            raise ValueError(f"mc_model must be one of {VALID_MC_MODELS}")
        return v

    @field_validator("n_sim")
    @classmethod
    def valid_nsim(cls, v):
        if v is not None and not (50 <= v <= 50000):
            raise ValueError("n_sim must be 50–50 000")
        return v

    @field_validator("n_forward")
    @classmethod
    def valid_nfwd(cls, v):
        if v is not None and not (1 <= v <= 100):
            raise ValueError("n_forward must be 1–100")
        return v

    @field_validator("lookback")
    @classmethod
    def valid_lookback(cls, v):
        if v is not None and not (20 <= v <= 500):
            raise ValueError("lookback must be 20–500")
        return v

    @field_validator("chart_bars")
    @classmethod
    def valid_chart_bars(cls, v):
        if v is not None and not (50 <= v <= 1000):
            raise ValueError("chart_bars must be 50–1000")
        return v

    @field_validator("poll_seconds")
    @classmethod
    def valid_poll(cls, v):
        if v is not None and not (10 <= v <= 3600):
            raise ValueError("poll_seconds must be 10–3600")
        return v


class ScanRequest(BaseModel):
    """Request model for POST /api/scan."""
    tickers:        Optional[List[str]] = None    # explicit list; overrides watchlist
    watchlist:      Optional[str]       = None    # named watchlist key
    interval:       Optional[str]       = None    # candle interval (default: 1d)
    lookback:       Optional[int]       = None    # bars to fetch (default: 60)
    extended:       Optional[bool]      = None
    max_concurrent: Optional[int]       = None    # parallel fetches (max 20)
    min_score_abs:  Optional[float]     = None    # filter weak signals

    @field_validator("tickers")
    @classmethod
    def valid_tickers(cls, v):
        if v is None:
            return v
        cleaned = []
        for t in v:
            t = t.upper().strip()
            if not _TICKER_RE.match(t):
                raise ValueError(f"Invalid ticker: {t!r}")
            cleaned.append(t)
        if len(cleaned) > 200:
            raise ValueError("Max 200 tickers per scan")
        return cleaned

    @field_validator("interval")
    @classmethod
    def valid_interval(cls, v):
        if v is not None and v not in VALID_INTERVALS:
            raise ValueError(f"interval must be one of {VALID_INTERVALS}")
        return v

    @field_validator("lookback")
    @classmethod
    def valid_lookback(cls, v):
        if v is not None and not (10 <= v <= 500):
            raise ValueError("lookback must be 10–500")
        return v

    @field_validator("min_score_abs")
    @classmethod
    def valid_min_score(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("min_score_abs must be 0.0–1.0")
        return v


class BacktestRequest(BaseModel):
    """Optional explicit overrides for /api/backtest."""
    ticker:       Optional[str]  = None
    interval:     Optional[str]  = None
    lookback:     Optional[int]  = None
    n_forward:    Optional[int]  = None
    n_sim:        Optional[int]  = None
    mc_model:     Optional[str]  = None
    history_bars: Optional[int]  = None  # how much history to walk-forward over

    @field_validator("history_bars")
    @classmethod
    def valid_hist(cls, v):
        if v is not None and not (50 <= v <= 2000):
            raise ValueError("history_bars must be 50–2000")
        return v
