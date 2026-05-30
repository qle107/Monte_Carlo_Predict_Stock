"""Pydantic request/response models."""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from config import VALID_INTERVALS, VALID_MC_MODELS

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class ConfigUpdate(BaseModel):
    ticker: str | None = None
    interval: str | None = None
    n_sim: int | None = None
    n_forward: int | None = None
    lookback: int | None = None
    chart_bars: int | None = None
    poll_seconds: int | None = None
    extended: bool | None = None
    mc_model: str | None = None

    garch_alpha: float | None = None
    garch_beta: float | None = None
    mc_clip: float | None = None
    jump_intensity: float | None = None
    jump_sigma_mult: float | None = None

    min_score: float | None = None
    min_adx: float | None = None
    min_conf: float | None = None
    min_mc_prob: float | None = None
    min_rr: float | None = None
    min_score_choppy: float | None = None
    rsi_overbought: float | None = None
    rsi_oversold: float | None = None
    sl_max_pct: float | None = None

    rsi_period: int | None = None
    ema_fast: int | None = None
    ema_slow: int | None = None
    ema_long: int | None = None
    macd_fast: int | None = None
    macd_slow: int | None = None
    macd_signal: int | None = None
    bb_period: int | None = None
    bb_k: float | None = None
    atr_period: int | None = None
    adx_period: int | None = None
    obv_period: int | None = None
    slope_period: int | None = None
    mom_period: int | None = None
    vwap_period: int | None = None
    rsi_div_lookback: int | None = None

    zone_pivot_window: int | None = None
    zone_cluster_atr: float | None = None
    zone_touch_atr: float | None = None
    zone_break_atr: float | None = None
    zone_max_demand: int | None = None
    zone_max_supply: int | None = None
    zone_width_atr: float | None = None

    backtest_band_pct: float | None = None
    backtest_commission: float | None = None
    backtest_slippage: float | None = None

    scan_min_score: float | None = None
    scan_max_concurrent: int | None = None

    regime_hurst_lags: int | None = None
    regime_donchian_n: int | None = None
    regime_pivot_wing: int | None = None

    signal_base_weights: str | None = None
    gap_threshold: float | None = None

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
            raise ValueError("n_sim must be 50-50 000")
        return v

    @field_validator("n_forward")
    @classmethod
    def valid_nfwd(cls, v):
        if v is not None and not (1 <= v <= 100):
            raise ValueError("n_forward must be 1-100")
        return v

    @field_validator("lookback")
    @classmethod
    def valid_lookback(cls, v):
        if v is not None and not (20 <= v <= 500):
            raise ValueError("lookback must be 20-500")
        return v

    @field_validator("chart_bars")
    @classmethod
    def valid_chart_bars(cls, v):
        if v is not None and not (50 <= v <= 1000):
            raise ValueError("chart_bars must be 50-1000")
        return v

    @field_validator("poll_seconds")
    @classmethod
    def valid_poll(cls, v):
        if v is not None and not (10 <= v <= 3600):
            raise ValueError("poll_seconds must be 10-3600")
        return v


class ScanRequest(BaseModel):
    """Request model for POST /api/scan."""

    tickers: list[str] | None = None
    watchlist: str | None = None
    interval: str | None = None
    lookback: int | None = None
    extended: bool | None = None
    max_concurrent: int | None = None
    min_score_abs: float | None = None

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
            raise ValueError("lookback must be 10-500")
        return v

    @field_validator("min_score_abs")
    @classmethod
    def valid_min_score(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("min_score_abs must be 0.0-1.0")
        return v


class BacktestRequest(BaseModel):
    """Optional overrides for POST /api/backtest."""

    ticker: str | None = None
    interval: str | None = None
    lookback: int | None = None
    n_forward: int | None = None
    n_sim: int | None = None
    mc_model: str | None = None
    history_bars: int | None = None

    @field_validator("history_bars")
    @classmethod
    def valid_hist(cls, v):
        if v is not None and not (50 <= v <= 2000):
            raise ValueError("history_bars must be 50-2000")
        return v
