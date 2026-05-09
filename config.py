"""
config.py — single source of truth for all runtime settings.

Import `cfg` anywhere; mutate it via POST /api/config.

All values are validated; out-of-range env values fall back to defaults
with a warning (no silent garbage).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_INTERVALS: List[str] = [
    "1m", "2m", "5m", "15m", "30m", "1h", "4h", "1d",
]

VALID_MC_MODELS: List[str] = [
    "gaussian",     # classic GBM with Normal innovations (legacy)
    "student_t",    # heavy-tailed innovations  (degrees of freedom from data)
    "garch",        # GARCH(1,1)-style volatility clustering
    "bootstrap",    # historical-return resampling (preserves real distribution)
    "jump",         # Merton jump-diffusion (Gaussian + Poisson jumps)
    "ensemble",     # adaptive blend of GARCH + bootstrap + jump (recommended)
]


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        logger.warning("config: %s=%r is not an int, using default %d", name, raw, default)
        return default
    if not (lo <= v <= hi):
        logger.warning("config: %s=%d out of range [%d, %d], using default %d",
                       name, v, lo, hi, default)
        return default
    return v


def _env_str_choice(name: str, default: str, choices: List[str]) -> str:
    raw = os.getenv(name, default)
    if raw not in choices:
        logger.warning("config: %s=%r not in %s, using default %s",
                       name, raw, choices, default)
        return default
    return raw


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        logger.warning("config: %s=%r is not a float, using default %s", name, raw, default)
        return default
    if not (lo <= v <= hi):
        logger.warning("config: %s=%s out of range [%s, %s], using default %s",
                       name, v, lo, hi, default)
        return default
    return v


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y", "t")


# ─────────────────────────────────────────────────────────────────────────────
# Config object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Trading / server ─────────────────────────────────────────────────────
    ticker:        str  = field(default_factory=lambda: os.getenv("TICKER", "PLTR").upper().strip())
    interval:      str  = field(default_factory=lambda: _env_str_choice("CANDLE_INTERVAL", "15m", VALID_INTERVALS))
    poll_seconds:  int  = field(default_factory=lambda: _env_int("POLL_SECONDS", 60, 10, 3600))
    extended:      bool = field(default_factory=lambda: _env_bool("EXTENDED_HOURS", False))
    port:          int  = field(default_factory=lambda: _env_int("PORT", 8000, 1, 65535))
    db_path:       str  = field(default_factory=lambda: os.getenv("DB_PATH", "mc_trader.db"))

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    n_sim:         int  = field(default_factory=lambda: _env_int("MC_SIMULATIONS",    10000, 50, 50000))
    n_forward:     int  = field(default_factory=lambda: _env_int("MC_FORWARD_CANDLES",  10,  1,  100))
    lookback:      int  = field(default_factory=lambda: _env_int("LOOKBACK",            50, 20, 500))
    chart_bars:    int  = field(default_factory=lambda: _env_int("CHART_BARS",         200, 50, 1000))
    mc_model:      str  = field(default_factory=lambda: _env_str_choice("MC_MODEL", "garch", VALID_MC_MODELS))
    # GARCH parameters (α + β must sum < 1 for stationarity)
    garch_alpha:   float = field(default_factory=lambda: _env_float("GARCH_ALPHA", 0.10, 0.01, 0.49))
    garch_beta:    float = field(default_factory=lambda: _env_float("GARCH_BETA",  0.85, 0.10, 0.98))
    # Return clipping per MC step (±fraction). E.g. 0.25 = ±25%
    mc_clip:       float = field(default_factory=lambda: _env_float("MC_CLIP", 0.25, 0.05, 1.0))
    # Jump diffusion parameters
    jump_intensity: float = field(default_factory=lambda: _env_float("JUMP_INTENSITY", 0.03, 0.0, 0.30))
    jump_sigma_mult: float = field(default_factory=lambda: _env_float("JUMP_SIGMA_MULT", 3.0, 1.0, 10.0))

    # ── Trade setup gates ─────────────────────────────────────────────────────
    # Minimum |score| to consider an entry
    min_score:     float = field(default_factory=lambda: _env_float("MIN_SCORE",   0.20, 0.0, 1.0))
    # Minimum ADX for trend-strength gate
    min_adx:       float = field(default_factory=lambda: _env_float("MIN_ADX",    15.0, 0.0, 100.0))
    # Minimum signal confidence gate
    min_conf:      float = field(default_factory=lambda: _env_float("MIN_CONF",    0.30, 0.0, 1.0))
    # Minimum MC directional probability gate
    min_mc_prob:   float = field(default_factory=lambda: _env_float("MIN_MC_PROB", 0.40, 0.0, 1.0))
    # Minimum acceptable Risk:Reward
    min_rr:        float = field(default_factory=lambda: _env_float("MIN_RR",      1.0, 0.1, 10.0))
    # Choppy/range-bound regime: require stronger score than normal
    min_score_choppy: float = field(default_factory=lambda: _env_float("MIN_SCORE_CHOPPY", 0.45, 0.0, 1.0))
    # RSI extreme gates (don't long above / short below these levels)
    rsi_overbought: float = field(default_factory=lambda: _env_float("RSI_OVERBOUGHT", 82.0, 50.0, 100.0))
    rsi_oversold:   float = field(default_factory=lambda: _env_float("RSI_OVERSOLD",   18.0, 0.0,  50.0))
    # Fixed-% SL hard ceiling
    sl_max_pct:    float = field(default_factory=lambda: _env_float("SL_MAX_PCT", 0.05, 0.01, 0.30))

    # ── Indicator periods ─────────────────────────────────────────────────────
    rsi_period:    int  = field(default_factory=lambda: _env_int("RSI_PERIOD",   14,  2,  100))
    ema_fast:      int  = field(default_factory=lambda: _env_int("EMA_FAST",      9,  2,  100))
    ema_slow:      int  = field(default_factory=lambda: _env_int("EMA_SLOW",     21,  3,  500))
    ema_long:      int  = field(default_factory=lambda: _env_int("EMA_LONG",    200, 10, 1000))
    macd_fast:     int  = field(default_factory=lambda: _env_int("MACD_FAST",   12,  2,  100))
    macd_slow:     int  = field(default_factory=lambda: _env_int("MACD_SLOW",   26,  5,  200))
    macd_signal:   int  = field(default_factory=lambda: _env_int("MACD_SIGNAL",  9,  2,   50))
    bb_period:     int  = field(default_factory=lambda: _env_int("BB_PERIOD",   20,  5,  200))
    bb_k:          float = field(default_factory=lambda: _env_float("BB_K",     2.0, 0.5, 5.0))
    atr_period:    int  = field(default_factory=lambda: _env_int("ATR_PERIOD",  14,  2,  100))
    adx_period:    int  = field(default_factory=lambda: _env_int("ADX_PERIOD",  14,  2,  100))
    obv_period:    int  = field(default_factory=lambda: _env_int("OBV_PERIOD",  14,  3,  100))
    slope_period:  int  = field(default_factory=lambda: _env_int("SLOPE_PERIOD", 8,  2,  100))
    mom_period:    int  = field(default_factory=lambda: _env_int("MOM_PERIOD",   5,  1,   50))
    vwap_period:   int  = field(default_factory=lambda: _env_int("VWAP_PERIOD", 26,  5,  390))
    rsi_div_lookback: int = field(default_factory=lambda: _env_int("RSI_DIV_LOOKBACK", 30, 10, 200))

    # ── Signal weights (each set must sum to 1.0 — validated at startup) ──────
    # Base weights used for choppy regime (and as fallback)
    # Format: comma-separated floats in this fixed order:
    #   rsi, slope, momentum, ema, macd, bollinger, adx, obv, vwap, skew, trend_bias, rsi_div, ema200
    # If not set, the hardcoded defaults in signal.py are used.
    signal_base_weights: str = field(default_factory=lambda: os.getenv("SIGNAL_BASE_WEIGHTS", ""))

    # ── Gap detection threshold ───────────────────────────────────────────────
    gap_threshold: float = field(default_factory=lambda: _env_float("GAP_THRESHOLD", 3.0, 0.1, 20.0))

    # ── Zone detection ────────────────────────────────────────────────────────
    zone_pivot_window:   int  = field(default_factory=lambda: _env_int("ZONE_PIVOT_WINDOW",   4,  1, 20))
    zone_cluster_atr:    float = field(default_factory=lambda: _env_float("ZONE_CLUSTER_ATR", 0.8, 0.1, 5.0))
    zone_touch_atr:      float = field(default_factory=lambda: _env_float("ZONE_TOUCH_ATR",   0.6, 0.1, 3.0))
    zone_break_atr:      float = field(default_factory=lambda: _env_float("ZONE_BREAK_ATR",   0.5, 0.1, 3.0))
    zone_max_demand:     int  = field(default_factory=lambda: _env_int("ZONE_MAX_DEMAND", 5,  1, 20))
    zone_max_supply:     int  = field(default_factory=lambda: _env_int("ZONE_MAX_SUPPLY", 5,  1, 20))
    zone_width_atr:      float = field(default_factory=lambda: _env_float("ZONE_WIDTH_ATR", 0.3, 0.05, 2.0))

    # ── Backtest ──────────────────────────────────────────────────────────────
    backtest_band_pct:   float = field(default_factory=lambda: _env_float("BACKTEST_BAND_PCT",  0.003, 0.0, 0.05))
    backtest_commission: float = field(default_factory=lambda: _env_float("BACKTEST_COMMISSION", 0.001, 0.0, 0.05))
    # Slippage per side as fraction of price (e.g. 0.0005 = 0.05%)
    backtest_slippage:   float = field(default_factory=lambda: _env_float("BACKTEST_SLIPPAGE",   0.0005, 0.0, 0.02))

    # ── Scanner ───────────────────────────────────────────────────────────────
    scan_min_score:      float = field(default_factory=lambda: _env_float("SCAN_MIN_SCORE", 0.20, 0.0, 1.0))
    scan_max_concurrent: int  = field(default_factory=lambda: _env_int("SCAN_MAX_CONCURRENT", 8, 1, 50))

    # ── Regime detection ──────────────────────────────────────────────────────
    regime_hurst_lags:   int  = field(default_factory=lambda: _env_int("REGIME_HURST_LAGS",  20,  5, 100))
    regime_donchian_n:   int  = field(default_factory=lambda: _env_int("REGIME_DONCHIAN_N",  20,  5, 200))
    regime_pivot_wing:   int  = field(default_factory=lambda: _env_int("REGIME_PIVOT_WING",   3,  1,  20))

    # ── Security ─────────────────────────────────────────────────────────────
    api_key:        str  = field(default_factory=lambda: os.getenv("API_KEY", ""))

    def __post_init__(self) -> None:
        """Validate cross-field invariants after dataclass construction."""
        # GARCH stationarity: α + β must be strictly < 1.
        # If the sum is too large we nudge β down to 0.94 - α (keeps ~95% persistence).
        if self.garch_alpha + self.garch_beta >= 1.0:
            safe_beta = max(0.10, 0.94 - self.garch_alpha)
            logger.warning(
                "config: garch_alpha (%.3f) + garch_beta (%.3f) ≥ 1 (non-stationary). "
                "Clamping garch_beta → %.3f",
                self.garch_alpha, self.garch_beta, safe_beta,
            )
            self.garch_beta = safe_beta

        # EMA ordering: fast < slow < long
        if self.ema_fast >= self.ema_slow:
            logger.warning(
                "config: ema_fast (%d) ≥ ema_slow (%d) — signals will be unreliable.",
                self.ema_fast, self.ema_slow,
            )
        if self.ema_slow >= self.ema_long:
            logger.warning(
                "config: ema_slow (%d) ≥ ema_long (%d) — EMA200 signal may be wrong.",
                self.ema_slow, self.ema_long,
            )

        # MACD: fast < slow
        if self.macd_fast >= self.macd_slow:
            logger.warning(
                "config: macd_fast (%d) ≥ macd_slow (%d) — MACD will be inverted.",
                self.macd_fast, self.macd_slow,
            )

        # RSI gates: oversold < overbought
        if self.rsi_oversold >= self.rsi_overbought:
            logger.warning(
                "config: rsi_oversold (%.1f) ≥ rsi_overbought (%.1f) — RSI gates are inverted.",
                self.rsi_oversold, self.rsi_overbought,
            )

        # Zone: touch_atr should be ≤ cluster_atr (can't cluster tighter than touch band)
        if self.zone_touch_atr > self.zone_cluster_atr:
            logger.warning(
                "config: zone_touch_atr (%.2f) > zone_cluster_atr (%.2f). "
                "Touch band is wider than cluster radius — zones may merge unexpectedly.",
                self.zone_touch_atr, self.zone_cluster_atr,
            )

    def to_dict(self) -> dict:
        return asdict(self)


cfg = Config()
