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
    # Trading
    ticker:        str  = field(default_factory=lambda: os.getenv("TICKER", "PLTR").upper().strip())
    interval:      str  = field(default_factory=lambda: _env_str_choice("CANDLE_INTERVAL", "15m", VALID_INTERVALS))

    # Monte Carlo
    n_sim:         int  = field(default_factory=lambda: _env_int("MC_SIMULATIONS",      500, 50, 5000))
    n_forward:     int  = field(default_factory=lambda: _env_int("MC_FORWARD_CANDLES",  10,  1,  100))
    lookback:      int  = field(default_factory=lambda: _env_int("LOOKBACK",            50, 20, 500))
    mc_model:      str  = field(default_factory=lambda: _env_str_choice("MC_MODEL", "garch", VALID_MC_MODELS))

    # Server
    poll_seconds:  int  = field(default_factory=lambda: _env_int("POLL_SECONDS", 60, 10, 3600))
    extended:      bool = field(default_factory=lambda: _env_bool("EXTENDED_HOURS", False))
    port:          int  = field(default_factory=lambda: _env_int("PORT", 8000, 1, 65535))

    # Persistence
    db_path:       str  = field(default_factory=lambda: os.getenv("DB_PATH", "mc_trader.db"))

    def to_dict(self) -> dict:
        return asdict(self)


cfg = Config()
