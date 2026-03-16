"""
config.py — single source of truth for all runtime settings.
Import `cfg` anywhere; mutate it via POST /api/config.
"""

import os
from dotenv import load_dotenv

load_dotenv()

VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "1h", "4h", "1d"]


class Config:
    # Trading settings
    ticker:       str = os.getenv("TICKER", "PLTR")
    interval:     str = os.getenv("CANDLE_INTERVAL", "15m")

    # Monte Carlo settings
    n_sim:        int = int(os.getenv("MC_SIMULATIONS", "500"))
    n_forward:    int = int(os.getenv("MC_FORWARD_CANDLES", "10"))
    lookback:     int = 50        # candles of history to load

    # Server settings
    poll_seconds: int = 60
    extended:     bool = False   # include pre/after-hours candles        # auto-refresh cadence
    port:         int = int(os.getenv("PORT", "8000"))


cfg = Config()
