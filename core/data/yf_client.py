"""Shared yfinance rate limiting, retry, and TTL cache."""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class _YFNoiseFilter(logging.Filter):
    """Suppress yfinance 'possibly delisted' noise for empty option windows."""

    _PHRASES = ("possibly delisted", "no price data found", "no data found")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        return not any(p in msg for p in self._PHRASES)


logging.getLogger("yfinance").addFilter(_YFNoiseFilter())

# yfinance connection limit - keep concurrent requests modest.
_YF_SEM_SLOTS = 4
yf_semaphore = threading.Semaphore(_YF_SEM_SLOTS)

_RATE_LIMIT_PHRASES = ("too many requests", "rate limit", "429")


def is_rate_limit(exc: Exception) -> bool:
    """True if the exception looks like a yfinance/Yahoo rate-limit error."""
    msg = str(exc).lower()
    return any(p in msg for p in _RATE_LIMIT_PHRASES)


def yf_call(fn: Callable, *args, retries: int = 3, base_delay: float = 2.0, **kwargs):
    """
    Call a yfinance function under the global semaphore with exponential-backoff
    retry on rate-limit errors. base_delay doubles each attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.2, 1.0)
            logger.debug("[yf_call] rate-limited, retry %d/%d in %.1fs", attempt, retries, delay)
            time.sleep(delay)

        with yf_semaphore:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if is_rate_limit(exc) and attempt < retries:
                    continue  # retry after backoff
                raise  # non-rate-limit error or out of retries

    raise last_exc  # type: ignore[misc]


def safe_int(val, default: int = 0) -> int:
    """Convert val to int, returning default for None / NaN / Inf."""
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else int(f)
    except (TypeError, ValueError):
        return default


def safe_float(val, default: float = 0.0) -> float:
    """Convert val to float, returning default for None / NaN / Inf."""
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default
