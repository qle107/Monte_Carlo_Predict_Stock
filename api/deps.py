"""Shared FastAPI dependencies: rate limiter, API-key guard."""

from __future__ import annotations

from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import cfg

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


def require_api_key(api_key: str | None) -> None:
    """Raise 401 if API_KEY env is set and the header doesn't match."""
    if cfg.api_key and (not api_key or api_key != cfg.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
