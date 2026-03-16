"""
api/models.py — Pydantic request/response models.
"""

from typing import Optional
from pydantic import BaseModel, field_validator
from config import VALID_INTERVALS


class ConfigUpdate(BaseModel):
    ticker:       Optional[str] = None
    interval:     Optional[str] = None
    n_sim:        Optional[int] = None
    n_forward:    Optional[int] = None
    lookback:     Optional[int] = None
    poll_seconds: Optional[int] = None
    extended:     Optional[bool] = None

    @field_validator("interval")
    @classmethod
    def valid_interval(cls, v):
        if v and v not in VALID_INTERVALS:
            raise ValueError(f"interval must be one of {VALID_INTERVALS}")
        return v

    @field_validator("n_sim")
    @classmethod
    def valid_nsim(cls, v):
        if v and not (50 <= v <= 2000):
            raise ValueError("n_sim must be 50–2000")
        return v

    @field_validator("n_forward")
    @classmethod
    def valid_nfwd(cls, v):
        if v and not (1 <= v <= 50):
            raise ValueError("n_forward must be 1–50")
        return v

    @field_validator("lookback")
    @classmethod
    def valid_lookback(cls, v):
        if v and not (20 <= v <= 200):
            raise ValueError("lookback must be 20–200")
        return v

    @field_validator("poll_seconds")
    @classmethod
    def valid_poll(cls, v):
        if v and not (10 <= v <= 3600):
            raise ValueError("poll_seconds must be 10–3600")
        return v
