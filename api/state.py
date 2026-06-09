"""Shared application runtime state."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from core.analysis.conformal import BandCalibrator
from core.data.store import SignalStore

clients: set[WebSocket] = set()
last_result: dict = {}
needs_full_candles: bool = True
poll_task: asyncio.Task | None = None


def payload_for_clients(result: dict, *, full_candles: bool) -> dict:
    """Full OHLCV history on connect/ticker change; poll ticks send the last bar only."""
    if full_candles or not result.get("candles"):
        return result
    slim = dict(result)
    slim["candles"] = [result["candles"][-1]]
    return slim
news_task: asyncio.Task | None = None
store: SignalStore | None = None
calibrator: BandCalibrator | None = None
analysis_lock: asyncio.Lock | None = None
