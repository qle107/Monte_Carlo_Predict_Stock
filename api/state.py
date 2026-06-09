"""Shared application runtime state."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from core.analysis.conformal import BandCalibrator
from core.data.store import SignalStore

clients: set[WebSocket] = set()
last_result: dict = {}
poll_task: asyncio.Task | None = None
news_task: asyncio.Task | None = None
store: SignalStore | None = None
calibrator: BandCalibrator | None = None
analysis_lock: asyncio.Lock | None = None
