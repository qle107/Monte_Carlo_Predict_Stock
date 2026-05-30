"""Shared application runtime state."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from core.store import SignalStore


class AppState:
    """Container for process-wide runtime state."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.last_result: dict = {}
        self.poll_task: asyncio.Task | None = None
        self.news_stream_task: asyncio.Task | None = None
        self.store: SignalStore | None = None
        # Guards _run_analysis - created inside the running loop in lifespan.
        self.analysis_lock: asyncio.Lock | None = None


state = AppState()
