"""WebSocket endpoints: live analysis push and news stream."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.news import news_stream

from . import state

logger = logging.getLogger(__name__)

router = APIRouter()


async def broadcast_json(clients: Iterable[WebSocket], payload: dict) -> None:
    """Send JSON to all clients; drop sockets that fail."""
    dead: set[WebSocket] = set()
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    if isinstance(clients, set):
        clients.difference_update(dead)


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.add(ws)
    logger.debug("WS client connected (%d total)", len(state.clients))
    if state.last_result:
        with suppress(Exception):
            await ws.send_json(state.payload_for_clients(state.last_result, full_candles=True))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state.clients.discard(ws)
        logger.debug("WS client disconnected (%d total)", len(state.clients))
    except Exception as e:
        state.clients.discard(ws)
        logger.warning("WS error: %s", e)


@router.websocket("/ws/news")
async def ws_news(ws: WebSocket):
    """Live news WebSocket."""
    await ws.accept()
    logger.debug("[ws/news] client connected")
    try:
        while True:
            msg = await ws.receive_json()
            ticker = (msg.get("ticker") or "").upper().strip()
            if not ticker:
                continue
            init_items = await news_stream.subscribe(ws, ticker)
            await ws.send_json(
                {
                    "type": "init",
                    "ticker": ticker,
                    "items": init_items,
                }
            )
    except WebSocketDisconnect:
        logger.debug("[ws/news] client disconnected")
    except Exception as exc:
        logger.warning("[ws/news] error: %s", exc)
    finally:
        await news_stream.unsubscribe(ws)
