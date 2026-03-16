"""
api/server.py — FastAPI application, all routes and WebSocket.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import cfg, VALID_INTERVALS
from core import analyse
from core.fetcher import fetch_candles
from .models import ConfigUpdate

logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="MC Trader")

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# State
clients:      Set[WebSocket] = set()
_last_result: dict           = {}
_poll_task:   asyncio.Task   = None


# ── HTML ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path(__file__).parent.parent / "templates" / "dashboard.html"
    return HTMLResponse(html.read_text())


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/api/signal")
async def get_signal():
    """Trigger fresh analysis with current config and return result."""
    result = await _run_analysis()
    if "error" not in result:
        await _broadcast(result)
    return result


@app.get("/api/config")
async def get_config():
    return {
        "ticker":          cfg.ticker,
        "interval":        cfg.interval,
        "n_sim":           cfg.n_sim,
        "n_forward":       cfg.n_forward,
        "lookback":        cfg.lookback,
        "poll_seconds":    cfg.poll_seconds,
        "extended":        cfg.extended,
        "valid_intervals": VALID_INTERVALS,
    }


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    global _poll_task
    changed = []

    if update.ticker       is not None: cfg.ticker       = update.ticker.upper().strip(); changed.append(f"ticker={cfg.ticker}")
    if update.interval     is not None: cfg.interval     = update.interval;               changed.append(f"interval={cfg.interval}")
    if update.n_sim        is not None: cfg.n_sim        = update.n_sim;                  changed.append(f"n_sim={cfg.n_sim}")
    if update.n_forward    is not None: cfg.n_forward    = update.n_forward;              changed.append(f"n_forward={cfg.n_forward}")
    if update.lookback     is not None: cfg.lookback     = update.lookback;               changed.append(f"lookback={cfg.lookback}")
    if update.poll_seconds is not None: cfg.poll_seconds = update.poll_seconds;           changed.append(f"poll_seconds={cfg.poll_seconds}")
    if update.extended     is not None: cfg.extended     = update.extended;               changed.append(f"extended={cfg.extended}");           changed.append(f"poll_seconds={cfg.poll_seconds}")

    logger.info(f"Config updated: {', '.join(changed)}")

    # Restart poll loop with new settings
    if _poll_task and not _poll_task.done():
        _poll_task.cancel()
    _poll_task = asyncio.create_task(_poll_loop())

    # Immediate fresh analysis
    result = await _run_analysis()
    if "error" not in result:
        await _broadcast(result)

    return {"status": "ok", "changed": changed, "config": await get_config()}


@app.get("/api/health")
async def health():
    return {"status": "ok", "ticker": cfg.ticker, "interval": cfg.interval,
            "timestamp": datetime.now(timezone.utc).isoformat()}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    logger.info(f"WS client connected ({len(clients)} total)")
    if _last_result:
        try:
            await ws.send_json(_last_result)
        except Exception:
            pass
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
        logger.info(f"WS client disconnected ({len(clients)} total)")


async def _broadcast(data: dict):
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ── Analysis ─────────────────────────────────────────────────────────────────

async def _run_analysis() -> dict:
    global _last_result
    try:
        loop = asyncio.get_event_loop()
        df   = await loop.run_in_executor(
            None, fetch_candles, cfg.ticker, cfg.interval, cfg.lookback, cfg.extended
        )
        result = await loop.run_in_executor(
            None, analyse, df, cfg.n_sim, cfg.n_forward
        )
        result.update({
            "ticker":     cfg.ticker,
            "interval":   cfg.interval,
            "extended":   cfg.extended,
            "config":     {"n_sim": cfg.n_sim, "n_forward": cfg.n_forward,
                           "lookback": cfg.lookback, "poll_seconds": cfg.poll_seconds, "extended": cfg.extended},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        _last_result = result

        sig = result["signal"]
        logger.info(
            f"{cfg.ticker} {cfg.interval}  "
            f"price={result['current_price']:.2f}  "
            f"signal={sig['label']} (confidence={sig['confidence']:.0%})  "
            f"up={result['mc']['prob_up']}% down={result['mc']['prob_down']}%"
            + (f"  ⚠ {result['warnings'][0]}" if result.get('warnings') else "")
        )
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": str(e)}


# ── Poll loop ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _poll_task
    _poll_task = asyncio.create_task(_poll_loop())


async def _poll_loop():
    logger.info(f"Poll loop started: {cfg.ticker} {cfg.interval} every {cfg.poll_seconds}s")
    while True:
        result = await _run_analysis()
        if "error" not in result:
            await _broadcast(result)
        await asyncio.sleep(cfg.poll_seconds)
