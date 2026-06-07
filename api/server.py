"""FastAPI app assembly."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import cfg
from core import contract_tracker, news_stream
from core.conformal import BandCalibrator
from core.store import SignalStore

from . import state, websockets
from .analysis import _poll_loop
from .deps import limiter
from .routers import ALL_ROUTERS

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.store = SignalStore(cfg.db_path)
    try:
        state.calibrator = BandCalibrator(cfg.db_path)
    except Exception as exc:
        logger.warning("BandCalibrator init failed: %s", exc)
        state.calibrator = None
    state.analysis_lock = asyncio.Lock()
    state.poll_task = asyncio.create_task(_poll_loop())
    state.news_task = asyncio.create_task(news_stream.poll_loop())
    logger.info("Lifespan startup complete (db=%s)", cfg.db_path)
    try:
        yield
    finally:
        for task in (state.poll_task, state.news_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        contract_tracker.stop_all()
        logger.info("Lifespan shutdown complete")


app = FastAPI(title="MC Trader", lifespan=lifespan)

_raw_cors = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _raw_cors.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

for r in ALL_ROUTERS:
    app.include_router(r)
app.include_router(websockets.router)
