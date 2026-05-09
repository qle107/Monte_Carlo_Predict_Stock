"""
main.py — Entry point.

Run:  python main.py
Open: http://localhost:8000
"""

import logging

import uvicorn

from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-18s  %(message)s",
)

# ── Silence third-party libraries ──────────────────────────────────────────
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)

# ── Silence uvicorn noise ───────────────────────────────────────────────────
# uvicorn.access  → all the "GET /api/... 200 OK" and "WebSocket /ws [accepted]"
# uvicorn.error   → "Started server process", "Waiting for application startup",
#                   "Application startup complete", "Uvicorn running on ..."
#                   (we log our own startup banner from lifespan instead)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# ── Silence websocket connection open/close chatter ─────────────────────────
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=cfg.port,
        reload=False,
        log_level="warning",   # suppresses uvicorn's own startup banner lines
        access_log=False,      # disables the "GET /... 200 OK" access log entirely
    )
