"""
main.py — Entry point.

Run:  python main.py
Open: http://localhost:8000 (or http://<HOST>:<PORT> if HOST/PORT are set)

Host binding:
  Default HOST=127.0.0.1 (localhost only — safe for local use).
  Set HOST=0.0.0.0 in .env to allow LAN / Docker access.
  Never expose 0.0.0.0 to the public internet without a firewall or reverse
  proxy (nginx/Caddy) in front and API_KEY set.
"""

import logging
import os

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
    host = os.getenv("HOST", "127.0.0.1")
    cors_origins = os.getenv("CORS_ORIGINS", "")
    if not cors_origins:
        logging.getLogger(__name__).warning(
            "CORS_ORIGINS is unset — cross-origin requests will be blocked. "
            "Set CORS_ORIGINS=http://localhost:%d in .env if needed.",
            cfg.port,
        )
    uvicorn.run(
        "api:app",
        host=host,
        port=cfg.port,
        reload=False,
        log_level="warning",  # suppresses uvicorn's own startup banner lines
        access_log=False,  # disables the "GET /... 200 OK" access log entirely
    )
