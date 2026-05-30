"""Application entry point."""

import logging
import os

import uvicorn

from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-18s  %(message)s",
)

# Silence third-party loggers
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)

# Silence uvicorn noise (we log startup via lifespan instead)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Silence websocket chatter
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    cors_origins = os.getenv("CORS_ORIGINS", "")
    if not cors_origins:
        logging.getLogger(__name__).warning(
            "CORS_ORIGINS is unset - cross-origin requests will be blocked. "
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
