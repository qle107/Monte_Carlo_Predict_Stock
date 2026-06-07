"""Application entry point."""

import logging
import os

import uvicorn

from config import cfg

logger = logging.getLogger("launcher")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-18s  %(message)s",
)

for _name in ("yfinance", "httpx", "urllib3", "peewee", "uvicorn.access",
              "uvicorn.error", "websockets", "websockets.server"):
    logging.getLogger(_name).setLevel(logging.WARNING)

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    if not os.getenv("CORS_ORIGINS", ""):
        logging.getLogger(__name__).warning(
            "CORS_ORIGINS is unset - set it in .env for cross-origin API clients."
        )

    logger.info("API server: http://%s:%d   (docs at /docs)", host, cfg.port)

    uvicorn.run(
        "api:app",
        host=host,
        port=cfg.port,
        reload=False,
        log_level="warning",
        access_log=False,
    )
