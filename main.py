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
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# Reduce noise from third-party libs
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=cfg.port,
        reload=False,
        log_level="info",
    )
