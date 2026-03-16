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
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=cfg.port,
        reload=False,
    )
