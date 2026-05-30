"""Application entry point.

Runs the FastAPI backend (uvicorn) and, by default, also launches the Next.js
frontend dev server (`npm run dev`) so a single `python main.py` starts both.

Disable the frontend with `--no-frontend` or `NO_FRONTEND=1`.
"""

import atexit
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn

from config import cfg

logger = logging.getLogger("launcher")

_FRONTEND_DIR = Path(__file__).parent / "frontend"
_FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))
_frontend_proc: subprocess.Popen | None = None


def _npm_cmd() -> str | None:
    """Locate npm (npm.cmd on Windows)."""
    return shutil.which("npm") or shutil.which("npm.cmd")


def _frontend_enabled() -> bool:
    if "--no-frontend" in sys.argv:
        return False
    if os.getenv("NO_FRONTEND", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return _FRONTEND_DIR.is_dir()


def start_frontend() -> None:
    """Install deps on first run, then launch `npm run dev` as a child process."""
    global _frontend_proc
    if not _frontend_enabled():
        return

    npm = _npm_cmd()
    if npm is None:
        logger.warning("npm not found on PATH - skipping frontend. Install Node.js to enable it.")
        return

    node_modules = _FRONTEND_DIR / "node_modules"
    next_bin = node_modules / "next" / "dist" / "bin" / "next"

    try:
        if not node_modules.exists():
            # Clean slate - safe to auto-install.
            logger.info("Frontend deps missing - running `npm install` (first run, may take a minute)...")
            subprocess.run([npm, "install"], cwd=_FRONTEND_DIR, check=True)
        elif not next_bin.exists():
            # node_modules exists but looks partial/broken. Do NOT auto-install:
            # a manual `npm install` may be mid-flight, and racing it corrupts
            # the tree (Windows EPERM/ENOTEMPTY). Tell the user and skip frontend.
            logger.warning(
                "Frontend node_modules looks incomplete (next not found). "
                "Skipping frontend launch. Fix it manually:\n"
                "    cd frontend\n"
                "    rmdir /s /q node_modules   (PowerShell: Remove-Item -Recurse -Force node_modules)\n"
                "    npm install\n"
                "Then re-run `python main.py`, or start it yourself with `npm run dev`."
            )
            return

        env = os.environ.copy()
        env.setdefault("BACKEND_URL", f"http://127.0.0.1:{cfg.port}")
        logger.info("Starting frontend dev server on http://localhost:%d ...", _FRONTEND_PORT)
        _frontend_proc = subprocess.Popen(
            [npm, "run", "dev", "--", "-p", str(_FRONTEND_PORT)],
            cwd=_FRONTEND_DIR,
            env=env,
        )
        atexit.register(stop_frontend)
    except Exception as exc:  # never let a frontend hiccup block the backend
        logger.warning("Could not start frontend: %s", exc)
        _frontend_proc = None


def stop_frontend() -> None:
    """Terminate the frontend process (and its child node tree on Windows)."""
    global _frontend_proc
    proc = _frontend_proc
    if proc is None or proc.poll() is not None:
        return
    logger.info("Stopping frontend dev server...")
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception as exc:
        logger.debug("stop_frontend error: %s", exc)
    finally:
        _frontend_proc = None


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

    # Launch the Next.js frontend alongside the backend (unless disabled).
    start_frontend()
    if _frontend_proc is not None:
        logger.info("Backend  : http://localhost:%d   (legacy dashboard at /)", cfg.port)
        logger.info("Frontend : http://localhost:%d   (Next.js UI)", _FRONTEND_PORT)

    try:
        uvicorn.run(
            "api:app",
            host=host,
            port=cfg.port,
            reload=False,
            log_level="warning",  # suppresses uvicorn's own startup banner lines
            access_log=False,  # disables the "GET /... 200 OK" access log entirely
        )
    finally:
        stop_frontend()
