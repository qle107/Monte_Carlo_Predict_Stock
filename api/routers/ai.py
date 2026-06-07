"""AI Analyst endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import re

from fastapi import APIRouter, HTTPException, Request

from config import cfg
from core.ai_analyst import build_prompt, run_ai_analysis

from ..deps import limiter
from .. import state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai"])

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")

_ai_lock = asyncio.Lock()


@router.get("/api/ai/status")
async def api_ai_status():
    """Whether the AI analyst is configured."""
    return {
        "configured": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "model": os.getenv("CLAUDE_MODEL", "claude-opus-4-8").strip(),
        "busy": _ai_lock.locked(),
    }


@router.get("/api/ai/prompt")
@limiter.limit("6/minute")
async def api_ai_prompt(request: Request, ticker: str | None = None):
    """Build a copy-paste prompt from current API data."""
    t = (ticker or cfg.ticker).upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="invalid ticker")
    try:
        return await asyncio.wait_for(build_prompt(t, state.last_result), timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="data gathering timed out (>120s)")  # noqa: B904
    except Exception as e:
        logger.exception("[ai] prompt build failed for %s", t)
        raise HTTPException(status_code=500, detail=f"prompt build failed: {e}")  # noqa: B904


@router.get("/api/ai/analyze")
@limiter.limit("4/minute")
async def api_ai_analyze(request: Request, ticker: str | None = None):
    """Run Claude analysis on aggregated API data."""
    t = (ticker or cfg.ticker).upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=400, detail="invalid ticker")

    if _ai_lock.locked():
        raise HTTPException(status_code=429, detail="An AI analysis is already running - wait for it to finish.")

    async with _ai_lock:
        try:
            result = await asyncio.wait_for(
                run_ai_analysis(t, state.last_result),
                timeout=300.0,
            )
            return result
        except asyncio.TimeoutError:
            logger.error("[ai] analysis timed out for %s", t)
            raise HTTPException(status_code=504, detail="AI analysis timed out (>300s)")  # noqa: B904
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))  # noqa: B904
        except Exception as e:
            logger.exception("[ai] analysis failed for %s", t)
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {e}")  # noqa: B904
