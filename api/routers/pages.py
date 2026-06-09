"""Legacy dashboard HTML pages."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["pages"])

ROOT_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = ROOT_DIR / "static"


def _serve(path: Path) -> HTMLResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.get("/", response_class=HTMLResponse)
async def root():
    return _serve(ROOT_DIR / "templates" / "dashboard.html")


@router.get("/flow", response_class=HTMLResponse)
async def flow():
    """Options flow feed (sweeps & blocks, ask-side conviction)."""
    return _serve(STATIC_DIR / "flow.html")


@router.get("/contract", response_class=HTMLResponse)
async def contract():
    """Single-contract premium history and live buy/sell tracker."""
    return _serve(STATIC_DIR / "contract.html")


@router.get("/portfolio", response_class=HTMLResponse)
async def portfolio():
    return _serve(ROOT_DIR / "templates" / "portfolio_tracker.html")
