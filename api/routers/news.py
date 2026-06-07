"""Sentiment, news, fear & greed, macro, and insider activity."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from config import cfg
from core.fear_greed import fetch_fear_greed
from core.insider import fetch_insider_activity
from core.macro import fetch_macro_indicators
from core.news_aggregator import fetch_news
from core.sentiment import _cramer_for_sector, fetch_global_market_sentiment, get_sentiment, get_ticker_sector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["news"])


@router.get("/api/sentiment")
async def api_sentiment(ticker: str | None = None, force: int = 0):
    """Social sentiment, Inverse Cramer, and options flow."""
    symbol = (ticker or cfg.ticker).upper().strip()
    try:
        result = await get_sentiment(symbol, force_refresh=bool(force))
        return result
    except Exception as e:
        logger.exception("Sentiment fetch failed for %s", symbol)
        raise HTTPException(status_code=500, detail=f"sentiment failed: {e}")  # noqa: B904


@router.get("/api/sentiment/global")
async def api_global_sentiment(force: int = 0):
    """Market-wide social sentiment."""
    try:
        result = await fetch_global_market_sentiment(force_refresh=bool(force))
        return result
    except Exception as e:
        logger.exception("Global sentiment fetch failed")
        raise HTTPException(status_code=500, detail=f"global sentiment failed: {e}")  # noqa: B904


@router.get("/api/news")
async def api_news(ticker: str | None = None, limit: int = 20):
    """Recent financial news headlines."""
    symbol = (ticker or "").upper().strip()
    return await fetch_news(symbol, limit)


@router.get("/api/fear-greed")
async def api_fear_greed():
    """CNN Fear & Greed Index."""
    return await fetch_fear_greed()


@router.get("/api/macro")
async def api_macro(force: int = 0):
    """Macroeconomic indicators (CPI, yields, unemployment, etc.)."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, fetch_macro_indicators, bool(force))
        return result
    except Exception as e:
        logger.exception("Macro indicators fetch failed")
        raise HTTPException(status_code=500, detail=f"macro fetch failed: {e}")  # noqa: B904


@router.get("/api/sentiment/sector-cramer")
async def api_sector_cramer(ticker: str | None = None, sector: str | None = None):
    """Sector-level Cramer sentiment fallback."""
    resolved_sector = sector
    if not resolved_sector and ticker:
        resolved_sector = get_ticker_sector(ticker.upper())
    if not resolved_sector:
        return {
            "available": False,
            "article_count": 0,
            "cramer_signal": "unknown",
            "inverse_signal": "WAIT",
            "inverse_score": 0.0,
            "confidence": "low",
            "articles": [],
            "source_label": "Sector Cramer Signal",
            "reason": "unknown sector",
        }
    result = await _cramer_for_sector(resolved_sector)
    result["sector"] = resolved_sector
    return result


@router.get("/api/insider-activity")
async def api_insider_activity(ticker: str | None = None, days: int = 30):
    """Recent insider Form 4 filings."""
    symbol = (ticker or cfg.ticker).upper().strip()
    return await fetch_insider_activity(symbol, days)
