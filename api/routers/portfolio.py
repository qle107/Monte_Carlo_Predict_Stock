"""Portfolio price helpers (historical close + live price)."""

from __future__ import annotations

import asyncio
import logging
from datetime import date as _date
from datetime import timedelta

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio"])


@router.get("/api/portfolio/price/historical")
async def portfolio_price_historical(ticker: str, date: str):
    """Closing price for `ticker` on `date` (YYYY-MM-DD)."""
    try:
        target = _date.fromisoformat(date)
        # Fetch a window around the target date to handle weekends/holidays
        start = (target - timedelta(days=5)).isoformat()
        end = (target + timedelta(days=2)).isoformat()
        df = await asyncio.get_running_loop().run_in_executor(
            None, lambda: yf.download(ticker.upper(), start=start, end=end, progress=False, auto_adjust=True)
        )
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        # Get the closest trading day on or before target
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
        target_ts = pd.Timestamp(target)
        available = df.index[df.index <= target_ts]
        if available.empty:
            available = df.index  # fallback: use first available
        row = df.loc[available[-1]]
        close = float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"])
        return {"ticker": ticker.upper(), "date": str(available[-1].date()), "close": round(close, 4)}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("portfolio historical price failed for %s on %s: %s", ticker, date, e)
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@router.get("/api/portfolio/price/live")
async def portfolio_price_live(ticker: str):
    """Latest market price for `ticker`."""
    try:
        t = yf.Ticker(ticker.upper())
        info = await asyncio.get_running_loop().run_in_executor(None, lambda: t.fast_info)
        price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        if price is None:
            # fallback: grab last close from 5d history
            df = await asyncio.get_running_loop().run_in_executor(
                None, lambda: t.history(period="5d", auto_adjust=True)
            )
            if not df.empty:
                price = float(df["Close"].iloc[-1])
        if price is None:
            raise HTTPException(status_code=404, detail=f"No live price for {ticker}")
        return {"ticker": ticker.upper(), "price": round(float(price), 4)}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("portfolio live price failed for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904
