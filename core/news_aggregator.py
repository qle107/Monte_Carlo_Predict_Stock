"""Financial news aggregation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import yfinance as yf

from .headline_sentiment import _classify_category, _score_sentiment
from .news_sources import dedup_key, fetch_google_news, fetch_yahoo_rss, fetch_yfinance_news


async def fetch_news(symbol: str, limit: int = 20) -> dict:
    """Aggregate recent financial news for ``symbol`` (empty = market-wide)."""
    loop = asyncio.get_running_loop()
    articles: list[dict] = []

    if symbol:
        articles.extend(await fetch_yahoo_rss(symbol, cutoff_days=30))
    articles.extend(await fetch_yfinance_news(symbol or "SPY", loop, cutoff_days=30, limit=15))
    articles.extend(
        await fetch_google_news(
            f"{symbol} stock" if symbol else "stock market",
            cutoff_days=30,
            limit=40,
        )
    )

    seen: set[str] = set()
    unique: list[dict] = []
    for article in articles:
        key = dedup_key(article["title"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)

    unique.sort(key=lambda x: x["published"] or "0000", reverse=True)

    for article in unique:
        article["sentiment"] = _score_sentiment(article["title"])
        article["category"] = _classify_category(article["title"], ticker=symbol)

    vix = None
    try:

        def _vix():
            info = yf.Ticker("^VIX").fast_info
            return getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)

        vix = await loop.run_in_executor(None, _vix)
        if vix:
            vix = round(float(vix), 2)
    except Exception:
        pass

    return {
        "ticker": symbol or "MARKET",
        "articles": unique[:limit],
        "vix": vix,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
