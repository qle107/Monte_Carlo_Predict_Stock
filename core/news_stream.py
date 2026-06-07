"""Live news RSS polling and WebSocket broadcast."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from .headline_sentiment import _classify_category, _score_sentiment
from .news_sources import dedup_key, fetch_google_news, fetch_yahoo_rss, fetch_yfinance_news

logger = logging.getLogger(__name__)

_BUFFER_MAX = 50
_POLL_INTERVAL = 20
_MIN_TICKER_CYCLE = 5

_buffers: dict[str, list[dict]] = defaultdict(list)
_seen: dict[str, set[str]] = defaultdict(set)
_subscribers: dict[object, str] = {}
_lock = asyncio.Lock()


def _to_stream_item(ticker: str, article: dict) -> dict:
    return {
        "ticker": ticker,
        "title": article["title"],
        "url": article["url"],
        "source": article["source"],
        "published_iso": article["published"],
        "sentiment": _score_sentiment(article["title"]),
        "category": _classify_category(article["title"], ticker),
    }


def recent(ticker: str) -> list[dict]:
    return list(_buffers.get(ticker.upper(), []))


async def subscribe(ws, ticker: str) -> list[dict]:
    t = ticker.upper()
    async with _lock:
        _subscribers[ws] = t
    return recent(t)


async def unsubscribe(ws) -> None:
    async with _lock:
        _subscribers.pop(ws, None)


def active_tickers() -> set[str]:
    return set(_subscribers.values())


async def poll_loop() -> None:
    logger.info("[news_stream] poll loop started")
    while True:
        try:
            for ticker in active_tickers():
                try:
                    new_items = await _poll_ticker(ticker)
                    if new_items:
                        await _broadcast(ticker, new_items)
                except Exception as exc:
                    logger.debug("[news_stream] poll error for %s: %s", ticker, exc)
                await asyncio.sleep(_MIN_TICKER_CYCLE)
        except asyncio.CancelledError:
            logger.info("[news_stream] poll loop cancelled")
            raise
        except Exception as exc:
            logger.warning("[news_stream] loop error: %s", exc)

        await asyncio.sleep(_POLL_INTERVAL)


async def _poll_ticker(ticker: str) -> list[dict]:
    loop = asyncio.get_running_loop()
    raw: list[dict] = []
    raw.extend(await fetch_yahoo_rss(ticker, cutoff_days=7))
    raw.extend(await fetch_yfinance_news(ticker, loop, cutoff_days=7, limit=12))
    raw.extend(await fetch_google_news(f"{ticker} stock", cutoff_days=7, limit=40))

    new_items: list[dict] = []
    async with _lock:
        seen = _seen[ticker]
        buf = _buffers[ticker]
        for article in raw:
            item = _to_stream_item(ticker, article)
            key = dedup_key(item["title"])
            if key in seen:
                continue
            seen.add(key)
            new_items.append(item)
            buf.insert(0, item)
        if len(buf) > _BUFFER_MAX:
            del buf[_BUFFER_MAX:]
            active_keys = {dedup_key(i["title"]) for i in buf}
            _seen[ticker] = active_keys

    return new_items


async def _broadcast(ticker: str, items: list[dict]) -> None:
    async with _lock:
        subscribers = [ws for ws, t in _subscribers.items() if t == ticker]
    dead: list[object] = []
    payload = {"type": "update", "ticker": ticker, "items": items}
    for ws in subscribers:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    if dead:
        async with _lock:
            for ws in dead:
                _subscribers.pop(ws, None)
