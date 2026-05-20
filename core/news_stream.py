"""
core/news_stream.py — Live news streaming: background RSS poll + WebSocket broadcast.

Architecture
------------
* A single asyncio task (run_loop) polls Yahoo Finance + Google News RSS every 20 s.
* Each item is deduplicated by md5(title[:60].lower()) — the same scheme used in
  /api/news so the two feeds won't diverge on dedup logic.
* Per-ticker ring buffers (max 50 items) keep a recent-headline window.
* WebSocket clients register via subscribe(ws, ticker).  When the background loop
  finds a new item it fans out only to subscribers watching that ticker.
* Sentiment + category classification uses lightweight keyword helpers defined here
  (extracted from api/server.py so both paths share identical logic).

Item shape (every item emitted over the wire):
    {
        "ticker":        str,
        "title":         str,
        "url":           str,
        "source":        str,
        "published_iso": str,   # ISO-8601 UTC or ""
        "sentiment":     "Positive" | "Negative" | "Neutral",
        "category":      "Company" | "Macro" | "Sector" | "General",
    }
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Sentiment + category keyword sets ─────────────────────────────────────────
# Kept in sync with the originals in api/server.py.

_SENTIMENT_POSITIVE: set[str] = {
    "beat",
    "beats",
    "surge",
    "surges",
    "rally",
    "rallies",
    "gain",
    "gains",
    "growth",
    "grew",
    "profit",
    "profits",
    "record",
    "upgrade",
    "upgraded",
    "bullish",
    "outperform",
    "strong",
    "strength",
    "boom",
    "booming",
    "breakthrough",
    "positive",
    "rise",
    "rises",
    "rose",
    "higher",
    "upbeat",
    "recovery",
    "recover",
    "momentum",
    "accelerate",
    "accelerates",
    "expansion",
    "exceeds",
    "exceed",
    "topped",
    "tops",
    "above expectations",
    "above forecast",
}

_SENTIMENT_NEGATIVE: set[str] = {
    "miss",
    "misses",
    "drop",
    "drops",
    "dropped",
    "fall",
    "falls",
    "fell",
    "loss",
    "losses",
    "decline",
    "declines",
    "declined",
    "bearish",
    "downgrade",
    "downgraded",
    "underperform",
    "weak",
    "weakness",
    "recession",
    "crash",
    "crashing",
    "fear",
    "risk",
    "risks",
    "cut",
    "cuts",
    "layoff",
    "layoffs",
    "below expectations",
    "below forecast",
    "disappoints",
    "disappointing",
    "concern",
    "concerns",
    "warning",
    "warns",
    "slump",
    "slumps",
    "plunge",
    "plunges",
    "correction",
    "sell-off",
    "selloff",
    "contraction",
    "default",
    "bankruptcy",
    "debt",
    "inflation",
    "stagflation",
    "tariff",
    "tariffs",
}

_MACRO_KEYWORDS: set[str] = {
    "fed",
    "federal reserve",
    "fomc",
    "rate",
    "rates",
    "inflation",
    "cpi",
    "ppi",
    "pce",
    "gdp",
    "unemployment",
    "jobs",
    "payroll",
    "treasury",
    "yield",
    "yields",
    "recession",
    "economy",
    "economic",
    "interest rate",
    "bls",
    "bea",
    "ism",
    "pmi",
    "debt ceiling",
    "fiscal",
    "monetary",
    "jerome powell",
    "powell",
    "central bank",
    "quantitative",
    "tapering",
}

_SECTOR_KEYWORDS: dict[str, set[str]] = {
    "tech": {
        "technology",
        "software",
        "chip",
        "semiconductor",
        "ai",
        "cloud",
        "nvidia",
        "apple",
        "google",
        "microsoft",
        "meta",
        "amazon",
        "tesla",
    },
    "energy": {
        "oil",
        "gas",
        "energy",
        "opec",
        "crude",
        "refinery",
        "exxon",
        "chevron",
        "lng",
        "pipeline",
        "coal",
        "renewables",
        "solar",
        "wind",
    },
    "financials": {
        "bank",
        "banking",
        "jpmorgan",
        "goldman",
        "morgan stanley",
        "credit",
        "loan",
        "lending",
        "fintech",
        "insurance",
        "brokerage",
    },
    "healthcare": {
        "fda",
        "drug",
        "pharma",
        "biotech",
        "clinical",
        "trial",
        "approval",
        "vaccine",
        "healthcare",
        "hospital",
        "medical",
    },
}

_BUFFER_MAX = 50
_POLL_INTERVAL = 20  # seconds between poll cycles per ticker
_MIN_TICKER_CYCLE = 5  # seconds between polling individual tickers in a cycle


def _score_sentiment(title: str, summary: str = "") -> str:
    """Keyword-based sentiment classifier.  Returns 'Positive', 'Negative', or 'Neutral'."""
    text = (title + " " + summary).lower()
    pos = sum(1 for w in _SENTIMENT_POSITIVE if w in text)
    neg = sum(1 for w in _SENTIMENT_NEGATIVE if w in text)
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def _classify_category(title: str, ticker: str = "") -> str:
    """Classify article into: 'Company', 'Macro', 'Sector', or 'General'."""
    text = title.lower()
    if ticker and ticker.lower() in text:
        return "Company"
    if any(kw in text for kw in _MACRO_KEYWORDS):
        return "Macro"
    for _, keywords in _SECTOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return "Sector"
    return "General"


def _dedup_key(title: str) -> str:
    """md5 of first 60 chars of lowercased title — same as /api/news."""
    return hashlib.md5(title[:60].lower().encode()).hexdigest()


# ── NewsStream ─────────────────────────────────────────────────────────────────


class NewsStream:
    """
    Singleton-ish object that owns the polling loop and subscription state.
    Instantiated once at import time; wired into the FastAPI lifespan.
    """

    def __init__(self) -> None:
        # Per-ticker ring buffer: ticker → list[item] (newest-first, max 50)
        self._buffers: dict[str, list[dict]] = defaultdict(list)
        # Per-ticker seen-set for dedup
        self._seen: dict[str, set[str]] = defaultdict(set)
        # WS client → subscribed ticker
        self._subscribers: dict[object, str] = {}  # WebSocket → ticker
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def recent(self, ticker: str) -> list[dict]:
        """Return current buffer for ticker (newest first, up to 50 items)."""
        return list(self._buffers.get(ticker.upper(), []))

    async def subscribe(self, ws, ticker: str) -> list[dict]:
        """
        Register (or update) a WebSocket client's ticker subscription.
        Returns the current buffer so the caller can send an init burst.
        """
        t = ticker.upper()
        async with self._lock:
            self._subscribers[ws] = t
        return self.recent(t)

    async def unsubscribe(self, ws) -> None:
        async with self._lock:
            self._subscribers.pop(ws, None)

    def active_tickers(self) -> set[str]:
        """Set of tickers currently watched by at least one WS client."""
        return set(self._subscribers.values())

    # ── Background loop ────────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """
        Asyncio task: poll news for every subscribed ticker every POLL_INTERVAL seconds.
        New items are broadcast immediately to subscribed clients.
        """
        logger.info("[news_stream] poll loop started")
        while True:
            try:
                tickers = self.active_tickers()
                if tickers:
                    for t in tickers:
                        try:
                            new_items = await self._poll_ticker(t)
                            if new_items:
                                await self._broadcast(t, new_items)
                        except Exception as exc:
                            logger.debug("[news_stream] poll error for %s: %s", t, exc)
                        # Brief pause between individual ticker polls
                        await asyncio.sleep(_MIN_TICKER_CYCLE)
            except asyncio.CancelledError:
                logger.info("[news_stream] poll loop cancelled — shutting down")
                raise
            except Exception as exc:
                logger.warning("[news_stream] unexpected loop error: %s", exc)

            await asyncio.sleep(_POLL_INTERVAL)

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _poll_ticker(self, ticker: str) -> list[dict]:
        """
        Fetch fresh news for `ticker` from yfinance + Google News RSS.
        Returns only items not already in the buffer (new-to-us).
        """
        loop = asyncio.get_running_loop()
        raw: list[dict] = []

        # 1. yfinance news (quick, blocking, offloaded to thread pool)
        try:

            def _yf():
                t_obj = yf.Ticker(ticker)
                return t_obj.news or []

            yf_items = await loop.run_in_executor(None, _yf)
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            for item in yf_items[:12]:
                pub_ts = item.get("providerPublishTime") or item.get("publish_time")
                dt: datetime | None = None
                if pub_ts:
                    try:
                        dt = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
                    except Exception:
                        dt = None
                if dt and dt < cutoff:
                    continue
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                raw.append(
                    {
                        "ticker": ticker,
                        "title": title,
                        "url": item.get("link") or item.get("url", ""),
                        "source": item.get("publisher", "Yahoo Finance"),
                        "published_iso": dt.isoformat() if dt else "",
                        "sentiment": _score_sentiment(title),
                        "category": _classify_category(title, ticker),
                    }
                )
        except Exception as exc:
            logger.debug("[news_stream] yfinance error for %s: %s", ticker, exc)

        # 2. Google News RSS
        try:
            query = f"{ticker} stock"
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(rss_url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                for item in root.find("channel") or []:
                    if item.tag != "item":
                        continue
                    title = (item.findtext("title") or "").strip()
                    url = (item.findtext("link") or "").strip()
                    pub_str = (item.findtext("pubDate") or "").strip()
                    source = (item.findtext("source") or "Google News").strip()
                    if not title or not url:
                        continue
                    try:
                        dt = parsedate_to_datetime(pub_str).astimezone(timezone.utc) if pub_str else None
                    except Exception:
                        dt = None
                    if dt and dt < cutoff:
                        continue
                    raw.append(
                        {
                            "ticker": ticker,
                            "title": title,
                            "url": url,
                            "source": source,
                            "published_iso": dt.isoformat() if dt else "",
                            "sentiment": _score_sentiment(title),
                            "category": _classify_category(title, ticker),
                        }
                    )
                    if len(raw) >= 40:
                        break
        except Exception as exc:
            logger.debug("[news_stream] Google News RSS error for %s: %s", ticker, exc)

        # 3. Dedup against seen set and update buffer
        new_items: list[dict] = []
        async with self._lock:
            seen = self._seen[ticker]
            buf = self._buffers[ticker]
            for item in raw:
                key = _dedup_key(item["title"])
                if key in seen:
                    continue
                seen.add(key)
                new_items.append(item)
                buf.insert(0, item)  # newest first
            # Trim buffer to max size
            if len(buf) > _BUFFER_MAX:
                buf[_BUFFER_MAX:]
                del buf[_BUFFER_MAX:]
                # Also prune the seen-set to avoid unbounded growth
                # (keep only keys that are still in the buffer)
                active_keys = {_dedup_key(i["title"]) for i in buf}
                self._seen[ticker] = active_keys

        return new_items  # list ordered newest-first

    async def _broadcast(self, ticker: str, items: list[dict]) -> None:
        """Fan-out new items to all clients subscribed to `ticker`."""
        dead = []
        async with self._lock:
            subscribers = {ws for ws, t in self._subscribers.items() if t == ticker}
        for ws in subscribers:
            try:
                await ws.send_json(
                    {
                        "type": "update",
                        "ticker": ticker,
                        "items": items,
                    }
                )
            except Exception:
                dead.append(ws)
        # Clean up broken connections
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subscribers.pop(ws, None)


# Module-level singleton — imported by api/server.py
news_stream = NewsStream()
