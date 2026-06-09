"""Shared news source fetchers (Yahoo RSS, yfinance, Google News)."""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx
import yfinance as yf

logger = logging.getLogger(__name__)
_UA = {"User-Agent": "Mozilla/5.0"}


def dedup_key(title: str) -> str:
    return hashlib.md5(title[:60].lower().encode()).hexdigest()


def _cutoff(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _parse_yf_item(item: dict, cutoff: datetime) -> dict | None:
    if "content" in item and isinstance(item.get("content"), dict):
        content = item["content"]
        title = (content.get("title") or "").strip()
        url = (content.get("canonicalUrl") or {}).get("url", "")
        source = (content.get("provider") or {}).get("displayName", "Yahoo Finance")
        pub_str = content.get("pubDate", "")
        dt: datetime | None = None
        if pub_str:
            try:
                dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except Exception:
                dt = None
        thumb = ""
        thumbnail = content.get("thumbnail")
        if thumbnail and isinstance(thumbnail, dict):
            resolutions = thumbnail.get("resolutions") or []
            thumb = resolutions[0].get("url", "") if resolutions else ""
    else:
        title = (item.get("title") or "").strip()
        url = item.get("link") or item.get("url", "")
        source = item.get("publisher", "Yahoo Finance")
        pub_ts = item.get("providerPublishTime") or item.get("publish_time")
        dt = None
        if pub_ts:
            try:
                dt = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
            except Exception:
                dt = None
        thumb = (
            (item.get("thumbnail") or {}).get("resolutions", [{}])[0].get("url", "")
            if item.get("thumbnail")
            else ""
        )

    if not title:
        return None
    if dt and dt < cutoff:
        return None
    return {
        "title": title,
        "url": url,
        "source": source,
        "published": dt.isoformat() if dt else "",
        "img": thumb,
    }


async def fetch_yahoo_rss(symbol: str, *, cutoff_days: int = 30) -> list[dict]:
    articles: list[dict] = []
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_UA)
        if resp.status_code != 200:
            return articles
        root = ET.fromstring(resp.text)
        cutoff = _cutoff(cutoff_days)
        for item in root.find("channel") or []:
            if item.tag != "item":
                continue
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_str = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "Yahoo Finance").strip()
            if not title or not link:
                continue
            try:
                dt = parsedate_to_datetime(pub_str).astimezone(timezone.utc) if pub_str else None
            except Exception:
                dt = None
            if dt and dt < cutoff:
                continue
            articles.append(
                {
                    "title": title,
                    "url": link,
                    "source": source,
                    "published": dt.isoformat() if dt else "",
                    "img": "",
                }
            )
    except Exception as exc:
        logger.debug("Yahoo RSS failed for %s: %s", symbol, exc)
    return articles


async def fetch_yfinance_news(
    symbol: str,
    loop,
    *,
    cutoff_days: int = 30,
    limit: int = 15,
) -> list[dict]:
    articles: list[dict] = []
    try:
        cutoff = _cutoff(cutoff_days)

        def _pull():
            return yf.Ticker(symbol).news or []

        for item in (await loop.run_in_executor(None, _pull))[:limit]:
            parsed = _parse_yf_item(item, cutoff)
            if parsed:
                articles.append(parsed)
    except Exception as exc:
        logger.debug("yfinance news failed for %s: %s", symbol, exc)
    return articles


async def fetch_google_news(
    query: str,
    *,
    cutoff_days: int = 30,
    limit: int = 40,
) -> list[dict]:
    articles: list[dict] = []
    try:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=_UA)
        if resp.status_code != 200:
            return articles
        root = ET.fromstring(resp.text)
        cutoff = _cutoff(cutoff_days)
        for item in root.find("channel") or []:
            if item.tag != "item":
                continue
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_str = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "Google News").strip()
            if not title or not link:
                continue
            try:
                dt = parsedate_to_datetime(pub_str).astimezone(timezone.utc) if pub_str else None
            except Exception:
                dt = None
            if dt and dt < cutoff:
                continue
            articles.append(
                {
                    "title": title,
                    "url": link,
                    "source": source,
                    "published": dt.isoformat() if dt else "",
                    "img": "",
                }
            )
            if len(articles) >= limit:
                break
    except Exception as exc:
        logger.debug("Google News RSS failed for %s: %s", query, exc)
    return articles
