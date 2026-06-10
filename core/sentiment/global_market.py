"""Market-wide sentiment from Reddit hot posts and Google News headlines."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone

import httpx

from .scoring import _classify_sentiment_type, _detect_option_bias, _score_text

logger = logging.getLogger(__name__)

# Global market sentiment (no ticker).

_GLOBAL_CACHE_TTL = 600.0  # 10 min (market-wide data changes slower)
_global_cache_lock = threading.Lock()
_global_cache: dict[str, tuple] = {}  # 'GLOBAL' -> (result, expire)

# Hot market themes to detect in posts
_MARKET_THEMES = {
    "Fed / Rates": ["fed", "federal reserve", "fomc", "rate hike", "rate cut", "powell", "interest rate"],
    "Inflation": ["inflation", "cpi", "pce", "deflation", "disinflation", "price", "consumer price"],
    "Recession": ["recession", "gdp", "slowdown", "contraction", "stagflation", "layoffs"],
    "Earnings": ["earnings", "eps", "beat", "miss", "guidance", "revenue", "quarter"],
    "VIX / Fear": ["vix", "volatility", "fear", "panic", "protection", "hedge"],
    "Market Rally": ["rally", "breakout", "squeeze", "rip", "moon", "ath", "all-time high"],
    "Market Crash": ["crash", "correction", "dump", "sell-off", "selloff", "collapse", "drop"],
    "Tech / AI": ["ai", "artificial intelligence", "nvidia", "semiconductor", "chips", "chatgpt"],
    "Options Flow": ["calls", "puts", "options", "spreads", "iron condor", "theta", "gamma"],
}

# Ticker-mention regex - looks for $SYMBOL or plain uppercase 2-5 letter words near $ or %
_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")

# Junk words that get confused for tickers
_TICKER_BLACKLIST = {
    "I",
    "A",
    "AT",
    "BE",
    "DO",
    "GO",
    "IN",
    "IS",
    "IT",
    "ME",
    "MY",
    "NO",
    "OF",
    "ON",
    "OR",
    "SO",
    "TO",
    "UP",
    "US",
    "WE",
    "HE",
    "AN",
    "AI",
    "DD",
    "OP",
    "IV",
    "OI",
    "PE",
    "EV",
    "OK",
    "YO",
    "IF",
    "PM",
    "AM",
    "FY",
    "QQ",
    "QE",
    "MM",
    "YTD",
    "YOY",
    "ATH",
    "HODL",
    "YOLO",
    "FOMO",
    "TLDR",
    "IIRC",
    "AFAIK",
    "IMO",
    "IMHO",
}


async def fetch_global_market_sentiment(force_refresh: bool = False) -> dict:
    """
    Fetch overall stock-market mood from Reddit hot posts + Google News headlines.
    No ticker filter - covers market-wide talk (SPY/QQQ/Fed/VIX/earnings/etc.)

    Returns

    {
        mood_score       : float [-1 bearish ... +1 bullish],
        market_mood      : 'bullish'|'bearish'|'neutral',
        post_count       : int,
        call_mentions    : int,
        put_mentions     : int,
        trending_tickers : [{ticker, count, sentiment}],
        hot_themes       : [{theme, count, pct}],
        cramer_market    : {cramer_signal, inverse_signal, confidence, articles},
        top_posts        : [{title, url, subreddit, score, upvotes, sentiment_type}],
        timestamp        : str,
    }
    """
    import xml.etree.ElementTree as ET

    # Cache check
    if not force_refresh:
        with _global_cache_lock:
            entry = _global_cache.get("GLOBAL")
            if entry:
                result, exp = entry
                if time.monotonic() < exp:
                    return result

    headers_reddit = {
        "User-Agent": "MCTrader/2.0 (market-pulse; research)",
        "Accept": "application/json",
    }
    headers_rss = {
        "User-Agent": "Mozilla/5.0 (compatible; MCTrader/2.0)",
        "Accept": "application/rss+xml, application/xml, */*",
    }

    _GLOBAL_SUBS = [
        ("wallstreetbets", "hot"),
        ("stocks", "hot"),
        ("investing", "hot"),
        ("options", "hot"),
        ("StockMarket", "hot"),
    ]

    all_posts: list[dict] = []
    ticker_counter: Counter = Counter()
    theme_counter: Counter = Counter()
    total_bull = total_bear = 0
    call_total = put_total = 0

    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        # Reddit hot posts
        for sub, sort in _GLOBAL_SUBS:
            try:
                url = f"https://www.reddit.com/r/{sub}/{sort}.json?limit=25&t=day"
                resp = await client.get(url, headers=headers_reddit)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                children = data.get("data", {}).get("children", [])

                for child in children:
                    p = child.get("data", {})
                    title = p.get("title", "")
                    body = p.get("selftext", "")
                    text = f"{title} {body}"

                    score = _score_text(text)
                    calls, puts = _detect_option_bias(text)
                    sent_type = _classify_sentiment_type(text)

                    call_total += calls
                    put_total += puts
                    if score > 0.1:
                        total_bull += 1
                    elif score < -0.1:
                        total_bear += 1

                    # Trending tickers ($AAPL, $NVDA etc.)
                    for m in _TICKER_RE.findall(text.upper()):
                        if m not in _TICKER_BLACKLIST and 2 <= len(m) <= 5:
                            ticker_counter[m] += 1

                    # Hot themes
                    tl = text.lower()
                    for theme, kws in _MARKET_THEMES.items():
                        if any(kw in tl for kw in kws):
                            theme_counter[theme] += 1

                    all_posts.append(
                        {
                            "title": title,
                            "url": p.get("url", ""),
                            "subreddit": f"r/{sub}",
                            "score": round(score, 3),
                            "upvotes": int(p.get("score", 0)),
                            "sentiment_type": sent_type,
                            "call_mentions": calls,
                            "put_mentions": puts,
                        }
                    )

            except Exception as exc:
                logger.debug("Global Reddit r/%s failed: %s", sub, exc)

        # Google News RSS - market-wide headlines
        rss_queries = [
            "stock+market+today",
            "S%26P+500+today",
            "Wall+Street+market",
        ]
        for q in rss_queries:
            try:
                rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
                resp = await client.get(rss_url, headers=headers_rss)
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.text)
                items = root.findall(".//item")
                for item in items[:10]:
                    title = item.findtext("title", "") or ""
                    link = item.findtext("link", "") or ""
                    desc = item.findtext("description", "") or ""
                    text = f"{title} {desc}"

                    score = _score_text(text)
                    calls, puts = _detect_option_bias(text)
                    call_total += calls
                    put_total += puts
                    if score > 0.1:
                        total_bull += 1
                    elif score < -0.1:
                        total_bear += 1

                    tl = text.lower()
                    for theme, kws in _MARKET_THEMES.items():
                        if any(kw in tl for kw in kws):
                            theme_counter[theme] += 1

                    for m in _TICKER_RE.findall(text.upper()):
                        if m not in _TICKER_BLACKLIST and 2 <= len(m) <= 5:
                            ticker_counter[m] += 1

                    all_posts.append(
                        {
                            "title": title,
                            "url": link,
                            "subreddit": "📰 News",
                            "score": round(score, 3),
                            "upvotes": 0,
                            "sentiment_type": _classify_sentiment_type(text),
                            "call_mentions": calls,
                            "put_mentions": puts,
                        }
                    )
            except Exception as exc:
                logger.debug("Global RSS %s failed: %s", q, exc)

    cramer_market = {
        "cramer_signal": "unknown",
        "inverse_signal": "WAIT",
        "confidence": "low",
        "article_count": 0,
        "articles": [],
        "picks": [],
    }

    n_posts = len(all_posts)
    total_sent = total_bull + total_bear or 1
    bull_pct = round(total_bull / total_sent * 100)
    bear_pct = round(total_bear / total_sent * 100)

    raw_score = (total_bull - total_bear) / total_sent if total_sent else 0.0
    # Clamp and label
    mood_score = max(-1.0, min(1.0, raw_score))
    market_mood = "bullish" if mood_score > 0.15 else "bearish" if mood_score < -0.15 else "neutral"

    # Trending tickers - top 15 by mention count
    trending = [
        {
            "ticker": t,
            "count": c,
            "sentiment": sum(p["score"] for p in all_posts if t.lower() in (p.get("title", "") or "").lower())
            / max(sum(1 for p in all_posts if t.lower() in (p.get("title", "") or "").lower()), 1),
        }
        for t, c in ticker_counter.most_common(15)
    ]

    # Hot themes - percentage of posts mentioning each theme
    total_theme_posts = max(n_posts, 1)
    hot_themes = [
        {"theme": th, "count": cnt, "pct": round(cnt / total_theme_posts * 100)}
        for th, cnt in theme_counter.most_common(8)
        if cnt > 0
    ]

    # Top posts by upvotes
    top_posts = sorted(all_posts, key=lambda x: x["upvotes"], reverse=True)[:15]

    result = {
        "mood_score": round(mood_score, 4),
        "market_mood": market_mood,
        "bull_pct": bull_pct,
        "bear_pct": bear_pct,
        "post_count": n_posts,
        "call_mentions": call_total,
        "put_mentions": put_total,
        "trending_tickers": trending,
        "hot_themes": hot_themes,
        "cramer_market": cramer_market,
        "top_posts": top_posts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with _global_cache_lock:
        _global_cache["GLOBAL"] = (result, time.monotonic() + _GLOBAL_CACHE_TTL)

    return result
