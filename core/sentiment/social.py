"""Reddit and StockTwits sentiment fetchers."""

from __future__ import annotations

import logging
from collections import Counter

import httpx

from .scoring import _classify_sentiment_type, _detect_option_bias, _extract_price_mentions, _score_text

logger = logging.getLogger(__name__)

# Keep Reddit fetcher for global market pulse.
_REDDIT_HEADERS = {"User-Agent": "Mozilla/5.0 mc-trader-bot/1.0 (research)"}
_REDDIT_SUBREDDITS = ["wallstreetbets", "options", "stocks", "investing"]


async def _reddit_fetch_subreddit(client: httpx.AsyncClient, ticker: str, subreddit: str) -> list[dict]:
    """Fetch Reddit posts for a subreddit search."""
    url = (
        f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&sort=new&limit=20&restrict_sr=on&t=day"
    )
    try:
        resp = await client.get(url, headers=_REDDIT_HEADERS, timeout=10)
        resp.raise_for_status()
        children = resp.json()["data"]["children"]
        results = []
        for child in children:
            d = child["data"]
            text = f"{d.get('title', '')} {d.get('selftext', '')}"
            results.append(
                {
                    "source": f"reddit/{subreddit}",
                    "title": d.get("title", "")[:150],
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "score": _score_text(text),
                    "upvotes": d.get("ups", 0),
                    "comments": d.get("num_comments", 0),
                    "call_mentions": _detect_option_bias(text)[0],
                    "put_mentions": _detect_option_bias(text)[1],
                    "sentiment_type": _classify_sentiment_type(text),
                    "price_mentions": _extract_price_mentions(text),
                }
            )
        return results
    except Exception as exc:
        logger.debug("Reddit r/%s/%s failed: %s", subreddit, ticker, exc)
        return []


async def fetch_stocktwits_sentiment(ticker: str) -> dict:
    """Fetch StockTwits stream for ticker (cached 5 min)."""
    result = await _fetch_stocktwits_sentiment(ticker)
    result.setdefault("post_count", result.get("message_count", 0))
    return result


async def _fetch_stocktwits_sentiment(ticker: str) -> dict:
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

        messages = data.get("messages", [])
        bull = bear = neutral = 0
        call_total = put_total = 0
        scored: list[float] = []
        type_counts: Counter = Counter()
        st_price_mentions: list[dict] = []

        for msg in messages:
            body = msg.get("body", "")
            text_score = _score_text(body)
            calls, puts = _detect_option_bias(body)
            sent_type = _classify_sentiment_type(body)
            prices = _extract_price_mentions(body)

            call_total += calls
            put_total += puts
            type_counts[sent_type] += 1
            st_price_mentions.extend(prices)

            label = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic", "")
            if label == "Bullish":
                bull += 1
                effective = max(0.05, text_score)
            elif label == "Bearish":
                bear += 1
                effective = min(-0.05, text_score)
            else:
                neutral += 1
                effective = text_score

            scored.append(effective)

        total = len(messages)
        avg_score = sum(scored) / total if total else 0.0
        bull_pct = round(bull / total * 100) if total else 0
        bear_pct = round(bear / total * 100) if total else 0

        # Lightweight price aggregation from StockTwits
        st_price_counter: Counter = Counter()
        for pm in st_price_mentions:
            key = (round(pm["price"], 2), pm["type"])
            st_price_counter[key] += 1
        st_top_prices = [
            {"price": k[0], "type": k[1], "count": v} for k, v in st_price_counter.most_common(8)
        ]

        type_pcts = {k: round(v / total * 100) for k, v in type_counts.items()} if total else {}

        return {
            "available": True,
            "message_count": total,
            "bullish_count": bull,
            "bearish_count": bear,
            "neutral_count": neutral,
            "sentiment_score": round(avg_score, 4),
            "call_mentions": call_total,
            "put_mentions": put_total,
            "bull_pct": bull_pct,
            "bear_pct": bear_pct,
            "type_breakdown": type_pcts,
            "price_targets": st_top_prices,
        }

    except Exception as exc:
        logger.debug("StockTwits %s failed: %s", ticker, exc)
        return {
            "available": False,
            "message_count": 0,
            "sentiment_score": 0.0,
            "bull_pct": 0,
            "bear_pct": 0,
            "call_mentions": 0,
            "put_mentions": 0,
            "type_breakdown": {},
            "price_targets": [],
        }
