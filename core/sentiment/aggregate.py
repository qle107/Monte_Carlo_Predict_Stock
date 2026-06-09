"""Per-ticker sentiment aggregation across all sources (cached 5 min)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter
from datetime import datetime, timezone

from .cramer import fetch_cramer_sentiment
from .options_activity import _options_flow_sync, _score_options_activity
from .social import fetch_stocktwits_sentiment

logger = logging.getLogger(__name__)

_SENTIMENT_TTL = 300.0
_sentiment_lock = threading.Lock()
_sentiment_cache: dict[str, tuple] = {}  # ticker -> (result_dict, expire_time)


def _sentiment_cache_get(ticker: str) -> dict | None:
    with _sentiment_lock:
        entry = _sentiment_cache.get(ticker.upper())
        if entry is None:
            return None
        result, exp = entry
        if time.monotonic() > exp:
            del _sentiment_cache[ticker.upper()]
            return None
        return result


def _sentiment_cache_put(ticker: str, result: dict) -> None:
    with _sentiment_lock:
        _sentiment_cache[ticker.upper()] = (result, time.monotonic() + _SENTIMENT_TTL)


async def get_sentiment(ticker: str, force_refresh: bool = False) -> dict:
    """
    Fetch and aggregate all sentiment sources for `ticker`.
    Returns a JSON-serialisable dict.

    Sources:
      - X (Twitter) - per-ticker social posts (requires X_USERNAME + X_PASSWORD in .env)
      - Inverse Cramer - Google News RSS, Cramer signal inverted
      - Options flow  - yfinance call/put volume & OI

    Results are cached for _SENTIMENT_TTL seconds (default 5 min) per ticker.
    Pass force_refresh=True to bypass the cache.
    """
    if not force_refresh:
        cached = _sentiment_cache_get(ticker)
        if cached is not None:
            logger.debug("[sentiment] %s - cache hit", ticker)
            return cached

    loop = asyncio.get_running_loop()

    x_task = asyncio.create_task(fetch_stocktwits_sentiment(ticker))
    cramer_task = asyncio.create_task(fetch_cramer_sentiment(ticker))
    options_coro = loop.run_in_executor(None, _options_flow_sync, ticker)

    x_data, cramer, options = await asyncio.gather(x_task, cramer_task, options_coro, return_exceptions=True)

    if isinstance(x_data, Exception):
        logger.warning("StockTwits sentiment exception: %s", x_data)
        x_data = {
            "available": False,
            "needs_setup": False,
            "sentiment_score": 0.0,
            "call_mentions": 0,
            "put_mentions": 0,
            "post_count": 0,
            "message_count": 0,
            "bull_pct": 0,
            "bear_pct": 0,
            "type_breakdown": {},
            "top_posts": [],
        }
    if isinstance(cramer, Exception):
        logger.warning("Cramer exception: %s", cramer)
        cramer = {
            "available": False,
            "article_count": 0,
            "cramer_signal": "unknown",
            "inverse_signal": "WAIT",
            "inverse_score": 0.0,
            "confidence": "low",
            "buy_signals": 0,
            "sell_signals": 0,
            "type_breakdown": {},
            "articles": [],
        }
    if isinstance(options, Exception):
        logger.warning("Options exception: %s", options)
        options = {"available": False, "reason": str(options)}

    # Weights:
    #   Options activity : 2.0  - real money = strongest signal
    #   Inverse Cramer   : 1.5  - high-conviction contrarian
    #   X / social posts : 1.0  - social mood
    scores, weights = [], []

    if x_data.get("available") and x_data.get("post_count", 0) > 0:
        scores.append(x_data["sentiment_score"])
        weights.append(1.0)

    if cramer.get("available") and cramer.get("inverse_score", 0.0) != 0.0:
        # Cramer score is ALREADY inverted - positive = crowd should BUY.
        scores.append(cramer["inverse_score"])
        weights.append(1.5)

    # Options activity: derived from ATM/OTM volume, call ladder, expiry urgency
    opt_score, opt_label, opt_details = _score_options_activity(options or {})
    if opt_score != 0.0:
        scores.append(opt_score)
        weights.append(2.0)

    agg_score = sum(s * w for s, w in zip(scores, weights, strict=False)) / sum(weights) if scores else 0.0
    agg_score = round(max(-1.0, min(1.0, agg_score)), 4)

    label = "bullish" if agg_score > 0.15 else "bearish" if agg_score < -0.15 else "neutral"

    text_calls = x_data.get("call_mentions", 0)
    text_puts = x_data.get("put_mentions", 0)

    combined_types: Counter = Counter()
    for src in (x_data, cramer):
        for t_type, pct in src.get("type_breakdown", {}).items():
            combined_types[t_type] += pct
    total_type = sum(combined_types.values()) or 1
    merged_type_pcts = {k: round(v / total_type * 100) for k, v in combined_types.items()}

    result = {
        "ticker": ticker.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "score": agg_score,
            "label": label,
            "text_calls": text_calls,
            "text_puts": text_puts,
            "type_breakdown": merged_type_pcts,
            # Options contribution to the aggregate
            "options_score": opt_score,
            "options_label": opt_label,
        },
        # Full breakdown of options-activity signals (for UI display)
        "options_activity": opt_details,
        "x": x_data,
        "cramer": cramer,
        "options_flow": options,
    }
    _sentiment_cache_put(ticker, result)
    return result
