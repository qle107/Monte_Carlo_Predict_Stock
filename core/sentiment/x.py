"""X/Twitter sentiment via twikit (cookies in x_cookies.json after first login).

NOTE: dormant without an X account. All functions degrade gracefully to
'unavailable' results when twikit / credentials / cookies are missing.
"""

from __future__ import annotations

import asyncio
import logging
import os as _os
from collections import Counter

from .scoring import _classify_sentiment_type, _detect_option_bias, _extract_price_mentions, _score_text

logger = logging.getLogger(__name__)

# Project root is two levels up from core/sentiment/.
_TWIKIT_COOKIES = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "x_cookies.json")
)

_twikit_client: object | None = None
_twikit_lock: asyncio.Lock | None = None  # created lazily inside async ctx


async def _get_twikit_client() -> object | None:
    """
    Return a logged-in twikit.Client singleton.

    Auth priority

    1. x_cookies.json exists -> load it and skip login entirely.
       This is the recommended path: export cookies from your browser and
       place them at  <project_root>/x_cookies.json  (see README).
    2. x_cookies.json not found AND X_USERNAME + X_PASSWORD are set -> do a
       fresh password login and save the resulting cookies for next time.

    Cookie file format (create manually from browser if password login fails):
        {
            "auth_token": "<value of auth_token cookie on x.com>",
            "ct0":        "<value of ct0 cookie on x.com>"
        }
    Get these from: Chrome F12 -> Application -> Cookies -> https://x.com
    """
    global _twikit_client, _twikit_lock
    if _twikit_client is not None:
        return _twikit_client

    if _twikit_lock is None:
        _twikit_lock = asyncio.Lock()

    async with _twikit_lock:
        if _twikit_client is not None:
            return _twikit_client

        try:
            from twikit import Client  # type: ignore
        except ImportError:
            logger.warning("[X] twikit not installed - run: pip install twikit")
            return None

        import os

        client = Client("en-US")

        if _os.path.exists(_TWIKIT_COOKIES):
            try:
                client.load_cookies(_TWIKIT_COOKIES)
                _twikit_client = client
                logger.info("[X] twikit - loaded cookies from %s", _TWIKIT_COOKIES)
                return client
            except Exception as exc:
                logger.warning("[X] Failed to load cookie file: %s", exc)
                # Don't auto-delete: user may have manually created the file.
                # Just fall through to password login.

        username = os.getenv("X_USERNAME", "").strip().lstrip("@")
        password = os.getenv("X_PASSWORD", "").strip().strip("\"'")
        email = os.getenv("X_EMAIL", "").strip()

        if not username or not password:
            logger.warning(
                "[X] No cookie file found at %s and no credentials in .env. "
                "Either create the cookie file manually or set X_USERNAME + X_PASSWORD.",
                _TWIKIT_COOKIES,
            )
            return None

        try:
            logger.info("[X] twikit - attempting password login as @%s ...", username)
            login_kwargs: dict = {"auth_info_1": username, "password": password}
            if email:
                login_kwargs["auth_info_2"] = email
            await client.login(**login_kwargs)
            client.save_cookies(_TWIKIT_COOKIES)
            _twikit_client = client
            logger.info("[X] twikit - login OK, cookies saved to %s", _TWIKIT_COOKIES)
            return client
        except Exception as exc:
            logger.warning(
                "[X] twikit password login failed for @%s: %s\n"
                "  -> Fix: export cookies from your browser and save to %s\n"
                "  -> Get auth_token + ct0 from: F12 -> Application -> Cookies -> x.com",
                username,
                exc,
                _TWIKIT_COOKIES,
            )
            return None


def _twikit_reset() -> None:
    """
    Discard the cached client so the next call reloads cookies / re-logs in.
    Does NOT delete the cookie file - the user may have manually created it.
    If cookies have expired, the next search call will fail and log a warning
    prompting the user to refresh their browser cookies.
    """
    global _twikit_client
    _twikit_client = None
    logger.debug("[X] twikit client reset - will reload on next request")


async def fetch_x_sentiment(ticker: str) -> dict:
    """
    Fetch recent X (Twitter) posts mentioning $TICKER via twikit (free scraping).

    Setup (one-time)

    1. pip install twikit
    2. Add X_USERNAME=your_handle  and  X_PASSWORD=your_pass  to .env
    3. Use a burner account - NOT your main account.

    Returns

    {
        available       : bool,
        needs_setup     : bool,   # True when credentials are missing
        post_count      : int,
        sentiment_score : float,
        call_mentions   : int,
        put_mentions    : int,
        type_breakdown  : dict,
        price_targets   : list,
        top_posts       : list[{title, url, source, score, likes, retweets,
                                 replies, call_mentions, put_mentions,
                                 sentiment_type, price_mentions}],
    }
    """
    import os
    import random

    has_creds = bool(
        os.getenv("X_USERNAME", "").strip().lstrip("@") and os.getenv("X_PASSWORD", "").strip().strip("\"'")
    )

    # needs_setup = True ONLY when credentials are literally absent or twikit missing
    _no_creds = {
        "available": False,
        "needs_setup": True,
        "post_count": 0,
        "sentiment_score": 0.0,
        "call_mentions": 0,
        "put_mentions": 0,
        "type_breakdown": {},
        "top_posts": [],
    }
    # credentials present but something went wrong -> needs_setup stays False
    _failed = {**_no_creds, "needs_setup": False}

    if not has_creds:
        logger.debug("X_USERNAME/X_PASSWORD not set - X sentiment unavailable")
        return _no_creds

    # Check twikit is importable before trying login
    try:
        import twikit  # noqa: F401  type: ignore
    except ImportError:
        logger.warning("[X] twikit not installed - run: pip install twikit")
        return _no_creds  # show setup notice - package is literally missing

    client = await _get_twikit_client()
    if client is None:
        # Credentials set, twikit installed, but login failed
        return {**_failed, "error": "Login failed - check X_USERNAME / X_PASSWORD in .env"}

    try:
        from twikit.errors import TooManyRequests  # type: ignore
    except ImportError:
        TooManyRequests = Exception

    query = f"${ticker} -filter:retweets lang:en"

    try:
        await asyncio.sleep(random.uniform(0.5, 1.5))  # polite delay
        results = await client.search_tweet(query, product="Top", count=20)
    except TooManyRequests:
        logger.warning("[X] Rate limited while fetching $%s - backing off", ticker)
        return {**_failed, "error": "Rate limited - try again in a few minutes"}
    except Exception as exc:
        err_str = str(exc)
        # KEY_BYTE / key_byte is a known twikit internal parse error triggered when
        # Twitter changes their internal API structure.  It is not actionable and
        # very noisy, so log at DEBUG instead of WARNING.
        if "KEY_BYTE" in err_str or "key_byte" in err_str.lower():
            logger.debug(
                "[X] twikit parse error for $%s (Twitter API drift - not actionable): %s",
                ticker,
                err_str[:80],
            )
        else:
            logger.warning("[X] Search failed for $%s: %s", ticker, err_str[:120])
        _twikit_reset()  # force re-login next call in case session expired
        return _failed

    tweets = list(results) if results else []
    if not tweets:
        return {**_failed, "available": True, "needs_setup": False, "post_count": 0}

    posts: list[dict] = []
    total_weight = 0.0
    weighted_score = 0.0
    call_total = put_total = 0
    type_counts: Counter = Counter()

    for tw in tweets:
        text = getattr(tw, "text", "") or ""
        likes = int(getattr(tw, "favorite_count", 0) or 0)
        retweets = int(getattr(tw, "retweet_count", 0) or 0)
        replies = int(getattr(tw, "reply_count", 0) or 0)

        # Weight by engagement - high-engagement tweets carry more signal
        weight = max(1, likes + retweets * 2)

        score = _score_text(text)
        calls, puts = _detect_option_bias(text)
        sent_type = _classify_sentiment_type(text)
        price_mentions = _extract_price_mentions(text)

        weighted_score += score * weight
        total_weight += weight
        call_total += calls
        put_total += puts
        type_counts[sent_type] += 1

        tweet_id = getattr(tw, "id", "")
        user = getattr(tw, "user", None)
        screen_name = getattr(user, "screen_name", "") if user else ""
        tweet_url = (
            f"https://x.com/{screen_name}/status/{tweet_id}"
            if screen_name and tweet_id
            else f"https://x.com/i/web/status/{tweet_id}"
            if tweet_id
            else ""
        )

        posts.append(
            {
                "title": text[:200],
                "url": tweet_url,
                "source": "X",
                "score": round(score, 3),
                "likes": likes,
                "retweets": retweets,
                "replies": replies,
                "call_mentions": calls,
                "put_mentions": puts,
                "sentiment_type": sent_type,
                "price_mentions": price_mentions,
            }
        )

    n = len(posts)
    avg_score = weighted_score / total_weight if total_weight else 0.0
    top_posts = sorted(posts, key=lambda x: x["likes"] + x["retweets"] * 2, reverse=True)[:15]
    type_pcts = {k: round(v / n * 100) for k, v in type_counts.items()}

    return {
        "available": True,
        "needs_setup": False,
        "post_count": n,
        "sentiment_score": round(avg_score, 4),
        "call_mentions": call_total,
        "put_mentions": put_total,
        "type_breakdown": type_pcts,
        "top_posts": top_posts,
    }
