"""
core/sentiment.py — Social sentiment + options flow aggregator.

Sources
-------
* Reddit         — r/wallstreetbets, r/options, r/stocks, r/investing
                   Public JSON API (no auth, no API key required)
* Inverse Cramer — Google News RSS for Jim Cramer / CNBC Mad Money mentions
                   Detects Cramer's buy/sell signal then INVERTS it
* Options flow   — yfinance call/put volume & open interest (real market data)

Extracted signals per post/message
-----------------------------------
* Sentiment score  [-1 bullish … +1 bearish]
* Sentiment type   stock | options | mixed | general
* Price mentions   list of {price, type: call/put/stock, raw, sentiment}
* Call / Put mention counts

All network I/O is async (httpx). The options flow call is blocking (yfinance)
and is offloaded to a thread-pool executor.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx
import yfinance as yf

logger = logging.getLogger(__name__)

# ── In-memory TTL cache ───────────────────────────────────────────────────────
# Sentiment data is slow to fetch (Reddit + Cramer/Google News + yfinance options).
# Cache results per ticker for 5 minutes so the dashboard / scanner can call
# get_sentiment() multiple times without hammering the public APIs.

_SENTIMENT_TTL  = 300.0          # 5 minutes
_sentiment_lock = threading.Lock()
_sentiment_cache: Dict[str, tuple] = {}   # ticker → (result_dict, expire_time)


def _sentiment_cache_get(ticker: str) -> Optional[dict]:
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

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment scoring
# ─────────────────────────────────────────────────────────────────────────────

_BULL_WORDS: List[str] = [
    "bullish", "moon", "mooning", "calls", "long", "buy", "buying",
    "breakout", "squeeze", "yolo", "hodl", "undervalued", "accumulate",
    "strong", "support", "bounce", "rocket", "gains", "green", "ripping",
    "rip", "pump", "pumping", "beat", "upgrade", "upside", "higher",
    "🚀", "💎", "🙌", "📈",
]

_BEAR_WORDS: List[str] = [
    "bearish", "puts", "short", "sell", "selling", "breakdown", "dump",
    "dumping", "crash", "crashing", "overvalued", "weak", "resistance",
    "fall", "falling", "drop", "dropping", "red", "miss", "downgrade",
    "downside", "lower", "hedge", "fear", "risky", "bag", "bagholder",
    "🐻", "💀", "📉",
]


def _score_text(text: str) -> float:
    """Return [-1, +1] sentiment score. Positive = bullish."""
    t = text.lower()
    bull = sum(1 for w in _BULL_WORDS if w in t)
    bear = sum(1 for w in _BEAR_WORDS if w in t)
    total = bull + bear
    return 0.0 if total == 0 else (bull - bear) / total


def _detect_option_bias(text: str) -> Tuple[int, int]:
    """Return (call_mentions, put_mentions)."""
    t = text.lower()
    calls = len(re.findall(r'\bcalls?\b', t))
    puts  = len(re.findall(r'\bputs?\b',  t))
    return calls, puts


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment type classification  (stock vs options vs mixed)
# ─────────────────────────────────────────────────────────────────────────────

_OPTIONS_SIGNALS: List[str] = [
    "call", "calls", "put", "puts", "option", "options", "strike",
    "expir", "expiry", "theta", "delta", "gamma", "vega", "iv",
    "implied volatility", "otm", "itm", "atm", "leaps", "contract",
    "contracts", "premium", "0dte", "0 dte", "dte", "spread",
    "covered call", "cash secured put", "iron condor", "straddle",
    "strangle", "debit", "credit", "butterfly", "condor",
]

_STOCK_SIGNALS: List[str] = [
    "share", "shares", "stock", "stonk", "stonks", "holding", "held",
    "dividend", "earnings", "eps", "revenue", "guidance", "buyback",
    "pe ratio", "market cap", "float", "short interest", "catalyst",
    "price target", "pt ", " pt$", "analyst", "upgrade", "downgrade",
]


def _classify_sentiment_type(text: str) -> str:
    """
    Classify the discussion type: 'stock', 'options', 'mixed', or 'general'.
    """
    t = text.lower()
    opt   = sum(1 for w in _OPTIONS_SIGNALS if w in t)
    stock = sum(1 for w in _STOCK_SIGNALS   if w in t)
    total = opt + stock
    if total == 0:
        return "general"
    ratio = opt / total
    if ratio >= 0.65:
        return "options"
    if ratio <= 0.35:
        return "stock"
    return "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# Price / strike extraction
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (compiled regex, option_type or None, group_index_for_price)
# option_type: 'call' | 'put' | 'stock'
_PRICE_REGEXES: List[Tuple[re.Pattern, str, int]] = [
    # ── Options — highest specificity first ──────────────────────────────────
    # "$150c", "$200p", "$150C", "$200P"
    (re.compile(r'\$(\d{1,5}(?:\.\d{1,2})?)\s*([Cc])\b'),       'call',  1),
    (re.compile(r'\$(\d{1,5}(?:\.\d{1,2})?)\s*([Pp])\b'),       'put',   1),
    # "150c", "200p", "150C", "200P" (bare chain notation ≥2 digits)
    (re.compile(r'\b(\d{2,5})([Cc])\b'),                          'call',  1),
    (re.compile(r'\b(\d{2,5})([Pp])\b'),                          'put',   1),
    # "$150 calls", "$200 puts", "$150 call"
    (re.compile(r'\$(\d{1,5}(?:\.\d{1,2})?)\s+calls?\b', re.I),  'call',  1),
    (re.compile(r'\$(\d{1,5}(?:\.\d{1,2})?)\s+puts?\b',  re.I),  'put',   1),
    # "150 calls", "200 puts" (no dollar sign)
    (re.compile(r'\b(\d{1,5}(?:\.\d{1,2})?)\s+calls?\b', re.I),  'call',  1),
    (re.compile(r'\b(\d{1,5}(?:\.\d{1,2})?)\s+puts?\b',  re.I),  'put',   1),
    # "strike of $150 / strike 150"
    (re.compile(r'strike\s+(?:of\s+)?\$?(\d{1,5}(?:\.\d{1,2})?)', re.I), 'call', 1),
    # ── Stock price ──────────────────────────────────────────────────────────
    # "PT $150", "PT: $150", "price target $150"
    (re.compile(r'(?:price\s+target|pt)[:\s]+\$?(\d{1,5}(?:\.\d{1,2})?)', re.I), 'stock', 1),
    # "target $150", "targeting $150", "target of $150"
    (re.compile(r'target(?:ing)?\s+(?:of\s+)?\$(\d{1,5}(?:\.\d{1,2})?)', re.I), 'stock', 1),
    # "to $150", "at $150", "@ $150", "around $150"
    (re.compile(r'(?:to|at|@|around)\s+\$(\d{1,5}(?:\.\d{1,2})?)', re.I), 'stock', 1),
    # "support at 150", "resistance at 150"
    (re.compile(r'(?:support|resistance|level)\s+(?:at\s+)?\$?(\d{1,5}(?:\.\d{1,2})?)', re.I), 'stock', 1),
    # Generic "$NNN" fallback — lowest priority
    (re.compile(r'\$(\d{1,5}(?:\.\d{1,2})?)'), 'stock', 1),
]

_PRICE_MIN = 0.5    # filter out sub-cent
_PRICE_MAX = 25_000 # filter out obviously wrong numbers


def _extract_price_mentions(text: str) -> List[Dict]:
    """
    Extract all price / strike level mentions from `text`.

    Returns a deduplicated list of:
      {price: float, type: 'call'|'put'|'stock', raw: str}
    ordered from most-specific to least-specific.
    """
    seen: set = set()
    results: List[Dict] = []

    for pattern, ptype, grp in _PRICE_REGEXES:
        for m in pattern.finditer(text):
            try:
                price = float(m.group(grp).replace(',', ''))
            except (ValueError, IndexError):
                continue
            if not (_PRICE_MIN <= price <= _PRICE_MAX):
                continue
            # round to 2 dp for dedup key
            key = (round(price, 2), ptype)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "price": round(price, 2),
                "type":  ptype,
                "raw":   m.group(0)[:30],
            })

    return results



# ─────────────────────────────────────────────────────────────────────────────
# X (Twitter) — per-ticker social sentiment via twikit (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────
# Uses twikit to simulate a logged-in browser session.
# Setup (one-time):
#   1. pip install twikit
#   2. Add X_USERNAME=your_handle  and  X_PASSWORD=your_password  to .env
#   3. Use a burner/alt account — NOT your main account.
#
# After first login, cookies are saved to x_cookies.json in the project root
# so subsequent restarts don't need to log in again.
# ─────────────────────────────────────────────────────────────────────────────

import os as _os
_TWIKIT_COOKIES = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '..', 'x_cookies.json')
)

_twikit_client:    Optional[object]       = None
_twikit_lock:      Optional[asyncio.Lock] = None   # created lazily inside async ctx


async def _get_twikit_client() -> Optional[object]:
    """
    Return a logged-in twikit.Client singleton.

    Auth priority
    -------------
    1. x_cookies.json exists → load it and skip login entirely.
       This is the recommended path: export cookies from your browser and
       place them at  <project_root>/x_cookies.json  (see README).
    2. x_cookies.json not found AND X_USERNAME + X_PASSWORD are set → do a
       fresh password login and save the resulting cookies for next time.

    Cookie file format (create manually from browser if password login fails):
        {
            "auth_token": "<value of auth_token cookie on x.com>",
            "ct0":        "<value of ct0 cookie on x.com>"
        }
    Get these from: Chrome F12 → Application → Cookies → https://x.com
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
            logger.warning("[X] twikit not installed — run: pip install twikit")
            return None

        import os
        client = Client('en-US')

        # ── Path 1: cookie file ───────────────────────────────────────────────
        if _os.path.exists(_TWIKIT_COOKIES):
            try:
                client.load_cookies(_TWIKIT_COOKIES)
                _twikit_client = client
                logger.info("[X] twikit — loaded cookies from %s", _TWIKIT_COOKIES)
                return client
            except Exception as exc:
                logger.warning("[X] Failed to load cookie file: %s", exc)
                # Don't auto-delete: user may have manually created the file.
                # Just fall through to password login.

        # ── Path 2: password login ────────────────────────────────────────────
        username = os.getenv('X_USERNAME', '').strip().lstrip('@')
        password = os.getenv('X_PASSWORD', '').strip().strip('"\'')
        email    = os.getenv('X_EMAIL',    '').strip()

        if not username or not password:
            logger.warning(
                "[X] No cookie file found at %s and no credentials in .env. "
                "Either create the cookie file manually or set X_USERNAME + X_PASSWORD.",
                _TWIKIT_COOKIES,
            )
            return None

        try:
            logger.info("[X] twikit — attempting password login as @%s …", username)
            login_kwargs: dict = {"auth_info_1": username, "password": password}
            if email:
                login_kwargs["auth_info_2"] = email
            await client.login(**login_kwargs)
            client.save_cookies(_TWIKIT_COOKIES)
            _twikit_client = client
            logger.info("[X] twikit — login OK, cookies saved to %s", _TWIKIT_COOKIES)
            return client
        except Exception as exc:
            logger.warning(
                "[X] twikit password login failed for @%s: %s\n"
                "  → Fix: export cookies from your browser and save to %s\n"
                "  → Get auth_token + ct0 from: F12 → Application → Cookies → x.com",
                username, exc, _TWIKIT_COOKIES,
            )
            return None


def _twikit_reset() -> None:
    """
    Discard the cached client so the next call reloads cookies / re-logs in.
    Does NOT delete the cookie file — the user may have manually created it.
    If cookies have expired, the next search call will fail and log a warning
    prompting the user to refresh their browser cookies.
    """
    global _twikit_client
    _twikit_client = None
    logger.debug("[X] twikit client reset — will reload on next request")


# Keep Reddit fetcher available for the global market pulse feed (no key needed)
_REDDIT_HEADERS    = {"User-Agent": "Mozilla/5.0 mc-trader-bot/1.0 (research)"}
_REDDIT_SUBREDDITS = ["wallstreetbets", "options", "stocks", "investing"]


async def _reddit_fetch_subreddit(
    client: httpx.AsyncClient, ticker: str, subreddit: str
) -> List[Dict]:
    """Used by the Global Market Pulse feed — NOT the per-ticker sentiment."""
    url = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        f"?q={ticker}&sort=new&limit=20&restrict_sr=on&t=day"
    )
    try:
        resp = await client.get(url, headers=_REDDIT_HEADERS, timeout=10)
        resp.raise_for_status()
        children = resp.json()["data"]["children"]
        results = []
        for child in children:
            d    = child["data"]
            text = f"{d.get('title', '')} {d.get('selftext', '')}"
            results.append({
                "source":         f"reddit/{subreddit}",
                "title":          d.get("title", "")[:150],
                "url":            f"https://reddit.com{d.get('permalink', '')}",
                "score":          _score_text(text),
                "upvotes":        d.get("ups", 0),
                "comments":       d.get("num_comments", 0),
                "call_mentions":  _detect_option_bias(text)[0],
                "put_mentions":   _detect_option_bias(text)[1],
                "sentiment_type": _classify_sentiment_type(text),
                "price_mentions": _extract_price_mentions(text),
            })
        return results
    except Exception as exc:
        logger.debug("Reddit r/%s/%s failed: %s", subreddit, ticker, exc)
        return []


async def fetch_x_sentiment(ticker: str) -> dict:
    """
    Fetch recent X (Twitter) posts mentioning $TICKER via twikit (free scraping).

    Setup (one-time)
    ----------------
    1. pip install twikit
    2. Add X_USERNAME=your_handle  and  X_PASSWORD=your_pass  to .env
    3. Use a burner account — NOT your main account.

    Returns
    -------
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
    import os, random

    has_creds = bool(
        os.getenv('X_USERNAME', '').strip().lstrip('@') and
        os.getenv('X_PASSWORD', '').strip().strip('"\'')
    )

    # needs_setup = True ONLY when credentials are literally absent or twikit missing
    _no_creds = {
        "available": False, "needs_setup": True,
        "post_count": 0, "sentiment_score": 0.0,
        "call_mentions": 0, "put_mentions": 0,
        "type_breakdown": {}, "top_posts": [],
    }
    # credentials present but something went wrong → needs_setup stays False
    _failed = {**_no_creds, "needs_setup": False}

    if not has_creds:
        logger.debug("X_USERNAME/X_PASSWORD not set — X sentiment unavailable")
        return _no_creds

    # Check twikit is importable before trying login
    try:
        import twikit  # noqa: F401  type: ignore
    except ImportError:
        logger.warning("[X] twikit not installed — run: pip install twikit")
        return _no_creds   # show setup notice — package is literally missing

    client = await _get_twikit_client()
    if client is None:
        # Credentials set, twikit installed, but login failed
        return {**_failed, "error": "Login failed — check X_USERNAME / X_PASSWORD in .env"}

    try:
        from twikit.errors import TooManyRequests, BadRequest  # type: ignore
    except ImportError:
        TooManyRequests = BadRequest = Exception

    query = f"${ticker} -filter:retweets lang:en"

    try:
        await asyncio.sleep(random.uniform(0.5, 1.5))   # polite delay
        results = await client.search_tweet(query, product='Top', count=20)
    except TooManyRequests:
        logger.warning("[X] Rate limited while fetching $%s — backing off", ticker)
        return {**_failed, "error": "Rate limited — try again in a few minutes"}
    except Exception as exc:
        err_str = str(exc)
        # KEY_BYTE / key_byte is a known twikit internal parse error triggered when
        # Twitter changes their internal API structure.  It is not actionable and
        # very noisy, so log at DEBUG instead of WARNING.
        if 'KEY_BYTE' in err_str or 'key_byte' in err_str.lower():
            logger.debug("[X] twikit parse error for $%s (Twitter API drift — not actionable): %s",
                         ticker, err_str[:80])
        else:
            logger.warning("[X] Search failed for $%s: %s", ticker, err_str[:120])
        _twikit_reset()   # force re-login next call in case session expired
        return _failed

    tweets = list(results) if results else []
    if not tweets:
        return {**_empty, "available": True, "needs_setup": False, "post_count": 0}

    posts:         List[Dict] = []
    total_weight   = 0.0
    weighted_score = 0.0
    call_total = put_total = 0
    type_counts: Counter = Counter()

    for tw in tweets:
        text     = getattr(tw, 'text', '') or ''
        likes    = int(getattr(tw, 'favorite_count', 0) or 0)
        retweets = int(getattr(tw, 'retweet_count',  0) or 0)
        replies  = int(getattr(tw, 'reply_count',    0) or 0)

        # Weight by engagement — high-engagement tweets carry more signal
        weight = max(1, likes + retweets * 2)

        score          = _score_text(text)
        calls, puts    = _detect_option_bias(text)
        sent_type      = _classify_sentiment_type(text)
        price_mentions = _extract_price_mentions(text)

        weighted_score += score * weight
        total_weight   += weight
        call_total     += calls
        put_total      += puts
        type_counts[sent_type] += 1

        tweet_id    = getattr(tw, 'id', '')
        user        = getattr(tw, 'user', None)
        screen_name = getattr(user, 'screen_name', '') if user else ''
        tweet_url   = (
            f"https://x.com/{screen_name}/status/{tweet_id}"
            if screen_name and tweet_id else
            f"https://x.com/i/web/status/{tweet_id}" if tweet_id else ""
        )

        posts.append({
            "title":          text[:200],
            "url":            tweet_url,
            "source":         "X",
            "score":          round(score, 3),
            "likes":          likes,
            "retweets":       retweets,
            "replies":        replies,
            "call_mentions":  calls,
            "put_mentions":   puts,
            "sentiment_type": sent_type,
            "price_mentions": price_mentions,
        })

    n         = len(posts)
    avg_score = weighted_score / total_weight if total_weight else 0.0
    top_posts = sorted(posts, key=lambda x: x["likes"] + x["retweets"] * 2,
                       reverse=True)[:15]
    type_pcts = {k: round(v / n * 100) for k, v in type_counts.items()}

    return {
        "available":       True,
        "needs_setup":     False,
        "post_count":      n,
        "sentiment_score": round(avg_score, 4),
        "call_mentions":   call_total,
        "put_mentions":    put_total,
        "type_breakdown":  type_pcts,
        "top_posts":       top_posts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Jim Cramer  —  Inverse Cramer sentiment  (Google News RSS, no auth required)
# ─────────────────────────────────────────────────────────────────────────────
# The "Inverse Cramer" strategy: whatever Cramer says to do, do the opposite.
# We scrape Google News RSS for "[Jim Cramer] [TICKER]" headlines, detect his
# bullish/bearish stance per article, then flip the signal for the UI.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Cramer stance keyword lists  (ordered from strongest signal → weakest)
# ─────────────────────────────────────────────────────────────────────────────

_CRAMER_BULLISH_KW: List[str] = [
    # Superlatives / iconic Cramer phrases
    "buy buy buy", "screaming buy", "strong buy", "must own",
    "back up the truck", "buying opportunity", "buy the dip",
    "great opportunity", "once-in-a-lifetime",
    # Standard buy
    "buy", "buying", "bought", "bullish", "go long", "long",
    # Positive verbs
    "like", "likes", "love", "loves", "fan of", "big fan",
    "recommend", "recommends", "recommending", "endorses",
    # Upgrades & price-target raises
    "upgrade", "upgraded", "outperform", "overweight",
    "raised price target", "price target raised", "raised target",
    # Ownership / accumulation
    "own", "owns", "owning", "holding", "adding", "accumulate",
    "accumulating", "staying long", "still long",
    # Positive framing
    "still a buy", "still like", "undervalued", "upside",
    "opportunity", "positive on", "backing", "backs",
    "great stock", "solid", "winner",
]

_CRAMER_BEARISH_KW: List[str] = [
    # Superlatives / iconic Cramer phrases
    "sell sell sell", "avoid at all costs", "run from", "get out now",
    "terrible stock", "avoid like the plague",
    # Standard sell
    "sell", "selling", "sold", "bearish", "short", "shorting",
    # Negative verbs
    "hate", "hates", "hated", "avoid", "avoiding",
    "not a fan", "don't like", "doesn't like", "no longer likes",
    "no longer", "wrong call",
    # Downgrades & target cuts
    "downgrade", "downgraded", "underperform", "underweight",
    "cut target", "price target cut", "lowered target", "lowered price target",
    # Risk / exit signals
    "risky", "too risky", "concerned", "worried", "worrying",
    "stay away", "take profit", "take profits",
    "dump", "exit", "reduce exposure", "trimming", "cut position",
    # Negative framing
    "overvalued", "peaked", "too expensive", "vulnerable",
    "danger", "warning", "negative on",
]

_CRAMER_NEUTRAL_KW: List[str] = [
    "watch", "watching", "wait", "waiting", "monitor", "monitoring",
    "keep an eye", "uncertain", "mixed", "neutral",
    "on the fence", "not sure", "could go either",
    "interesting situation", "considering",
]

# Regex to find stock tickers embedded in article text
# Covers:  (NYSE: AAPL)  (Nasdaq: NVDA)  (NYSEARCA: SPY)  $AAPL
_ARTICLE_TICKER_RE = re.compile(
    r'(?:'
    r'\((?:NYSE|NASDAQ|Nasdaq|AMEX|NYSEARCA|NYSEMKT|BATS):\s*([A-Z]{1,5})\)'
    r'|\$([A-Z]{1,5})\b'
    r')'
)

# Words that look like tickers but aren't
_ARTICLE_TICKER_BL: set = {
    "A", "I", "AM", "AN", "AT", "BE", "DO", "GO", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "AI", "DD", "OP", "IV", "OI", "PE", "EV", "OK", "PM", "FY", "QQ",
    "ATH", "IPO", "ETF", "CEO", "CFO", "COO", "SEC", "FED", "GDP",
    "CPI", "PPI", "PCE", "EPS", "FCF", "TTM", "YOY", "IMO", "TLDR",
}


def _score_cramer_text(text: str) -> Tuple[int, int]:
    """Return (buy_signals, sell_signals) — kept for backward-compat with global fn."""
    t = text.lower()
    buys  = sum(1 for w in _CRAMER_BULLISH_KW  if w in t)
    sells = sum(1 for w in _CRAMER_BEARISH_KW if w in t)
    return buys, sells


def _classify_stance_from_window(window: str) -> Tuple[str, str]:
    """
    Classify Cramer's stance on a stock from the text window around its mention.

    Returns
    -------
    (stance, evidence)
    stance ∈ {'bullish', 'bearish', 'neutral', 'unknown'}
    evidence : first 120 chars of the window (for display)
    """
    w = window.lower()
    bull = sum(1 for kw in _CRAMER_BULLISH_KW if kw in w)
    bear = sum(1 for kw in _CRAMER_BEARISH_KW if kw in w)
    neut = sum(1 for kw in _CRAMER_NEUTRAL_KW if kw in w)

    evidence = window.strip()[:140].replace("\n", " ")

    if bull == 0 and bear == 0 and neut == 0:
        return "unknown", evidence
    if bull > bear and bull >= neut:
        return "bullish", evidence
    if bear > bull and bear >= neut:
        return "bearish", evidence
    return "neutral", evidence


def _extract_cramer_picks(full_text: str) -> List[Dict]:
    """
    Scan article text for ticker symbols and classify Cramer's stance on each.

    Strategy
    --------
    1. Find every ticker via _ARTICLE_TICKER_RE  (exchange notation + $TICKER)
    2. Take ±400 chars around each match as context window
    3. Score the window against bullish/bearish/neutral keyword lists
    4. Deduplicate per ticker — keep the strongest signal seen across all windows

    Returns list of {ticker, stance, evidence} dicts.
    """
    strength = {"bullish": 3, "bearish": 3, "neutral": 2, "unknown": 1}
    picks: Dict[str, Dict] = {}

    for match in _ARTICLE_TICKER_RE.finditer(full_text):
        ticker = (match.group(1) or match.group(2) or "").upper().strip()
        if not ticker or ticker in _ARTICLE_TICKER_BL or len(ticker) < 2:
            continue

        start  = max(0, match.start() - 400)
        end    = min(len(full_text), match.end() + 400)
        window = full_text[start:end]

        stance, evidence = _classify_stance_from_window(window)

        if ticker not in picks or strength[stance] > strength[picks[ticker]["stance"]]:
            picks[ticker] = {"ticker": ticker, "stance": stance, "evidence": evidence}

    return list(picks.values())


async def _fetch_article_text(url: str, client: httpx.AsyncClient) -> str:
    """
    Fetch a news article URL and return its plain text (HTML stripped).
    Returns empty string on any failure.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":          "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = await client.get(url, headers=headers, timeout=8.0,
                                follow_redirects=True)
        if resp.status_code != 200:
            return ""
        html = resp.text
        # Strip <script> and <style> blocks entirely
        html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>",
                      " ", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:10_000]   # cap — enough for stance extraction
    except Exception:
        return ""


async def fetch_cramer_sentiment(ticker: str) -> dict:
    """
    Fetch Jim Cramer's recent coverage of `ticker` from Google News RSS.

    For each article found:
      1. Score the title + description for buy/sell keywords (fast pass).
      2. Fetch the full article body (up to 5 articles) and run
         _extract_cramer_picks() to find EVERY ticker Cramer mentioned
         with a stance classified from surrounding context.
      3. Collect picks for THIS ticker specifically.
      4. Invert the overall signal (Inverse Cramer strategy).

    Returns
    -------
    {
        available        : bool,
        article_count    : int,
        cramer_signal    : 'bullish'|'bearish'|'mixed'|'unknown',
        inverse_signal   : 'BUY'|'SELL'|'WAIT',
        inverse_score    : float [-1 … +1],
        confidence       : 'high'|'medium'|'low',
        ticker_stance    : 'bullish'|'bearish'|'neutral'|'unknown',
        buy_signals      : int,
        sell_signals     : int,
        type_breakdown   : dict,
        picks            : list[{ticker, stance, evidence}],   # all tickers in articles
        articles         : list[{title, url, date, cramer_bias, picks, price_mentions}],
    }
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    rss_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MCTrader/2.0)",
        "Accept":     "application/rss+xml, application/xml, text/xml, */*",
    }

    rss_urls = [
        f'https://news.google.com/rss/search?q=%22Jim+Cramer%22+%22{ticker}%22&hl=en-US&gl=US&ceid=US:en',
        f'https://news.google.com/rss/search?q=Cramer+{ticker}+CNBC&hl=en-US&gl=US&ceid=US:en',
        f'https://news.google.com/rss/search?q=%22Mad+Money%22+{ticker}&hl=en-US&gl=US&ceid=US:en',
    ]

    raw_items: List[Dict] = []   # items from RSS before article fetch

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:

        # ── Step 1: RSS scan ────────────────────────────────────────────────
        for rss_url in rss_urls:
            try:
                resp = await client.get(rss_url, headers=rss_headers)
                if resp.status_code != 200:
                    continue
                root  = ET.fromstring(resp.text)
                for item in root.findall(".//item")[:20]:
                    title = item.findtext("title",       "") or ""
                    link  = item.findtext("link",        "") or ""
                    pub   = item.findtext("pubDate",     "") or ""
                    desc  = item.findtext("description", "") or ""
                    tl    = f"{title} {desc}".lower()
                    if "cramer" not in tl and "mad money" not in tl:
                        continue
                    try:
                        pub_str = parsedate_to_datetime(pub).strftime("%Y-%m-%d")
                    except Exception:
                        pub_str = pub[:10] if pub else ""
                    raw_items.append({"title": title, "url": link,
                                      "date": pub_str, "desc": desc})
                if raw_items:
                    break
            except Exception as exc:
                logger.debug("Cramer RSS %s: %s", rss_url, exc)

        # ── Step 2: Fetch full article body for top 5 items ─────────────────
        articles:    List[Dict] = []
        all_picks_map: Dict[str, Dict] = {}   # ticker → best pick across all articles
        total_buy = total_sell = 0
        type_counts: Counter = Counter()
        strength_order = {"bullish": 3, "bearish": 3, "neutral": 2, "unknown": 1}

        fetch_tasks = [
            _fetch_article_text(it["url"], client)
            for it in raw_items[:5]
        ]
        bodies = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for it, body in zip(raw_items[:5], bodies):
            body_text = body if isinstance(body, str) else ""
            # Combine RSS snippet + full body for maximum coverage
            combined  = f"{it['title']} {it['desc']} {body_text}"

            b, s = _score_cramer_text(combined)
            total_buy  += b
            total_sell += s
            cramer_bias = "bullish" if b > s else "bearish" if s > b else "neutral"
            type_counts[_classify_sentiment_type(combined)] += 1
            prices = _extract_price_mentions(combined)

            # Extract all per-ticker picks from the article
            art_picks = _extract_cramer_picks(combined)
            for pk in art_picks:
                t = pk["ticker"]
                if t not in all_picks_map or (
                    strength_order[pk["stance"]] > strength_order[all_picks_map[t]["stance"]]
                ):
                    all_picks_map[t] = pk

            articles.append({
                "title":          it["title"],
                "url":            it["url"],
                "date":           it["date"],
                "cramer_bias":    cramer_bias,
                "buy_signals":    b,
                "sell_signals":   s,
                "picks":          art_picks,
                "price_mentions": prices,
            })

        # For remaining RSS items beyond top-5, score title+desc only (no fetch)
        for it in raw_items[5:]:
            combined = f"{it['title']} {it['desc']}"
            b, s = _score_cramer_text(combined)
            total_buy  += b
            total_sell += s
            cramer_bias = "bullish" if b > s else "bearish" if s > b else "neutral"
            type_counts[_classify_sentiment_type(combined)] += 1
            art_picks = _extract_cramer_picks(combined)
            for pk in art_picks:
                t = pk["ticker"]
                if t not in all_picks_map or (
                    strength_order[pk["stance"]] > strength_order[all_picks_map[t]["stance"]]
                ):
                    all_picks_map[t] = pk
            articles.append({
                "title":       it["title"], "url": it["url"],
                "date":        it["date"],  "cramer_bias": cramer_bias,
                "buy_signals": b, "sell_signals": s,
                "picks": art_picks, "price_mentions": [],
            })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(articles)
    all_picks = sorted(all_picks_map.values(),
                       key=lambda x: strength_order[x["stance"]], reverse=True)

    # Specific stance for THIS ticker
    ticker_stance = all_picks_map.get(ticker.upper(), {}).get("stance", "unknown")

    # Overall cramer signal (from keyword counts across all articles)
    if n == 0:
        cramer_signal  = "unknown"
        inverse_signal = "WAIT"
        inverse_score  = 0.0
        confidence     = "low"
    else:
        if total_buy > total_sell * 1.4:
            cramer_signal = "bullish"
        elif total_sell > total_buy * 1.4:
            cramer_signal = "bearish"
        else:
            cramer_signal = "mixed"

        # Override with direct pick if available
        if ticker_stance == "bullish":
            cramer_signal = "bullish"
        elif ticker_stance == "bearish":
            cramer_signal = "bearish"

        # ── Inverse Cramer flip ───────────────────────────────────────────────
        if cramer_signal == "bullish":
            inverse_signal = "SELL"
            inverse_score  = -0.60
        elif cramer_signal == "bearish":
            inverse_signal = "BUY"
            inverse_score  = +0.60
        else:
            inverse_signal = "WAIT"
            inverse_score  = 0.0

        total_signals   = max(total_buy + total_sell, 1)
        signal_strength = abs(total_buy - total_sell) / total_signals
        if n >= 5 and signal_strength >= 0.5:
            confidence = "high"
        elif n >= 2 and signal_strength >= 0.25:
            confidence = "medium"
        else:
            confidence = "low"

    total_types = sum(type_counts.values()) or 1
    type_pcts   = {k: round(v / total_types * 100) for k, v in type_counts.items()}

    return {
        "available":      n > 0,
        "article_count":  n,
        "cramer_signal":  cramer_signal,
        "inverse_signal": inverse_signal,
        "inverse_score":  round(inverse_score, 4),
        "confidence":     confidence,
        "ticker_stance":  ticker_stance,
        "buy_signals":    total_buy,
        "sell_signals":   total_sell,
        "type_breakdown": type_pcts,
        "picks":          all_picks[:20],          # all tickers from articles
        "articles":       articles[:10],
    }


async def fetch_stocktwits_sentiment(ticker: str) -> dict:
    """Kept for backward-compat; StockTwits is now replaced by Cramer in get_sentiment().
    Returns empty/unavailable skeleton so callers don't crash if directly invoked."""
    return {
        "available": False, "message_count": 0, "sentiment_score": 0.0,
        "call_mentions": 0, "put_mentions": 0,
        "bull_pct": 0, "bear_pct": 0, "type_breakdown": {},
    }


# ── (historical reference kept below for completeness) ──────────────────────
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
        scored: List[float] = []
        type_counts: Counter = Counter()
        st_price_mentions: List[Dict] = []

        for msg in messages:
            body           = msg.get("body", "")
            text_score     = _score_text(body)
            calls, puts    = _detect_option_bias(body)
            sent_type      = _classify_sentiment_type(body)
            prices         = _extract_price_mentions(body)

            call_total    += calls
            put_total     += puts
            type_counts[sent_type] += 1
            st_price_mentions.extend(prices)

            label = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic", "")
            if label == "Bullish":
                bull += 1; effective = max(0.05, text_score)
            elif label == "Bearish":
                bear += 1; effective = min(-0.05, text_score)
            else:
                neutral += 1; effective = text_score

            scored.append(effective)

        total     = len(messages)
        avg_score = sum(scored) / total if total else 0.0
        bull_pct  = round(bull / total * 100) if total else 0
        bear_pct  = round(bear / total * 100) if total else 0

        # Lightweight price aggregation from StockTwits
        st_price_counter: Counter = Counter()
        for pm in st_price_mentions:
            key = (round(pm["price"], 2), pm["type"])
            st_price_counter[key] += 1
        st_top_prices = [
            {"price": k[0], "type": k[1], "count": v}
            for k, v in st_price_counter.most_common(8)
        ]

        type_pcts = {k: round(v / total * 100) for k, v in type_counts.items()} if total else {}

        return {
            "available":       True,
            "message_count":   total,
            "bullish_count":   bull,
            "bearish_count":   bear,
            "neutral_count":   neutral,
            "sentiment_score": round(avg_score, 4),
            "call_mentions":   call_total,
            "put_mentions":    put_total,
            "bull_pct":        bull_pct,
            "bear_pct":        bear_pct,
            "type_breakdown":  type_pcts,
            "price_targets":   st_top_prices,
        }

    except Exception as exc:
        logger.debug("StockTwits %s failed: %s", ticker, exc)
        return {
            "available":       False,
            "message_count":   0,
            "sentiment_score": 0.0,
            "bull_pct":        0,
            "bear_pct":        0,
            "call_mentions":   0,
            "put_mentions":    0,
            "type_breakdown":  {},
            "price_targets":   [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Options flow  (yfinance — blocking → run in executor)
# ─────────────────────────────────────────────────────────────────────────────

def _options_flow_sync(ticker: str) -> dict:
    """
    Fetch the nearest 3 expiry dates' call + put volume / open interest.
    Also identifies the top call and put strikes by volume.

    PCR < 0.7  → call-heavy (bullish crowd positioning)
    0.7–1.0    → neutral
    PCR > 1.0  → put-heavy  (bearish / hedging)
    """
    try:
        t           = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return {"available": False, "reason": "no options data"}

        n_exp          = min(3, len(expirations))
        total_call_vol = total_put_vol = 0
        total_call_oi  = total_put_oi  = 0
        chains_summary = []
        hot_calls: List[Dict] = []
        hot_puts:  List[Dict] = []

        for exp in expirations[:n_exp]:
            try:
                chain = t.option_chain(exp)
                cv = int(chain.calls["volume"].fillna(0).sum())
                pv = int(chain.puts["volume"].fillna(0).sum())
                co = int(chain.calls["openInterest"].fillna(0).sum())
                po = int(chain.puts["openInterest"].fillna(0).sum())

                total_call_vol += cv
                total_put_vol  += pv
                total_call_oi  += co
                total_put_oi   += po

                chains_summary.append({
                    "expiry":  exp,
                    "call_vol": cv,
                    "put_vol":  pv,
                    "call_oi":  co,
                    "put_oi":   po,
                    "pcr_vol":  round(pv / cv, 3) if cv else None,
                })

                # Top 5 call strikes by volume for this expiry
                calls_df = chain.calls.copy()
                calls_df["volume"] = calls_df["volume"].fillna(0)
                calls_df = calls_df[calls_df["volume"] > 0].nlargest(5, "volume")
                for _, row in calls_df.iterrows():
                    lp  = float(row.get("lastPrice", 0) or 0)
                    chg = float(row.get("percentChange", 0) or 0)
                    hot_calls.append({
                        "expiry":      exp,
                        "strike":      float(row["strike"]),
                        "volume":      int(row["volume"]),
                        "oi":          int(row.get("openInterest", 0) or 0),
                        "iv":          round(float(row.get("impliedVolatility", 0) or 0) * 100, 1),
                        "type":        "call",
                        "in_money":    bool(row.get("inTheMoney", False)),
                        "last_price":  round(lp, 2),
                        "pct_change":  round(chg, 2),
                    })

                # Top 5 put strikes by volume
                puts_df = chain.puts.copy()
                puts_df["volume"] = puts_df["volume"].fillna(0)
                puts_df = puts_df[puts_df["volume"] > 0].nlargest(5, "volume")
                for _, row in puts_df.iterrows():
                    lp  = float(row.get("lastPrice", 0) or 0)
                    chg = float(row.get("percentChange", 0) or 0)
                    hot_puts.append({
                        "expiry":      exp,
                        "strike":      float(row["strike"]),
                        "volume":      int(row["volume"]),
                        "oi":          int(row.get("openInterest", 0) or 0),
                        "iv":          round(float(row.get("impliedVolatility", 0) or 0) * 100, 1),
                        "type":        "put",
                        "in_money":    bool(row.get("inTheMoney", False)),
                        "last_price":  round(lp, 2),
                        "pct_change":  round(chg, 2),
                    })

            except Exception:
                pass

        total_vol = total_call_vol + total_put_vol
        pcr_vol   = round(total_put_vol / total_call_vol,  3) if total_call_vol else None
        pcr_oi    = round(total_put_oi  / total_call_oi,   3) if total_call_oi  else None

        flow_bias = (
            "call_heavy" if pcr_vol is not None and pcr_vol < 0.70 else
            "put_heavy"  if pcr_vol is not None and pcr_vol > 1.00 else
            "neutral"    if pcr_vol is not None else "unknown"
        )

        call_pct = round(total_call_vol / total_vol * 100) if total_vol else 0
        put_pct  = round(total_put_vol  / total_vol * 100) if total_vol else 0

        # Keep top-10 hottest strikes across all expiries
        hot_calls.sort(key=lambda x: x["volume"], reverse=True)
        hot_puts.sort( key=lambda x: x["volume"], reverse=True)

        # Get current spot price for ATM/OTM classification downstream
        try:
            info = t.fast_info
            spot_price = float(
                getattr(info, "last_price", None)
                or getattr(info, "regular_market_price", None)
                or 0.0
            )
        except Exception:
            spot_price = 0.0

        return {
            "available":        True,
            "expirations_used": n_exp,
            "spot_price":       round(spot_price, 2),
            "call_volume":      total_call_vol,
            "put_volume":       total_put_vol,
            "call_oi":          total_call_oi,
            "put_oi":           total_put_oi,
            "pcr_volume":       pcr_vol,
            "pcr_oi":           pcr_oi,
            "flow_bias":        flow_bias,
            "call_pct":         call_pct,
            "put_pct":          put_pct,
            "chains":           chains_summary,
            "hot_calls":        hot_calls[:10],
            "hot_puts":         hot_puts[:10],
        }

    except Exception as exc:
        logger.debug("Options flow %s failed: %s", ticker, exc)
        return {"available": False, "reason": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Global market sentiment  (no specific ticker — overall market mood)
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CACHE_TTL = 600.0   # 10 min (market-wide data changes slower)
_global_cache_lock = threading.Lock()
_global_cache: Dict[str, tuple] = {}   # 'GLOBAL' → (result, expire)

# Hot market themes to detect in posts
_MARKET_THEMES = {
    "Fed / Rates":   ["fed", "federal reserve", "fomc", "rate hike", "rate cut", "powell", "interest rate"],
    "Inflation":     ["inflation", "cpi", "pce", "deflation", "disinflation", "price", "consumer price"],
    "Recession":     ["recession", "gdp", "slowdown", "contraction", "stagflation", "layoffs"],
    "Earnings":      ["earnings", "eps", "beat", "miss", "guidance", "revenue", "quarter"],
    "VIX / Fear":    ["vix", "volatility", "fear", "panic", "protection", "hedge"],
    "Market Rally":  ["rally", "breakout", "squeeze", "rip", "moon", "ath", "all-time high"],
    "Market Crash":  ["crash", "correction", "dump", "sell-off", "selloff", "collapse", "drop"],
    "Tech / AI":     ["ai", "artificial intelligence", "nvidia", "semiconductor", "chips", "chatgpt"],
    "Options Flow":  ["calls", "puts", "options", "spreads", "iron condor", "theta", "gamma"],
}

# Ticker-mention regex — looks for $SYMBOL or plain uppercase 2-5 letter words near $ or %
_TICKER_RE = re.compile(r'\$([A-Z]{1,5})\b')

# Junk words that get confused for tickers
_TICKER_BLACKLIST = {
    "I", "A", "AT", "BE", "DO", "GO", "IN", "IS", "IT", "ME", "MY",
    "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "HE", "AN",
    "AI", "DD", "OP", "IV", "OI", "IV", "PE", "EV", "OK", "YO", "IF",
    "PM", "AM", "FY", "QQ", "QE", "MM", "YTD", "YOY", "ATH", "HODL",
    "YOLO", "FOMO", "TLDR", "IIRC", "AFAIK", "IMO", "IMHO",
}


async def fetch_global_market_sentiment(force_refresh: bool = False) -> dict:
    """
    Fetch overall stock-market mood from Reddit hot posts + Google News headlines.
    No ticker filter — covers market-wide talk (SPY/QQQ/Fed/VIX/earnings/etc.)

    Returns
    -------
    {
        mood_score       : float [-1 bearish … +1 bullish],
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
    from email.utils import parsedate_to_datetime

    # Cache check
    if not force_refresh:
        with _global_cache_lock:
            entry = _global_cache.get('GLOBAL')
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

    # ── Step 1: Reddit — broad market subreddits, no ticker filter ────────────
    _GLOBAL_SUBS = [
        ("wallstreetbets", "hot"),
        ("stocks",         "hot"),
        ("investing",      "hot"),
        ("options",        "hot"),
        ("StockMarket",    "hot"),
    ]

    all_posts: List[Dict] = []
    ticker_counter: Counter = Counter()
    theme_counter:  Counter = Counter()
    total_bull = total_bear = 0
    call_total = put_total  = 0

    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
        # Reddit hot posts
        for sub, sort in _GLOBAL_SUBS:
            try:
                url  = f"https://www.reddit.com/r/{sub}/{sort}.json?limit=25&t=day"
                resp = await client.get(url, headers=headers_reddit)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                children = data.get("data", {}).get("children", [])

                for child in children:
                    p     = child.get("data", {})
                    title = p.get("title", "")
                    body  = p.get("selftext", "")
                    text  = f"{title} {body}"

                    score     = _score_text(text)
                    calls, puts = _detect_option_bias(text)
                    sent_type = _classify_sentiment_type(text)

                    call_total += calls
                    put_total  += puts
                    if score > 0.1: total_bull += 1
                    elif score < -0.1: total_bear += 1

                    # Trending tickers ($AAPL, $NVDA etc.)
                    for m in _TICKER_RE.findall(text.upper()):
                        if m not in _TICKER_BLACKLIST and 2 <= len(m) <= 5:
                            ticker_counter[m] += 1

                    # Hot themes
                    tl = text.lower()
                    for theme, kws in _MARKET_THEMES.items():
                        if any(kw in tl for kw in kws):
                            theme_counter[theme] += 1

                    all_posts.append({
                        "title":         title,
                        "url":           p.get("url", ""),
                        "subreddit":     f"r/{sub}",
                        "score":         round(score, 3),
                        "upvotes":       int(p.get("score", 0)),
                        "sentiment_type": sent_type,
                        "call_mentions": calls,
                        "put_mentions":  puts,
                    })

            except Exception as exc:
                logger.debug("Global Reddit r/%s failed: %s", sub, exc)

        # Google News RSS — market-wide headlines
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
                root  = ET.fromstring(resp.text)
                items = root.findall('.//item')
                for item in items[:10]:
                    title = item.findtext('title', '') or ''
                    link  = item.findtext('link',  '') or ''
                    desc  = item.findtext('description', '') or ''
                    text  = f"{title} {desc}"

                    score = _score_text(text)
                    calls, puts = _detect_option_bias(text)
                    call_total += calls
                    put_total  += puts
                    if score > 0.1: total_bull += 1
                    elif score < -0.1: total_bear += 1

                    tl = text.lower()
                    for theme, kws in _MARKET_THEMES.items():
                        if any(kw in tl for kw in kws):
                            theme_counter[theme] += 1

                    for m in _TICKER_RE.findall(text.upper()):
                        if m not in _TICKER_BLACKLIST and 2 <= len(m) <= 5:
                            ticker_counter[m] += 1

                    all_posts.append({
                        "title":         title,
                        "url":           link,
                        "subreddit":     "📰 News",
                        "score":         round(score, 3),
                        "upvotes":       0,
                        "sentiment_type": _classify_sentiment_type(text),
                        "call_mentions": calls,
                        "put_mentions":  puts,
                    })
            except Exception as exc:
                logger.debug("Global RSS %s failed: %s", q, exc)

    # ── Step 1b: X (Twitter) — broad market tweets via twikit ─────────────────
    _GLOBAL_X_QUERIES = [
        "($SPY OR $QQQ OR $SPX) -filter:retweets lang:en",
        "(stock market OR wall street) -filter:retweets lang:en",
    ]
    try:
        import random as _rand
        x_client = await _get_twikit_client()
        if x_client is not None:
            for xq in _GLOBAL_X_QUERIES:
                try:
                    await asyncio.sleep(_rand.uniform(0.5, 1.2))
                    xresults = await x_client.search_tweet(xq, product='Top', count=30)
                    xtweets  = list(xresults) if xresults else []
                    for tw in xtweets:
                        text     = getattr(tw, 'text', '') or ''
                        likes    = int(getattr(tw, 'favorite_count', 0) or 0)
                        retweets = int(getattr(tw, 'retweet_count',  0) or 0)

                        score     = _score_text(text)
                        calls, puts = _detect_option_bias(text)
                        sent_type = _classify_sentiment_type(text)
                        call_total += calls
                        put_total  += puts
                        if score > 0.1:  total_bull += 1
                        elif score < -0.1: total_bear += 1

                        for m in _TICKER_RE.findall(text.upper()):
                            if m not in _TICKER_BLACKLIST and 2 <= len(m) <= 5:
                                ticker_counter[m] += 1

                        tl_lower = text.lower()
                        for theme, kws in _MARKET_THEMES.items():
                            if any(kw in tl_lower for kw in kws):
                                theme_counter[theme] += 1

                        tweet_id    = getattr(tw, 'id', '')
                        user        = getattr(tw, 'user', None)
                        screen_name = getattr(user, 'screen_name', '') if user else ''
                        tweet_url   = (
                            f"https://x.com/{screen_name}/status/{tweet_id}"
                            if screen_name and tweet_id else ""
                        )
                        all_posts.append({
                            "title":         text[:150],
                            "url":           tweet_url,
                            "subreddit":     "𝕏 X",
                            "score":         round(score, 3),
                            "upvotes":       likes + retweets,   # proxy for engagement
                            "sentiment_type": sent_type,
                            "call_mentions": calls,
                            "put_mentions":  puts,
                        })
                    if xtweets:
                        break   # one successful query is enough
                except Exception as exc:
                    logger.debug("Global X query failed: %s", exc)
                    _twikit_reset()
    except Exception as exc:
        logger.debug("Global X block failed: %s", exc)

    # ── Step 2: Cramer market-wide view — fetch articles + extract per-ticker picks
    cramer_market = {"cramer_signal": "unknown", "inverse_signal": "WAIT",
                     "confidence": "low", "article_count": 0,
                     "articles": [], "picks": []}
    try:
        import xml.etree.ElementTree as ET2
        from email.utils import parsedate_to_datetime as p2dt

        cramer_buy = cramer_sell = 0
        cramer_raw_items: List[Dict] = []
        mkt_urls = [
            "https://news.google.com/rss/search?q=%22Jim+Cramer%22+stock&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=%22Mad+Money%22+CNBC+stock&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=Cramer+CNBC+buy+sell&hl=en-US&gl=US&ceid=US:en",
        ]
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as c2:
            # Gather RSS items
            for u in mkt_urls:
                try:
                    r = await c2.get(u, headers=headers_rss)
                    if r.status_code != 200: continue
                    root2 = ET2.fromstring(r.text)
                    for item in root2.findall(".//item")[:20]:
                        title = item.findtext("title",       "") or ""
                        link  = item.findtext("link",        "") or ""
                        pub   = item.findtext("pubDate",     "") or ""
                        desc  = item.findtext("description", "") or ""
                        tl    = f"{title} {desc}".lower()
                        if "cramer" not in tl and "mad money" not in tl:
                            continue
                        try:    pub_str = p2dt(pub).strftime("%Y-%m-%d")
                        except: pub_str = pub[:10] if pub else ""
                        cramer_raw_items.append({"title": title, "url": link,
                                                 "date": pub_str, "desc": desc})
                    if cramer_raw_items: break
                except Exception as exc:
                    logger.debug("Global Cramer RSS: %s", exc)

            # Fetch full article bodies for top 5 items
            g_fetch_tasks = [_fetch_article_text(it["url"], c2)
                             for it in cramer_raw_items[:5]]
            g_bodies = await asyncio.gather(*g_fetch_tasks, return_exceptions=True)

        # Process items — score + extract picks
        global_picks_map: Dict[str, Dict] = {}
        cramer_arts: List[Dict] = []
        g_strength = {"bullish": 3, "bearish": 3, "neutral": 2, "unknown": 1}

        for it, body in zip(cramer_raw_items[:5], g_bodies):
            body_text = body if isinstance(body, str) else ""
            combined  = f"{it['title']} {it['desc']} {body_text}"
            b, s = _score_cramer_text(combined)
            cramer_buy  += b; cramer_sell += s
            bias = "bullish" if b > s else "bearish" if s > b else "neutral"
            art_picks = _extract_cramer_picks(combined)
            for pk in art_picks:
                t = pk["ticker"]
                if t not in global_picks_map or (
                    g_strength[pk["stance"]] > g_strength[global_picks_map[t]["stance"]]
                ):
                    global_picks_map[t] = pk
            cramer_arts.append({"title": it["title"], "url": it["url"],
                                 "date": it["date"], "cramer_bias": bias,
                                 "picks": art_picks})

        for it in cramer_raw_items[5:]:
            combined = f"{it['title']} {it['desc']}"
            b, s = _score_cramer_text(combined)
            cramer_buy += b; cramer_sell += s
            bias = "bullish" if b > s else "bearish" if s > b else "neutral"
            art_picks = _extract_cramer_picks(combined)
            for pk in art_picks:
                t = pk["ticker"]
                if t not in global_picks_map or (
                    g_strength[pk["stance"]] > g_strength[global_picks_map[t]["stance"]]
                ):
                    global_picks_map[t] = pk
            cramer_arts.append({"title": it["title"], "url": it["url"],
                                 "date": it["date"], "cramer_bias": bias,
                                 "picks": art_picks})

        nc = len(cramer_arts)
        if nc > 0:
            if cramer_buy > cramer_sell * 1.4:   sig = "bullish"
            elif cramer_sell > cramer_buy * 1.4: sig = "bearish"
            else:                                 sig = "mixed"
            inv  = "SELL" if sig == "bullish" else "BUY" if sig == "bearish" else "WAIT"
            sstr = abs(cramer_buy - cramer_sell) / max(cramer_buy + cramer_sell, 1)
            conf = ("high"   if nc >= 5 and sstr >= 0.5  else
                    "medium" if nc >= 2 and sstr >= 0.25 else "low")
            all_global_picks = sorted(global_picks_map.values(),
                                      key=lambda x: g_strength[x["stance"]], reverse=True)
            cramer_market = {
                "cramer_signal": sig, "inverse_signal": inv,
                "confidence":    conf, "article_count": nc,
                "articles":      cramer_arts[:5],
                "picks":         all_global_picks[:25],   # full pick list with stances
            }
    except Exception as exc:
        logger.debug("Global Cramer block failed: %s", exc)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_posts    = len(all_posts)
    total_sent = total_bull + total_bear or 1
    bull_pct   = round(total_bull / total_sent * 100)
    bear_pct   = round(total_bear / total_sent * 100)

    raw_score = (total_bull - total_bear) / total_sent if total_sent else 0.0
    # Clamp and label
    mood_score = max(-1.0, min(1.0, raw_score))
    market_mood = "bullish" if mood_score > 0.15 else "bearish" if mood_score < -0.15 else "neutral"

    # Trending tickers — top 15 by mention count
    trending = [
        {"ticker": t, "count": c,
         "sentiment": sum(
             p["score"] for p in all_posts
             if t.lower() in (p.get("title","") or "").lower()
         ) / max(sum(1 for p in all_posts if t.lower() in (p.get("title","") or "").lower()), 1)}
        for t, c in ticker_counter.most_common(15)
    ]

    # Hot themes — percentage of posts mentioning each theme
    total_theme_posts = max(n_posts, 1)
    hot_themes = [
        {"theme": th, "count": cnt, "pct": round(cnt / total_theme_posts * 100)}
        for th, cnt in theme_counter.most_common(8)
        if cnt > 0
    ]

    # Top posts by upvotes
    top_posts = sorted(all_posts, key=lambda x: x["upvotes"], reverse=True)[:15]

    result = {
        "mood_score":       round(mood_score, 4),
        "market_mood":      market_mood,
        "bull_pct":         bull_pct,
        "bear_pct":         bear_pct,
        "post_count":       n_posts,
        "call_mentions":    call_total,
        "put_mentions":     put_total,
        "trending_tickers": trending,
        "hot_themes":       hot_themes,
        "cramer_market":    cramer_market,
        "top_posts":        top_posts,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    with _global_cache_lock:
        _global_cache['GLOBAL'] = (result, time.monotonic() + _GLOBAL_CACHE_TTL)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Options-activity sentiment scorer
# ─────────────────────────────────────────────────────────────────────────────

def _score_options_activity(options_data: dict) -> Tuple[float, str, dict]:
    """
    Derive a directional sentiment score from live options-flow data.

    Signals analysed
    ────────────────
    1. ATM call vs put volume  (strikes within ±1.5% of spot)
       — Someone paying ATM premium is making a near-term directional bet.

    2. OTM call stacking  (calls 0–12% above spot)
       — Heavy OTM call volume indicates speculative bullish interest.

    3. Call-ladder bonus  (≥2 distinct OTM call strikes active)
       — Multiple OTM strikes with real volume = high-conviction bull run bets.

    4. End-of-next-week expiry concentration
       — Short-dated options = urgency / near-term conviction.

    5. Overall PCR (put/call ratio by volume)
       — <0.60 → crowd is call-heavy (bullish positioning).
       — >1.20 → crowd is put-heavy (bearish / hedging).

    Returns
    ───────
    (score, label, details)
      score  ∈ [-1.0, +1.0]  positive = bullish options sentiment
      label  ∈ {strongly_bullish, bullish, neutral, bearish, strongly_bearish}
      details: breakdown dict for UI display
    """
    from datetime import date, timedelta

    if not options_data.get("available"):
        return 0.0, "neutral", {}

    spot = float(options_data.get("spot_price", 0.0))
    if spot <= 0:
        return 0.0, "neutral", {}

    hot_calls = options_data.get("hot_calls", [])
    hot_puts  = options_data.get("hot_puts",  [])
    if not hot_calls and not hot_puts:
        return 0.0, "neutral", {}

    # ── Price zones ───────────────────────────────────────────────────────────
    atm_lo      = spot * 0.985    # ATM band: ±1.5% of spot
    atm_hi      = spot * 1.015
    otm_call_lo = spot * 1.0     # OTM calls: 0–12% above spot
    otm_call_hi = spot * 1.12
    otm_put_lo  = spot * 0.88    # OTM puts:  0–12% below spot
    otm_put_hi  = spot * 1.0

    # ── End-of-next-week expiry window (Thursday–Monday) ─────────────────────
    today        = date.today()
    days_to_fri  = (4 - today.weekday()) % 7 or 7   # always ≥ 1
    next_fri     = today + timedelta(days=days_to_fri)
    nw_start     = (next_fri - timedelta(days=1)).isoformat()   # Thursday
    nw_end       = (next_fri + timedelta(days=3)).isoformat()   # following Monday

    # ── Volume aggregation by zone ────────────────────────────────────────────
    atm_call_vol = sum(c["volume"] for c in hot_calls
                       if atm_lo <= c["strike"] <= atm_hi)
    atm_put_vol  = sum(p["volume"] for p in hot_puts
                       if atm_lo <= p["strike"] <= atm_hi)

    otm_call_vol = sum(c["volume"] for c in hot_calls
                       if otm_call_lo < c["strike"] <= otm_call_hi)
    otm_put_vol  = sum(p["volume"] for p in hot_puts
                       if otm_put_lo  <= p["strike"] < otm_put_hi)

    # Distinct OTM call strikes — "call ladder" breadth indicator
    otm_call_strikes = sorted({
        c["strike"] for c in hot_calls
        if otm_call_lo < c["strike"] <= otm_call_hi and c["volume"] > 0
    })
    n_ladder = len(otm_call_strikes)

    # Next-week volume concentration
    nw_call_vol = sum(c["volume"] for c in hot_calls
                      if nw_start <= (c.get("expiry") or "") <= nw_end)
    nw_put_vol  = sum(p["volume"] for p in hot_puts
                      if nw_start <= (p.get("expiry") or "") <= nw_end)

    pcr = options_data.get("pcr_volume")   # None if unavailable

    # ── Score components ──────────────────────────────────────────────────────
    score = 0.0

    # 1. ATM call/put balance — 35% weight
    atm_total = atm_call_vol + atm_put_vol
    if atm_total > 0:
        score += ((atm_call_vol - atm_put_vol) / atm_total) * 0.35

    # 2. OTM call stacking — 30% weight
    otm_total = otm_call_vol + otm_put_vol
    if otm_total > 0:
        score += ((otm_call_vol - otm_put_vol) / otm_total) * 0.30

    # 3. Call-ladder bonus (multiple OTM strikes = conviction)
    if n_ladder >= 4:
        score += 0.12
    elif n_ladder == 3:
        score += 0.08
    elif n_ladder == 2:
        score += 0.04

    # 4. Next-week expiry urgency — 20% weight
    nw_total = nw_call_vol + nw_put_vol
    if nw_total > 0:
        score += ((nw_call_vol - nw_put_vol) / nw_total) * 0.20

    # 5. Overall PCR — up to ±0.15
    if pcr is not None:
        if   pcr < 0.35:  score += 0.15    # extremely call-heavy
        elif pcr < 0.60:  score += 0.10
        elif pcr < 0.80:  score += 0.05
        elif pcr > 1.80:  score -= 0.15    # extremely put-heavy
        elif pcr > 1.20:  score -= 0.10
        elif pcr > 1.00:  score -= 0.05

    score = round(max(-1.0, min(1.0, score)), 3)

    if   score >  0.35: label = "strongly_bullish"
    elif score >  0.12: label = "bullish"
    elif score < -0.35: label = "strongly_bearish"
    elif score < -0.12: label = "bearish"
    else:               label = "neutral"

    details = {
        "spot":                round(spot, 2),
        "atm_call_vol":        atm_call_vol,
        "atm_put_vol":         atm_put_vol,
        "otm_call_vol":        otm_call_vol,
        "otm_put_vol":         otm_put_vol,
        "otm_call_strikes":    otm_call_strikes,
        "call_ladder_count":   n_ladder,
        "next_week_expiry":    next_fri.isoformat(),
        "next_week_call_vol":  nw_call_vol,
        "next_week_put_vol":   nw_put_vol,
        "pcr":                 pcr,
        "score":               score,
        "label":               label,
    }
    return score, label, details


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────

async def get_sentiment(ticker: str, force_refresh: bool = False) -> dict:
    """
    Fetch and aggregate all sentiment sources for `ticker`.
    Returns a JSON-serialisable dict.

    Sources:
      - X (Twitter) — per-ticker social posts (requires X_USERNAME + X_PASSWORD in .env)
      - Inverse Cramer — Google News RSS, Cramer signal inverted
      - Options flow  — yfinance call/put volume & OI

    Results are cached for _SENTIMENT_TTL seconds (default 5 min) per ticker.
    Pass force_refresh=True to bypass the cache.
    """
    if not force_refresh:
        cached = _sentiment_cache_get(ticker)
        if cached is not None:
            logger.debug("[sentiment] %s — cache hit", ticker)
            return cached

    loop = asyncio.get_running_loop()

    x_task       = asyncio.create_task(fetch_x_sentiment(ticker))
    cramer_task  = asyncio.create_task(fetch_cramer_sentiment(ticker))
    options_coro = loop.run_in_executor(None, _options_flow_sync, ticker)

    x_data, cramer, options = await asyncio.gather(
        x_task, cramer_task, options_coro, return_exceptions=True
    )

    if isinstance(x_data, Exception):
        logger.warning("X sentiment exception: %s", x_data)
        x_data = {
            "available": False, "needs_setup": False,
            "sentiment_score": 0.0, "call_mentions": 0,
            "put_mentions": 0, "post_count": 0,
            "type_breakdown": {}, "top_posts": [],
        }
    if isinstance(cramer, Exception):
        logger.warning("Cramer exception: %s", cramer)
        cramer = {
            "available": False, "article_count": 0,
            "cramer_signal": "unknown", "inverse_signal": "WAIT",
            "inverse_score": 0.0, "confidence": "low",
            "buy_signals": 0, "sell_signals": 0,
            "type_breakdown": {}, "articles": [],
        }
    if isinstance(options, Exception):
        logger.warning("Options exception: %s", options)
        options = {"available": False, "reason": str(options)}

    # ── Weighted aggregate score ──────────────────────────────────────────────
    # Weights:
    #   Options activity : 2.0  — real money = strongest signal
    #   Inverse Cramer   : 1.5  — high-conviction contrarian
    #   X / social posts : 1.0  — social mood
    scores, weights = [], []

    if x_data.get("available") and x_data.get("post_count", 0) > 0:
        scores.append(x_data["sentiment_score"]); weights.append(1.0)

    if cramer.get("available") and cramer.get("inverse_score", 0.0) != 0.0:
        # Cramer score is ALREADY inverted — positive = crowd should BUY.
        scores.append(cramer["inverse_score"]); weights.append(1.5)

    # Options activity: derived from ATM/OTM volume, call ladder, expiry urgency
    opt_score, opt_label, opt_details = _score_options_activity(options or {})
    if opt_score != 0.0:
        scores.append(opt_score); weights.append(2.0)

    agg_score = (
        sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        if scores else 0.0
    )
    agg_score = round(max(-1.0, min(1.0, agg_score)), 4)

    label = ("bullish" if agg_score > 0.15 else "bearish" if agg_score < -0.15 else "neutral")

    text_calls = x_data.get("call_mentions", 0)
    text_puts  = x_data.get("put_mentions",  0)

    # ── Merge type breakdowns ─────────────────────────────────────────────────
    combined_types: Counter = Counter()
    for src in (x_data, cramer):
        for t_type, pct in src.get("type_breakdown", {}).items():
            combined_types[t_type] += pct
    total_type = sum(combined_types.values()) or 1
    merged_type_pcts = {k: round(v / total_type * 100) for k, v in combined_types.items()}

    result = {
        "ticker":    ticker.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "score":          agg_score,
            "label":          label,
            "text_calls":     text_calls,
            "text_puts":      text_puts,
            "type_breakdown": merged_type_pcts,
            # Options contribution to the aggregate
            "options_score":  opt_score,
            "options_label":  opt_label,
        },
        # Full breakdown of options-activity signals (for UI display)
        "options_activity": opt_details,
        "x":          x_data,
        "cramer":     cramer,
        "options_flow": options,
    }
    _sentiment_cache_put(ticker, result)
    return result
