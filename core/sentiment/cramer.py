"""Inverse Cramer signals from Google News RSS + article scraping."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter

import httpx

from .scoring import _classify_sentiment_type, _extract_price_mentions

logger = logging.getLogger(__name__)

# Inverse Cramer (Google News RSS).

_CRAMER_BULLISH_KW: list[str] = [
    # Superlatives / iconic Cramer phrases
    "buy buy buy",
    "screaming buy",
    "strong buy",
    "must own",
    "back up the truck",
    "buying opportunity",
    "buy the dip",
    "great opportunity",
    "once-in-a-lifetime",
    # Standard buy
    "buy",
    "buying",
    "bought",
    "bullish",
    "go long",
    "long",
    # Positive verbs
    "like",
    "likes",
    "love",
    "loves",
    "fan of",
    "big fan",
    "recommend",
    "recommends",
    "recommending",
    "endorses",
    # Upgrades & price-target raises
    "upgrade",
    "upgraded",
    "outperform",
    "overweight",
    "raised price target",
    "price target raised",
    "raised target",
    # Ownership / accumulation
    "own",
    "owns",
    "owning",
    "holding",
    "adding",
    "accumulate",
    "accumulating",
    "staying long",
    "still long",
    # Positive framing
    "still a buy",
    "still like",
    "undervalued",
    "upside",
    "opportunity",
    "positive on",
    "backing",
    "backs",
    "great stock",
    "solid",
    "winner",
]

_CRAMER_BEARISH_KW: list[str] = [
    # Superlatives / iconic Cramer phrases
    "sell sell sell",
    "avoid at all costs",
    "run from",
    "get out now",
    "terrible stock",
    "avoid like the plague",
    # Standard sell
    "sell",
    "selling",
    "sold",
    "bearish",
    "short",
    "shorting",
    # Negative verbs
    "hate",
    "hates",
    "hated",
    "avoid",
    "avoiding",
    "not a fan",
    "don't like",
    "doesn't like",
    "no longer likes",
    "no longer",
    "wrong call",
    # Downgrades & target cuts
    "downgrade",
    "downgraded",
    "underperform",
    "underweight",
    "cut target",
    "price target cut",
    "lowered target",
    "lowered price target",
    # Risk / exit signals
    "risky",
    "too risky",
    "concerned",
    "worried",
    "worrying",
    "stay away",
    "take profit",
    "take profits",
    "dump",
    "exit",
    "reduce exposure",
    "trimming",
    "cut position",
    # Negative framing
    "overvalued",
    "peaked",
    "too expensive",
    "vulnerable",
    "danger",
    "warning",
    "negative on",
]

_CRAMER_NEUTRAL_KW: list[str] = [
    "watch",
    "watching",
    "wait",
    "waiting",
    "monitor",
    "monitoring",
    "keep an eye",
    "uncertain",
    "mixed",
    "neutral",
    "on the fence",
    "not sure",
    "could go either",
    "interesting situation",
    "considering",
]

# Ticker regex for article text.
_ARTICLE_TICKER_RE = re.compile(
    r"(?:"
    r"\((?:NYSE|NASDAQ|Nasdaq|AMEX|NYSEARCA|NYSEMKT|BATS):\s*([A-Z]{1,5})\)"
    r"|\$([A-Z]{1,5})\b"
    r")"
)

# Words that look like tickers but aren't
_ARTICLE_TICKER_BL: set = {
    "A",
    "I",
    "AM",
    "AN",
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
    "AI",
    "DD",
    "OP",
    "IV",
    "OI",
    "PE",
    "EV",
    "OK",
    "PM",
    "FY",
    "QQ",
    "ATH",
    "IPO",
    "ETF",
    "CEO",
    "CFO",
    "COO",
    "SEC",
    "FED",
    "GDP",
    "CPI",
    "PPI",
    "PCE",
    "EPS",
    "FCF",
    "TTM",
    "YOY",
    "IMO",
    "TLDR",
}


def _score_cramer_text(text: str) -> tuple[int, int]:
    """Return (buy_signals, sell_signals)."""
    t = text.lower()
    buys = sum(1 for w in _CRAMER_BULLISH_KW if w in t)
    sells = sum(1 for w in _CRAMER_BEARISH_KW if w in t)
    return buys, sells


def _classify_stance_from_window(window: str) -> tuple[str, str]:
    """
    Classify Cramer's stance on a stock from the text window around its mention.

    Returns

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


def _extract_cramer_picks(full_text: str) -> list[dict]:
    """
    Scan article text for ticker symbols and classify Cramer's stance on each.

    Strategy

    1. Find every ticker via _ARTICLE_TICKER_RE  (exchange notation + $TICKER)
    2. Take ±400 chars around each match as context window
    3. Score the window against bullish/bearish/neutral keyword lists
    4. Deduplicate per ticker - keep the strongest signal seen across all windows

    Returns list of {ticker, stance, evidence} dicts.
    """
    strength = {"bullish": 3, "bearish": 3, "neutral": 2, "unknown": 1}
    picks: dict[str, dict] = {}

    for match in _ARTICLE_TICKER_RE.finditer(full_text):
        ticker = (match.group(1) or match.group(2) or "").upper().strip()
        if not ticker or ticker in _ARTICLE_TICKER_BL or len(ticker) < 2:
            continue

        start = max(0, match.start() - 400)
        end = min(len(full_text), match.end() + 400)
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
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = await client.get(url, headers=headers, timeout=8.0, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        html = resp.text
        # Strip <script> and <style> blocks entirely
        html = re.sub(
            r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE
        )
        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:10_000]  # cap - enough for stance extraction
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

    {
        available        : bool,
        article_count    : int,
        cramer_signal    : 'bullish'|'bearish'|'mixed'|'unknown',
        inverse_signal   : 'BUY'|'SELL'|'WAIT',
        inverse_score    : float [-1 ... +1],
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
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    rss_urls = [
        f"https://news.google.com/rss/search?q=%22Jim+Cramer%22+%22{ticker}%22&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q=Cramer+{ticker}+CNBC&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q=%22Mad+Money%22+{ticker}&hl=en-US&gl=US&ceid=US:en",
    ]

    raw_items: list[dict] = []  # items from RSS before article fetch

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for rss_url in rss_urls:
            try:
                resp = await client.get(rss_url, headers=rss_headers)
                if resp.status_code != 200:
                    continue
                root = ET.fromstring(resp.text)
                for item in root.findall(".//item")[:20]:
                    title = item.findtext("title", "") or ""
                    link = item.findtext("link", "") or ""
                    pub = item.findtext("pubDate", "") or ""
                    desc = item.findtext("description", "") or ""
                    tl = f"{title} {desc}".lower()
                    if "cramer" not in tl and "mad money" not in tl:
                        continue
                    try:
                        pub_str = parsedate_to_datetime(pub).strftime("%Y-%m-%d")
                    except Exception:
                        pub_str = pub[:10] if pub else ""
                    raw_items.append({"title": title, "url": link, "date": pub_str, "desc": desc})
                if raw_items:
                    break
            except Exception as exc:
                logger.debug("Cramer RSS %s: %s", rss_url, exc)

        articles: list[dict] = []
        all_picks_map: dict[str, dict] = {}  # ticker -> best pick across all articles
        total_buy = total_sell = 0
        type_counts: Counter = Counter()
        strength_order = {"bullish": 3, "bearish": 3, "neutral": 2, "unknown": 1}

        fetch_tasks = [_fetch_article_text(it["url"], client) for it in raw_items[:5]]
        bodies = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for it, body in zip(raw_items[:5], bodies, strict=False):
            body_text = body if isinstance(body, str) else ""
            # Combine RSS snippet + full body for maximum coverage
            combined = f"{it['title']} {it['desc']} {body_text}"

            b, s = _score_cramer_text(combined)
            total_buy += b
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

            articles.append(
                {
                    "title": it["title"],
                    "url": it["url"],
                    "date": it["date"],
                    "cramer_bias": cramer_bias,
                    "buy_signals": b,
                    "sell_signals": s,
                    "picks": art_picks,
                    "price_mentions": prices,
                }
            )

        # For remaining RSS items beyond top-5, score title+desc only (no fetch)
        for it in raw_items[5:]:
            combined = f"{it['title']} {it['desc']}"
            b, s = _score_cramer_text(combined)
            total_buy += b
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
            articles.append(
                {
                    "title": it["title"],
                    "url": it["url"],
                    "date": it["date"],
                    "cramer_bias": cramer_bias,
                    "buy_signals": b,
                    "sell_signals": s,
                    "picks": art_picks,
                    "price_mentions": [],
                }
            )

    n = len(articles)
    all_picks = sorted(all_picks_map.values(), key=lambda x: strength_order[x["stance"]], reverse=True)

    # Specific stance for THIS ticker
    ticker_stance = all_picks_map.get(ticker.upper(), {}).get("stance", "unknown")

    # Overall cramer signal (from keyword counts across all articles)
    if n == 0:
        cramer_signal = "unknown"
        inverse_signal = "WAIT"
        inverse_score = 0.0
        confidence = "low"
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

        if cramer_signal == "bullish":
            inverse_signal = "SELL"
            inverse_score = -0.60
        elif cramer_signal == "bearish":
            inverse_signal = "BUY"
            inverse_score = +0.60
        else:
            inverse_signal = "WAIT"
            inverse_score = 0.0

        total_signals = max(total_buy + total_sell, 1)
        signal_strength = abs(total_buy - total_sell) / total_signals
        if n >= 5 and signal_strength >= 0.5:
            confidence = "high"
        elif n >= 2 and signal_strength >= 0.25:
            confidence = "medium"
        else:
            confidence = "low"

    total_types = sum(type_counts.values()) or 1
    type_pcts = {k: round(v / total_types * 100) for k, v in type_counts.items()}

    return {
        "available": n > 0,
        "article_count": n,
        "cramer_signal": cramer_signal,
        "inverse_signal": inverse_signal,
        "inverse_score": round(inverse_score, 4),
        "confidence": confidence,
        "ticker_stance": ticker_stance,
        "buy_signals": total_buy,
        "sell_signals": total_sell,
        "type_breakdown": type_pcts,
        "picks": all_picks[:20],  # all tickers from articles
        "articles": articles[:10],
    }


# Sector -> ticker map for Cramer sector fallback.

_TICKER_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "tech",
    "MSFT": "tech",
    "GOOG": "tech",
    "GOOGL": "tech",
    "META": "tech",
    "AMZN": "tech",
    "TSLA": "tech",
    "NVDA": "tech",
    "AMD": "tech",
    "INTC": "tech",
    "CRM": "tech",
    "ORCL": "tech",
    "ADBE": "tech",
    "QCOM": "tech",
    "AVGO": "tech",
    "TXN": "tech",
    "NFLX": "tech",
    "UBER": "tech",
    "LYFT": "tech",
    "SNAP": "tech",
    "PINS": "tech",
    "TWTR": "tech",
    "SPOT": "tech",
    "ZM": "tech",
    "SHOP": "tech",
    "SQ": "tech",
    "PYPL": "tech",
    "NET": "tech",
    "DDOG": "tech",
    "SNOW": "tech",
    "PLTR": "tech",
    "RBLX": "tech",
    "HOOD": "tech",
    "COIN": "tech",
    "MSTR": "tech",
    # Energy
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    "SLB": "energy",
    "OXY": "energy",
    "MPC": "energy",
    "VLO": "energy",
    "PSX": "energy",
    "EOG": "energy",
    "PXD": "energy",
    "HAL": "energy",
    "BKR": "energy",
    "DVN": "energy",
    "FANG": "energy",
    "APA": "energy",
    # Financials
    "JPM": "financials",
    "BAC": "financials",
    "WFC": "financials",
    "GS": "financials",
    "MS": "financials",
    "C": "financials",
    "BRK.B": "financials",
    "BLK": "financials",
    "SCHW": "financials",
    "AXP": "financials",
    "V": "financials",
    "MA": "financials",
    "COF": "financials",
    "DFS": "financials",
    "SYF": "financials",
    "ALLY": "financials",
    "BX": "financials",
    "KKR": "financials",
    # Healthcare
    "JNJ": "healthcare",
    "PFE": "healthcare",
    "MRK": "healthcare",
    "ABBV": "healthcare",
    "BMY": "healthcare",
    "AMGN": "healthcare",
    "GILD": "healthcare",
    "BIIB": "healthcare",
    "REGN": "healthcare",
    "LLY": "healthcare",
    "UNH": "healthcare",
    "CVS": "healthcare",
    "HUM": "healthcare",
    "CI": "healthcare",
    "ISRG": "healthcare",
    "MRNA": "healthcare",
    "BNTX": "healthcare",
    "NVAX": "healthcare",
}

# Sector -> representative tickers to search Cramer coverage for
_SECTOR_PEER_TICKERS: dict[str, list[str]] = {
    "tech": ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN", "TSLA"],
    "energy": ["XOM", "CVX", "COP", "OXY", "SLB"],
    "financials": ["JPM", "BAC", "GS", "MS", "V", "MA"],
    "healthcare": ["JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY"],
}


async def cramer_for_sector(sector: str) -> dict:
    """
    Fetch recent Jim Cramer mentions for the peer tickers in `sector`.

    Called when the requested ticker has no Cramer coverage in the past 30 days.
    Returns a lightweight dict shaped like fetch_cramer_sentiment() but with
    source_label set to "Sector Cramer Signal".

    The function tries peers in order and stops once it has >= 3 articles.
    """
    peers = _SECTOR_PEER_TICKERS.get(sector, [])
    if not peers:
        return {
            "available": False,
            "article_count": 0,
            "cramer_signal": "unknown",
            "inverse_signal": "WAIT",
            "inverse_score": 0.0,
            "confidence": "low",
            "buy_signals": 0,
            "sell_signals": 0,
            "articles": [],
            "picks": [],
            "type_breakdown": {},
            "source_label": "Sector Cramer Signal",
        }

    collected_articles: list[dict] = []
    total_buy = total_sell = 0

    for peer in peers:
        if len(collected_articles) >= 5:
            break
        try:
            result = await fetch_cramer_sentiment(peer)
            if not result.get("available"):
                continue
            for art in result.get("articles", [])[:3]:
                # Tag each article with the peer ticker it came from
                art = dict(art)
                art["peer_ticker"] = peer
                collected_articles.append(art)
            total_buy += result.get("buy_signals", 0)
            total_sell += result.get("sell_signals", 0)
        except Exception as exc:
            logger.debug("[cramer_for_sector] peer %s error: %s", peer, exc)

    n = len(collected_articles)
    if n == 0:
        cramer_signal = "unknown"
        inverse_signal = "WAIT"
        inverse_score = 0.0
        confidence = "low"
    else:
        if total_buy > total_sell * 1.4:
            cramer_signal = "bullish"
        elif total_sell > total_buy * 1.4:
            cramer_signal = "bearish"
        else:
            cramer_signal = "mixed"

        if cramer_signal == "bullish":
            inverse_signal = "SELL"
            inverse_score = -0.45
        elif cramer_signal == "bearish":
            inverse_signal = "BUY"
            inverse_score = +0.45
        else:
            inverse_signal = "WAIT"
            inverse_score = 0.0

        strength = abs(total_buy - total_sell) / max(total_buy + total_sell, 1)
        confidence = (
            "high" if n >= 5 and strength >= 0.5 else ("medium" if n >= 2 and strength >= 0.25 else "low")
        )

    return {
        "available": n > 0,
        "article_count": n,
        "cramer_signal": cramer_signal,
        "inverse_signal": inverse_signal,
        "inverse_score": round(inverse_score, 4),
        "confidence": confidence,
        "buy_signals": total_buy,
        "sell_signals": total_sell,
        "articles": collected_articles[:5],
        "picks": [],
        "type_breakdown": {},
        "source_label": "Sector Cramer Signal",
    }


def get_ticker_sector(ticker: str) -> str | None:
    """Return the sector for a ticker, or None if unknown."""
    return _TICKER_SECTOR_MAP.get(ticker.upper())
