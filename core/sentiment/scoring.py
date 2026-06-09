"""Text-sentiment scoring: lexicons, option bias, type classification, price extraction."""

from __future__ import annotations

import re

# Sentiment scoring

_BULL_WORDS: list[str] = [
    "bullish",
    "moon",
    "mooning",
    "calls",
    "long",
    "buy",
    "buying",
    "breakout",
    "squeeze",
    "yolo",
    "hodl",
    "undervalued",
    "accumulate",
    "strong",
    "support",
    "bounce",
    "rocket",
    "gains",
    "green",
    "ripping",
    "rip",
    "pump",
    "pumping",
    "beat",
    "upgrade",
    "upside",
    "higher",
    "🚀",
    "💎",
    "🙌",
    "📈",
]

_BEAR_WORDS: list[str] = [
    "bearish",
    "puts",
    "short",
    "sell",
    "selling",
    "breakdown",
    "dump",
    "dumping",
    "crash",
    "crashing",
    "overvalued",
    "weak",
    "resistance",
    "fall",
    "falling",
    "drop",
    "dropping",
    "red",
    "miss",
    "downgrade",
    "downside",
    "lower",
    "hedge",
    "fear",
    "risky",
    "bag",
    "bagholder",
    "🐻",
    "💀",
    "📉",
]


def _score_text(text: str) -> float:
    """Return [-1, +1] sentiment score. Positive = bullish."""
    t = text.lower()
    bull = sum(1 for w in _BULL_WORDS if w in t)
    bear = sum(1 for w in _BEAR_WORDS if w in t)
    total = bull + bear
    return 0.0 if total == 0 else (bull - bear) / total


def _detect_option_bias(text: str) -> tuple[int, int]:
    """Return (call_mentions, put_mentions)."""
    t = text.lower()
    calls = len(re.findall(r"\bcalls?\b", t))
    puts = len(re.findall(r"\bputs?\b", t))
    return calls, puts


# Sentiment type classification

_OPTIONS_SIGNALS: list[str] = [
    "call",
    "calls",
    "put",
    "puts",
    "option",
    "options",
    "strike",
    "expir",
    "expiry",
    "theta",
    "delta",
    "gamma",
    "vega",
    "iv",
    "implied volatility",
    "otm",
    "itm",
    "atm",
    "leaps",
    "contract",
    "contracts",
    "premium",
    "0dte",
    "0 dte",
    "dte",
    "spread",
    "covered call",
    "cash secured put",
    "iron condor",
    "straddle",
    "strangle",
    "debit",
    "credit",
    "butterfly",
    "condor",
]

_STOCK_SIGNALS: list[str] = [
    "share",
    "shares",
    "stock",
    "stonk",
    "stonks",
    "holding",
    "held",
    "dividend",
    "earnings",
    "eps",
    "revenue",
    "guidance",
    "buyback",
    "pe ratio",
    "market cap",
    "float",
    "short interest",
    "catalyst",
    "price target",
    "pt ",
    " pt$",
    "analyst",
    "upgrade",
    "downgrade",
]


def _classify_sentiment_type(text: str) -> str:
    """
    Classify the discussion type: 'stock', 'options', 'mixed', or 'general'.
    """
    t = text.lower()
    opt = sum(1 for w in _OPTIONS_SIGNALS if w in t)
    stock = sum(1 for w in _STOCK_SIGNALS if w in t)
    total = opt + stock
    if total == 0:
        return "general"
    ratio = opt / total
    if ratio >= 0.65:
        return "options"
    if ratio <= 0.35:
        return "stock"
    return "mixed"


# Price / strike extraction

# Each entry: (compiled regex, option_type or None, group_index_for_price)
# option_type: 'call' | 'put' | 'stock'
_PRICE_REGEXES: list[tuple[re.Pattern, str, int]] = [
    # "$150c", "$200p", "$150C", "$200P"
    (re.compile(r"\$(\d{1,5}(?:\.\d{1,2})?)\s*([Cc])\b"), "call", 1),
    (re.compile(r"\$(\d{1,5}(?:\.\d{1,2})?)\s*([Pp])\b"), "put", 1),
    # "150c", "200p", "150C", "200P" (bare chain notation >=2 digits)
    (re.compile(r"\b(\d{2,5})([Cc])\b"), "call", 1),
    (re.compile(r"\b(\d{2,5})([Pp])\b"), "put", 1),
    # "$150 calls", "$200 puts", "$150 call"
    (re.compile(r"\$(\d{1,5}(?:\.\d{1,2})?)\s+calls?\b", re.I), "call", 1),
    (re.compile(r"\$(\d{1,5}(?:\.\d{1,2})?)\s+puts?\b", re.I), "put", 1),
    # "150 calls", "200 puts" (no dollar sign)
    (re.compile(r"\b(\d{1,5}(?:\.\d{1,2})?)\s+calls?\b", re.I), "call", 1),
    (re.compile(r"\b(\d{1,5}(?:\.\d{1,2})?)\s+puts?\b", re.I), "put", 1),
    # "strike of $150 / strike 150"
    (re.compile(r"strike\s+(?:of\s+)?\$?(\d{1,5}(?:\.\d{1,2})?)", re.I), "call", 1),
    # "PT $150", "PT: $150", "price target $150"
    (re.compile(r"(?:price\s+target|pt)[:\s]+\$?(\d{1,5}(?:\.\d{1,2})?)", re.I), "stock", 1),
    # "target $150", "targeting $150", "target of $150"
    (re.compile(r"target(?:ing)?\s+(?:of\s+)?\$(\d{1,5}(?:\.\d{1,2})?)", re.I), "stock", 1),
    # "to $150", "at $150", "@ $150", "around $150"
    (re.compile(r"(?:to|at|@|around)\s+\$(\d{1,5}(?:\.\d{1,2})?)", re.I), "stock", 1),
    # "support at 150", "resistance at 150"
    (re.compile(r"(?:support|resistance|level)\s+(?:at\s+)?\$?(\d{1,5}(?:\.\d{1,2})?)", re.I), "stock", 1),
    # Generic "$NNN" fallback - lowest priority
    (re.compile(r"\$(\d{1,5}(?:\.\d{1,2})?)"), "stock", 1),
]

_PRICE_MIN = 0.5  # filter out sub-cent
_PRICE_MAX = 25_000  # filter out obviously wrong numbers


def _extract_price_mentions(text: str) -> list[dict]:
    """
    Extract all price / strike level mentions from `text`.

    Returns a deduplicated list of:
      {price: float, type: 'call'|'put'|'stock', raw: str}
    ordered from most-specific to least-specific.
    """
    seen: set = set()
    results: list[dict] = []

    for pattern, ptype, grp in _PRICE_REGEXES:
        for m in pattern.finditer(text):
            try:
                price = float(m.group(grp).replace(",", ""))
            except (ValueError, IndexError):
                continue
            if not (_PRICE_MIN <= price <= _PRICE_MAX):
                continue
            # round to 2 dp for dedup key
            key = (round(price, 2), ptype)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "price": round(price, 2),
                    "type": ptype,
                    "raw": m.group(0)[:30],
                }
            )

    return results
