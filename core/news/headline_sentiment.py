"""Headline sentiment scoring from keyword counts."""

from __future__ import annotations

_SENTIMENT_POSITIVE = {
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

_SENTIMENT_NEGATIVE = {
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

_MACRO_KEYWORDS = {
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

_SECTOR_KEYWORDS = {
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
    "macro": _MACRO_KEYWORDS,
}


def _score_sentiment(title: str, summary: str = "") -> str:
    """Return Positive, Negative, or Neutral."""
    text = (title + " " + summary).lower()
    pos = sum(1 for w in _SENTIMENT_POSITIVE if w in text)
    neg = sum(1 for w in _SENTIMENT_NEGATIVE if w in text)
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def _classify_category(title: str, ticker: str = "") -> str:
    """Classify as Company, Macro, Sector, or General."""
    text = title.lower()
    if ticker and ticker.lower() in text:
        return "Company"
    if any(kw in text for kw in _MACRO_KEYWORDS):
        return "Macro"
    for sector, keywords in _SECTOR_KEYWORDS.items():
        if sector == "macro":
            continue
        if any(kw in text for kw in keywords):
            return "Sector"
    return "General"
