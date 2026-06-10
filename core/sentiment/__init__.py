"""Social sentiment and options flow aggregation."""

from __future__ import annotations

from .aggregate import get_sentiment
from .cramer import cramer_for_sector, get_ticker_sector
from .global_market import fetch_global_market_sentiment
from .social import fetch_stocktwits_sentiment

__all__ = [
    "cramer_for_sector",
    "fetch_global_market_sentiment",
    "fetch_stocktwits_sentiment",
    "get_sentiment",
    "get_ticker_sector",
]
