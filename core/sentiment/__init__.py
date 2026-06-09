"""Social sentiment and options flow aggregation.

Package layout (split from the former 2,355-line core/sentiment.py):
  scoring.py          - text lexicons, option bias, type classification, price extraction
  x.py                - X/Twitter via twikit (dormant without an account)
  social.py           - Reddit + StockTwits fetchers
  cramer.py           - Inverse Cramer (per-ticker and sector fallback)
  options_activity.py - yfinance options flow snapshot + activity scorer
  global_market.py    - market-wide mood (Reddit/News/X/Cramer)
  aggregate.py        - get_sentiment() per-ticker aggregation + cache
"""

from __future__ import annotations

from .aggregate import get_sentiment
from .cramer import cramer_for_sector, fetch_cramer_sentiment, get_ticker_sector
from .global_market import fetch_global_market_sentiment
from .social import fetch_stocktwits_sentiment
from .x import fetch_x_sentiment

__all__ = [
    "cramer_for_sector",
    "fetch_cramer_sentiment",
    "fetch_global_market_sentiment",
    "fetch_stocktwits_sentiment",
    "fetch_x_sentiment",
    "get_sentiment",
    "get_ticker_sector",
]
