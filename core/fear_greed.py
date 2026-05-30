"""CNN Fear & Greed Index with VIX fallback."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx
import yfinance as yf

logger = logging.getLogger(__name__)


def _fg_label(score: float) -> str:
    if score >= 75:
        return "Extreme Greed"
    if score >= 55:
        return "Greed"
    if score >= 45:
        return "Neutral"
    if score >= 25:
        return "Fear"
    return "Extreme Fear"


async def fetch_fear_greed() -> dict:
    """Fetch CNN Fear & Greed Index; fall back to VIX proxy."""
    loop = asyncio.get_running_loop()

    # Try CNN F&G first
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://edition.cnn.com/"},
            )
        if resp.status_code == 200:
            j = resp.json()
            fg = j.get("fear_and_greed", {})
            score = float(fg.get("score", 50))
            return {
                "score": round(score, 1),
                "label": fg.get("rating", _fg_label(score)),
                "previous_close": float(fg.get("previous_close", score)),
                "one_week_ago": float(fg.get("one_week_ago", score)),
                "one_month_ago": float(fg.get("one_month_ago", score)),
                "source": "cnn",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception as exc:
        logger.warning("CNN F&G failed: %s", exc)

    # Fallback: derive from VIX
    try:

        def _vix_score():
            v = yf.Ticker("^VIX")
            price = getattr(v.fast_info, "last_price", None) or 20.0
            # VIX 10 ≈ Extreme Greed (100); VIX 40 ≈ Extreme Fear (0)
            score = max(0, min(100, 100 - (float(price) - 10) * (100 / 30)))
            return round(score, 1)

        score = await loop.run_in_executor(None, _vix_score)
        return {
            "score": score,
            "label": _fg_label(score),
            "source": "vix_proxy",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("VIX fallback failed: %s", exc)
        return {
            "score": 50,
            "label": "Neutral",
            "source": "default",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
