"""Insider activity from SEC EDGAR Form 4."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)


async def fetch_insider_activity(symbol: str, days: int = 30) -> dict:
    """Return recent Form 4 insider filings for ``symbol``."""
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=max(1, min(days, 365)))).isoformat()
    end = today.isoformat()

    url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22"
        f"&dateRange=custom&startdt={start}&enddt={end}&forms=4"
    )

    filings: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "MCTrader/1.0 (contact@example.com)", "Accept": "application/json"},
            )
        if resp.status_code == 200:
            data = resp.json()
            hits = (data.get("hits") or {}).get("hits") or []
            for hit in hits[:10]:
                src = hit.get("_source") or {}
                period = src.get("period_of_report") or src.get("file_date") or ""
                entity = src.get("entity_name") or src.get("display_names") or symbol
                if isinstance(entity, list):
                    entity = ", ".join(entity)
                # Form 4 XML is large; we surface what EDGAR's index provides
                filings.append(
                    {
                        "date": period,
                        "filer": str(entity)[:60],
                        "form": src.get("form_type", "4"),
                        "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={symbol}&type=4&dateb=&owner=include&count=10",
                    }
                )
    except Exception as exc:
        logger.warning("SEC EDGAR insider fetch failed for %s: %s", symbol, exc)

    # De-duplicate by date+filer
    seen_keys: set[str] = set()
    unique: list[dict] = []
    for f in filings:
        key = f"{f['date']}|{f['filer'][:20]}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(f)

    return {
        "ticker": symbol,
        "days": days,
        "filings": unique[:5],
        "filing_count": len(unique),
        "period_start": start,
        "period_end": end,
        "source": "SEC EDGAR (official)",
        "message": (
            f"No insider activity in {days} days"
            if not unique
            else f"{len(unique)} filing(s) in the last {days} days"
        ),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
