"""Persisted signal history: history, metrics, prune, CSV export."""

from __future__ import annotations

import asyncio
import csv
import io
import logging

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from .. import state
from ..deps import limiter, require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["store"])


@router.get("/api/history")
async def api_history(ticker: str | None = None, limit: int = 100):
    """Recent persisted signals (newest first)."""
    if state.store is None:
        return {"items": []}
    limit = max(1, min(limit, 1000))
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, lambda: state.store.recent(ticker=ticker, limit=limit))
    return {"items": rows}


@router.get("/api/metrics")
async def api_metrics(ticker: str | None = None):
    """Aggregate accuracy stats from persisted history."""
    if state.store is None:
        return {"signals": 0}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: state.store.metrics(ticker=ticker))


@router.get("/api/metrics/accuracy")
async def api_accuracy(ticker: str | None = None, limit: int = 200):
    """Directional hit-rate from stored signal history."""
    if state.store is None:
        return {"n_calls": 0, "hit_rate": None, "avg_prob_up_on_buys": None, "avg_prob_up_on_sells": None}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: state.store.accuracy_window(ticker=ticker, limit=max(10, min(limit, 5000)))
    )


@router.post("/api/store/prune")
@limiter.limit("10/minute")
async def api_prune(request: Request, days: int = 30, api_key: str | None = Header(None)):
    """Delete signal records older than `days`."""
    require_api_key(api_key)
    if state.store is None:
        return {"deleted": 0}
    days = max(1, min(days, 3650))
    loop = asyncio.get_running_loop()
    deleted = await loop.run_in_executor(None, lambda: state.store.prune(days=days))
    return {"deleted": deleted, "days": days}


@router.get("/api/export.csv")
async def api_export_csv(ticker: str | None = None, limit: int = 1000):
    """Stream signal history as CSV."""
    if state.store is None:
        rows = []
    else:
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(
            None, lambda: state.store.recent(ticker=ticker, limit=max(1, min(limit, 10000)))
        )

    def _gen():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(
            [
                "ts",
                "ticker",
                "interval",
                "price",
                "label",
                "confidence",
                "drift_bias",
                "prob_up",
                "prob_flat",
                "prob_down",
                "median_price",
                "mc_model",
                "regime",
                "potential_up",
                "potential_down",
                "potential_flat",
            ]
        )
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)
        for r in rows:
            w.writerow(
                [
                    r.get("ts", ""),
                    r.get("ticker", ""),
                    r.get("interval", ""),
                    r.get("price", ""),
                    r.get("label", ""),
                    r.get("confidence", ""),
                    r.get("drift_bias", ""),
                    r.get("prob_up", ""),
                    r.get("prob_flat", ""),
                    r.get("prob_down", ""),
                    r.get("median_price", ""),
                    r.get("mc_model", ""),
                    r.get("regime", ""),
                    r.get("potential_up", ""),
                    r.get("potential_down", ""),
                    r.get("potential_flat", ""),
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    fname = f"mc_trader_{(ticker or 'all').lower()}.csv"
    return StreamingResponse(
        _gen(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
