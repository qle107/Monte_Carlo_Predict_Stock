"""
api/server.py — FastAPI application: routes, WebSocket, poll loop.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import VALID_INTERVALS, VALID_MC_MODELS, cfg
from core import analyse
from core.backtest import walk_forward
from core.fetcher import fetch_candles
from core.scanner import scan_tickers, get_watchlist, WATCHLISTS
from core.store import SignalStore
from core.trade_setup import trade_setup_from_analysis, trade_setup_from_scan
from core.zone_scanner import zone_scan_tickers

from .models import BacktestRequest, ConfigUpdate, ScanRequest

logger = logging.getLogger(__name__)

# ─── State ──────────────────────────────────────────────────────────────────
clients:       Set[WebSocket]            = set()
_last_result:  dict                      = {}
_poll_task:    Optional[asyncio.Task]    = None
_store:        Optional[SignalStore]     = None


# ─── Lifespan (replaces deprecated @app.on_event) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _poll_task, _store
    _store = SignalStore(cfg.db_path)
    _poll_task = asyncio.create_task(_poll_loop())
    logger.info("Lifespan startup complete (db=%s)", cfg.db_path)
    try:
        yield
    finally:
        if _poll_task and not _poll_task.done():
            _poll_task.cancel()
            try:
                await _poll_task
            except (asyncio.CancelledError, Exception):
                pass
        if _store:
            _store.close()
        logger.info("Lifespan shutdown complete")


app = FastAPI(title="MC Trader", lifespan=lifespan)

# Static (created lazily — empty by default, fine)
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── HTML ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path(__file__).parent.parent / "templates" / "dashboard.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


# ─── REST: signal/config/health ──────────────────────────────────────────────

@app.get("/api/signal")
async def get_signal():
    """Trigger fresh analysis with current config and return result."""
    result = await _run_analysis()
    if "error" not in result:
        await _broadcast(result)
    return result


@app.get("/api/config")
async def get_config():
    return {
        **cfg.to_dict(),
        "valid_intervals": VALID_INTERVALS,
        "valid_mc_models": VALID_MC_MODELS,
    }


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Apply config changes; restart poll loop; return fresh analysis."""
    global _poll_task

    changed: list[str] = []
    payload = update.model_dump(exclude_none=True)
    for key, value in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            changed.append(f"{key}={value}")

    if not changed:
        return {"status": "noop", "changed": [], "config": await get_config()}

    logger.info("Config updated: %s", ", ".join(changed))

    # Restart poll loop with new settings
    if _poll_task and not _poll_task.done():
        _poll_task.cancel()
        try:
            await _poll_task
        except (asyncio.CancelledError, Exception):
            pass
    _poll_task = asyncio.create_task(_poll_loop())

    # Immediate fresh analysis — include full result in response so the
    # frontend can update the chart instantly without waiting for WS broadcast.
    result = await _run_analysis()
    if "error" not in result:
        await _broadcast(result)

    return {
        "status":  "ok",
        "changed": changed,
        "config":  await get_config(),
        "result":  result if "error" not in result else None,
    }


@app.get("/api/health")
async def health():
    return {
        "status":    "ok",
        "ticker":    cfg.ticker,
        "interval":  cfg.interval,
        "mc_model":  cfg.mc_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── REST: backtest / history / metrics / export ────────────────────────────

@app.post("/api/backtest")
async def api_backtest(req: BacktestRequest):
    """Walk-forward backtest. Reports hit-rate, Brier score, calibration."""
    ticker       = (req.ticker or cfg.ticker).upper().strip()
    interval     = req.interval     or cfg.interval
    n_forward    = req.n_forward    or cfg.n_forward
    n_sim        = req.n_sim        or min(cfg.n_sim, 1000)  # cap for backtest speed
    mc_model     = req.mc_model     or cfg.mc_model
    history_bars = req.history_bars or 200

    try:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None, fetch_candles, ticker, interval, history_bars + 60, cfg.extended
        )
        report = await loop.run_in_executor(
            None,
            walk_forward,
            df, n_forward, n_sim, mc_model, 50,
        )
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=400, detail=f"backtest failed: {e}")

    return {
        "ticker":     ticker,
        "interval":   interval,
        "mc_model":   mc_model,
        "n_forward":  n_forward,
        **report,
    }


@app.get("/api/history")
async def api_history(ticker: Optional[str] = None, limit: int = 100):
    """Recent persisted signals (newest first)."""
    if _store is None:
        return {"items": []}
    limit = max(1, min(limit, 1000))
    rows = _store.recent(ticker=ticker, limit=limit)
    return {"items": rows}


@app.get("/api/metrics")
async def api_metrics(ticker: Optional[str] = None):
    """Aggregate accuracy stats from persisted history."""
    if _store is None:
        return {"signals": 0}
    return _store.metrics(ticker=ticker)


@app.get("/api/export.csv")
async def api_export_csv(ticker: Optional[str] = None, limit: int = 1000):
    """Stream signal history as CSV."""
    if _store is None:
        rows = []
    else:
        rows = _store.recent(ticker=ticker, limit=max(1, min(limit, 10000)))

    def _gen():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([
            "ts", "ticker", "interval", "price", "label", "confidence",
            "drift_bias", "prob_up", "prob_flat", "prob_down",
            "median_price", "mc_model", "regime",
            "potential_up", "potential_down", "potential_flat",
        ])
        yield buf.getvalue(); buf.seek(0); buf.truncate(0)
        for r in rows:
            w.writerow([
                r.get("ts", ""), r.get("ticker", ""), r.get("interval", ""),
                r.get("price", ""), r.get("label", ""), r.get("confidence", ""),
                r.get("drift_bias", ""), r.get("prob_up", ""),
                r.get("prob_flat", ""), r.get("prob_down", ""),
                r.get("median_price", ""), r.get("mc_model", ""),
                r.get("regime", ""),
                r.get("potential_up", ""), r.get("potential_down", ""),
                r.get("potential_flat", ""),
            ])
            yield buf.getvalue(); buf.seek(0); buf.truncate(0)

    fname = f"mc_trader_{(ticker or 'all').lower()}.csv"
    return StreamingResponse(
        _gen(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ─── REST: scanner ───────────────────────────────────────────────────────────

@app.get("/api/scan/watchlists")
async def api_scan_watchlists():
    """Return available watchlist names and their tickers."""
    return {name: tickers for name, tickers in WATCHLISTS.items()}


@app.post("/api/scan")
async def api_scan(req: ScanRequest):
    """
    Scan a list of tickers (or a named watchlist) for breakouts / breakdowns.
    Returns ranked results grouped into breakouts / breakdowns / neutral.
    """
    if req.tickers:
        tickers = [t.upper().strip() for t in req.tickers if t.strip()]
    else:
        tickers = get_watchlist(req.watchlist or "sp500_large")

    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(tickers) > 200:
        raise HTTPException(status_code=400, detail="Max 200 tickers per scan")

    interval   = req.interval or "1d"
    lookback   = req.lookback or 60
    extended   = req.extended if req.extended is not None else cfg.extended
    concurrent = min(req.max_concurrent or 8, 20)

    try:
        report = await scan_tickers(
            tickers        = tickers,
            interval       = interval,
            lookback       = lookback,
            extended       = extended,
            max_concurrent = concurrent,
            min_score_abs  = req.min_score_abs or 0.0,
        )
    except Exception as e:
        logger.exception("Scan failed")
        raise HTTPException(status_code=500, detail=f"scan failed: {e}")

    # ── Attach trade setup to every scan result ───────────────────────────
    def _enrich(items: list) -> list:
        enriched = []
        for r in items:
            try:
                r["trade_setup"] = trade_setup_from_scan(r)
            except Exception as ex:
                r["trade_setup"] = {"valid": False, "side": "none", "reason": str(ex)}
            enriched.append(r)
        return enriched

    report["breakouts"]  = _enrich(report.get("breakouts",  []))
    report["breakdowns"] = _enrich(report.get("breakdowns", []))
    report["neutral"]    = _enrich(report.get("neutral",    []))
    report["all"]        = _enrich(report.get("all",        []))

    return report


@app.post("/api/zone-scan")
async def api_zone_scan(req: ScanRequest):
    """
    Scan for Demand/Supply Zone + EMA 20/50/200 strategy setups.
    Uses zone detection + EMA alignment + MC-estimated trade setup.
    Returns results split by longs / shorts / no_setup.
    """
    if req.tickers:
        tickers = [t.upper().strip() for t in req.tickers if t.strip()]
    else:
        tickers = get_watchlist(req.watchlist or "sp500_large")

    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(tickers) > 200:
        raise HTTPException(status_code=400, detail="Max 200 tickers per scan")

    interval   = req.interval or "1d"
    lookback   = req.lookback or 120    # need more bars for EMA200
    extended   = req.extended if req.extended is not None else cfg.extended
    concurrent = min(req.max_concurrent or 8, 20)
    min_score  = float(req.min_score_abs or 0.0)

    try:
        report = await zone_scan_tickers(
            tickers        = tickers,
            interval       = interval,
            lookback       = lookback,
            extended       = extended,
            max_concurrent = concurrent,
            min_score      = min_score,
        )
    except Exception as e:
        logger.exception("Zone scan failed")
        raise HTTPException(status_code=500, detail=f"zone scan failed: {e}")

    return report


# ─── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    logger.info("WS client connected (%d total)", len(clients))
    if _last_result:
        try:
            await ws.send_json(_last_result)
        except Exception:
            pass
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)
        logger.info("WS client disconnected (%d total)", len(clients))
    except Exception as e:
        clients.discard(ws)
        logger.warning("WS error: %s", e)


async def _broadcast(data: dict):
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ─── Analysis ────────────────────────────────────────────────────────────────

async def _run_analysis() -> dict:
    global _last_result
    try:
        loop = asyncio.get_running_loop()
        df   = await loop.run_in_executor(
            None, fetch_candles, cfg.ticker, cfg.interval, cfg.lookback, cfg.extended
        )
        result = await loop.run_in_executor(
            None, analyse, df, cfg.n_sim, cfg.n_forward, cfg.mc_model
        )
        # ── Trade setup ───────────────────────────────────────────────
        try:
            trade_setup = trade_setup_from_analysis(
                cfg.ticker, result, interval=cfg.interval, df=df
            )
        except Exception as e:
            logger.warning("trade_setup failed: %s", e)
            trade_setup = {"valid": False, "side": "none", "reason": str(e)}

        result.update({
            "ticker":      cfg.ticker,
            "interval":    cfg.interval,
            "extended":    cfg.extended,
            "mc_model":    cfg.mc_model,
            "trade_setup": trade_setup,
            "config": {
                "n_sim":        cfg.n_sim,
                "n_forward":    cfg.n_forward,
                "lookback":     cfg.lookback,
                "poll_seconds": cfg.poll_seconds,
                "extended":     cfg.extended,
                "mc_model":     cfg.mc_model,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        _last_result = result

        # Persist
        if _store is not None:
            try:
                _store.record(result)
            except Exception as e:
                logger.warning("store.record failed: %s", e)

        sig = result["signal"]
        reg = result.get("regime", {}) or {}
        warn_str = f"  ⚠ {result['warnings'][0]}" if result.get("warnings") else ""
        logger.info(
            "%s %s [%s]  price=%.2f  regime=%s  pot up/dn/flat=%.0f/%.0f/%.0f  signal=%s (conf=%.0f%%)%s",
            cfg.ticker, cfg.interval, cfg.mc_model,
            result["current_price"],
            reg.get("regime", "?"),
            reg.get("potential_up", 0), reg.get("potential_down", 0), reg.get("potential_flat", 0),
            sig["label"], sig["confidence"] * 100,
            warn_str,
        )
        return result
    except Exception as e:
        logger.exception("Analysis failed")
        return {"error": str(e)}


# ─── Poll loop ──────────────────────────────────────────────────────────────

async def _poll_loop():
    logger.info("Poll loop started: %s %s every %ds", cfg.ticker, cfg.interval, cfg.poll_seconds)
    try:
        while True:
            result = await _run_analysis()
            if "error" not in result:
                await _broadcast(result)
            await asyncio.sleep(cfg.poll_seconds)
    except asyncio.CancelledError:
        logger.info("Poll loop cancelled")
        raise
