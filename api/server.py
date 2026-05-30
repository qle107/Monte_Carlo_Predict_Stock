"""FastAPI routes, WebSocket, and poll loop."""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
from contextlib import asynccontextmanager, suppress
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import VALID_INTERVALS, VALID_MC_MODELS, _lock_cfg, cfg
from core.backtest import walk_forward
from core.fear_greed import fetch_fear_greed
from core.fetcher import fetch_candles
from core.insider import fetch_insider_activity
from core.macro import fetch_macro_indicators
from core.market_structure import analyse_market_structure
from core.news_aggregator import fetch_news
from core.news_stream import news_stream
from core.options_flow import fetch_options_flow, scan_unusual_options, scan_volume_spikes
from core.scanner import WATCHLISTS, get_watchlist, scan_tickers
from core.sentiment import _cramer_for_sector, fetch_global_market_sentiment, get_sentiment, get_ticker_sector
from core.store import SignalStore
from core.trade_setup import trade_setup_from_scan
from core.zone_scanner import zone_scan_tickers

from .analysis import _broadcast, _poll_loop, _run_analysis
from .models import BacktestRequest, ConfigUpdate, ScanRequest
from .state import state

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.store = SignalStore(cfg.db_path)
    state.analysis_lock = asyncio.Lock()  # must be created inside the running loop
    state.poll_task = asyncio.create_task(_poll_loop())
    state.news_stream_task = asyncio.create_task(news_stream.run_loop())
    logger.info("Lifespan startup complete (db=%s)", cfg.db_path)
    try:
        yield
    finally:
        for task in (state.poll_task, state.news_stream_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        if state.store:
            state.store.close()
        logger.info("Lifespan shutdown complete")


app = FastAPI(title="MC Trader", lifespan=lifespan)

_raw_cors = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _raw_cors.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _require_api_key(api_key: str | None) -> None:
    """Raise 401 if API_KEY env is set and the header doesn't match."""
    if cfg.api_key and (not api_key or api_key != cfg.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path(__file__).parent.parent / "templates" / "dashboard.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/flow", response_class=HTMLResponse)
async def flow():
    """Options flow feed (sweeps & blocks, ask-side conviction)."""
    html = STATIC_DIR / "flow.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/signal")
@limiter.limit("30/minute")
async def get_signal(request: Request):
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
@limiter.limit("10/minute")
async def update_config(request: Request, update: ConfigUpdate, api_key: str | None = Header(None)):
    """Apply config changes; restart poll loop; return fresh analysis."""
    _require_api_key(api_key)

    changed: list[str] = []
    payload = update.model_dump(exclude_none=True)
    with _lock_cfg():
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                changed.append(f"{key}={value}")

    if not changed:
        return {"status": "noop", "changed": [], "config": await get_config()}

    logger.info("Config updated: %s", ", ".join(changed))

    # Restart poll loop with new settings
    if state.poll_task and not state.poll_task.done():
        state.poll_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await state.poll_task

    # Run analysis before restarting poll loop so fetcher cache is warm.
    result = await _run_analysis()
    if "error" not in result:
        await _broadcast(result)

    # Restart poll loop after analysis completes.
    state.poll_task = asyncio.create_task(_poll_loop())

    return {
        "status": "ok",
        "changed": changed,
        "config": await get_config(),
        "result": result if "error" not in result else None,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ticker": cfg.ticker,
        "interval": cfg.interval,
        "mc_model": cfg.mc_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/backtest")
@limiter.limit("10/minute")
async def api_backtest(request: Request, req: BacktestRequest, api_key: str | None = Header(None)):
    """Walk-forward backtest."""
    _require_api_key(api_key)
    ticker = (req.ticker or cfg.ticker).upper().strip()
    interval = req.interval or cfg.interval
    n_forward = req.n_forward or cfg.n_forward
    n_sim = req.n_sim or min(cfg.n_sim, 1000)  # cap for backtest speed
    mc_model = req.mc_model or cfg.mc_model
    history_bars = req.history_bars or 200

    try:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None, fetch_candles, ticker, interval, history_bars + 60, cfg.extended
        )
        report = await loop.run_in_executor(
            None,
            walk_forward,
            df,
            n_forward,
            n_sim,
            mc_model,
            50,
        )
    except Exception as e:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=400, detail=f"backtest failed: {e}")  # noqa: B904

    return {
        "ticker": ticker,
        "interval": interval,
        "mc_model": mc_model,
        "n_forward": n_forward,
        **report,
    }


@app.get("/api/history")
async def api_history(ticker: str | None = None, limit: int = 100):
    """Recent persisted signals (newest first)."""
    if state.store is None:
        return {"items": []}
    limit = max(1, min(limit, 1000))
    loop = asyncio.get_running_loop()
    rows = await loop.run_in_executor(None, lambda: state.store.recent(ticker=ticker, limit=limit))
    return {"items": rows}


@app.get("/api/metrics")
async def api_metrics(ticker: str | None = None):
    """Aggregate accuracy stats from persisted history."""
    if state.store is None:
        return {"signals": 0}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: state.store.metrics(ticker=ticker))


@app.get("/api/metrics/accuracy")
async def api_accuracy(ticker: str | None = None, limit: int = 200):
    """Directional hit-rate from stored signal history."""
    if state.store is None:
        return {"n_calls": 0, "hit_rate": None, "avg_prob_up_on_buys": None, "avg_prob_up_on_sells": None}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: state.store.accuracy_window(ticker=ticker, limit=max(10, min(limit, 5000)))
    )


@app.post("/api/store/prune")
@limiter.limit("10/minute")
async def api_prune(request: Request, days: int = 30, api_key: str | None = Header(None)):
    """Delete signal records older than `days`."""
    _require_api_key(api_key)
    if state.store is None:
        return {"deleted": 0}
    days = max(1, min(days, 3650))
    loop = asyncio.get_running_loop()
    deleted = await loop.run_in_executor(None, lambda: state.store.prune(days=days))
    return {"deleted": deleted, "days": days}


@app.get("/api/export.csv")
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


@app.get("/api/portfolio/price/historical")
async def portfolio_price_historical(ticker: str, date: str):
    """Closing price for `ticker` on `date` (YYYY-MM-DD)."""
    try:
        target = _date.fromisoformat(date)
        # Fetch a window around the target date to handle weekends/holidays
        start = (target - timedelta(days=5)).isoformat()
        end = (target + timedelta(days=2)).isoformat()
        df = await asyncio.get_running_loop().run_in_executor(
            None, lambda: yf.download(ticker.upper(), start=start, end=end, progress=False, auto_adjust=True)
        )
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        # Get the closest trading day on or before target
        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
        target_ts = pd.Timestamp(target)
        available = df.index[df.index <= target_ts]
        if available.empty:
            available = df.index  # fallback: use first available
        row = df.loc[available[-1]]
        close = float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"])
        return {"ticker": ticker.upper(), "date": str(available[-1].date()), "close": round(close, 4)}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("portfolio historical price failed for %s on %s: %s", ticker, date, e)
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@app.get("/api/portfolio/price/live")
async def portfolio_price_live(ticker: str):
    """Latest market price for `ticker`."""
    try:
        t = yf.Ticker(ticker.upper())
        info = await asyncio.get_running_loop().run_in_executor(None, lambda: t.fast_info)
        price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        if price is None:
            # fallback: grab last close from 5d history
            df = await asyncio.get_running_loop().run_in_executor(
                None, lambda: t.history(period="5d", auto_adjust=True)
            )
            if not df.empty:
                price = float(df["Close"].iloc[-1])
        if price is None:
            raise HTTPException(status_code=404, detail=f"No live price for {ticker}")
        return {"ticker": ticker.upper(), "price": round(float(price), 4)}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("portfolio live price failed for %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@app.get("/api/market-structure")
@limiter.limit("10/minute")
async def api_market_structure(request: Request, ticker: str | None = None):
    """Volume profile, options flow, Hawkes, and HMM analysis."""
    symbol = (ticker or cfg.ticker).upper().strip()
    loop = asyncio.get_running_loop()

    try:
        # Wrap entire analysis with timeout to prevent hanging
        result = await asyncio.wait_for(
            analyse_market_structure(symbol, loop),
            timeout=40.0,
        )
        return result

    except asyncio.TimeoutError:
        logger.error("market-structure timeout after 40s for %s", symbol)
        raise HTTPException(status_code=504, detail="Market structure analysis timed out (>40s)")  # noqa: B904
    except Exception as exc:
        logger.exception("market-structure failed for %s", symbol)
        raise HTTPException(status_code=500, detail=f"market structure failed: {exc}")  # noqa: B904


@app.get("/api/options/unusual")
@limiter.limit("30/minute")
async def api_unusual_options(
    request: Request,
    tickers: str | None = None,
    watchlist: str | None = None,
    min_volume: int = 100,
    min_oi: int = 50,
    vol_oi_threshold: float = 3.0,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    max_dte: int = 60,
    top_n: int = 50,
    min_premium: float = 0.0,
    new_positions_only: bool = False,
    min_sweep_premium: float = 50_000.0,
    min_block_premium: float = 100_000.0,
    exclude_bid_side: bool = True,
    exclude_high_volume_etfs: bool = True,
):
    """Scan tickers for unusual options activity."""
    # Resolve ticker list
    # Default (no tickers, no watchlist) - full all_optionable universe
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif watchlist:
        ticker_list = get_watchlist(watchlist)
    else:
        ticker_list = get_watchlist("all_optionable")

    # Deduplicate while preserving order
    seen: set[str] = set()
    ticker_list = [t for t in ticker_list if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers resolved")
    # No hard cap - the all_optionable universe is ~500 tickers
    if len(ticker_list) > 600:
        raise HTTPException(status_code=400, detail="Max 600 tickers per scan")

    top_n = max(1, min(top_n, 200))
    # Scale concurrency with universe size.
    # yfinance connection cap
    # regardless of thread count, so raising workers beyond ~8 only adds thread
    # overhead without increasing throughput - keep it modest.
    auto_concurrent = 4 if len(ticker_list) < 20 else (6 if len(ticker_list) < 100 else 8)

    logger.info(
        "Unusual options scan: %d tickers, max_dte=%d, vol_oi=%.1fx, workers=%d",
        len(ticker_list),
        max_dte,
        vol_oi_threshold,
        auto_concurrent,
    )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: scan_unusual_options(
                tickers=ticker_list,
                min_volume=min_volume,
                min_oi=min_oi,
                vol_oi_threshold=vol_oi_threshold,
                iv_spike_z=iv_spike_z,
                otm_pct=otm_pct,
                max_dte=max_dte,
                max_concurrent=auto_concurrent,
                top_n=top_n,
                min_premium=min_premium,
                new_positions_only=new_positions_only,
                min_sweep_premium=min_sweep_premium,
                min_block_premium=min_block_premium,
                exclude_bid_side=exclude_bid_side,
                exclude_high_volume_etfs=exclude_high_volume_etfs,
            ),
        )
        return result
    except Exception as e:
        logger.exception("Unusual options scan failed")
        raise HTTPException(status_code=500, detail=f"unusual options scan failed: {e}")  # noqa: B904


@app.get("/api/options/hot")
@limiter.limit("3/minute")
async def api_options_hot(
    request: Request,
    min_vol_ratio: float = 2.0,
    max_dte: int = 60,
    min_volume: int = 50,
    min_oi: int = 25,
    vol_oi_threshold: float = 2.5,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    top_n: int = 60,
    min_premium: float = 50_000.0,
    max_vol_spikes: int = 40,
    lookback_days: int = 20,
):
    """Hot scanner: volume spikes, then unusual options on those tickers."""
    universe = get_watchlist("all_optionable")

    try:
        loop = asyncio.get_running_loop()

        # Stage 1: volume spike scan
        logger.info(
            "[hot_scan] Stage-1 volume scan: %d tickers, min_ratio=%.1fx",
            len(universe),
            min_vol_ratio,
        )
        vol_spikes: list[dict] = await loop.run_in_executor(
            None,
            lambda: scan_volume_spikes(
                tickers=universe,
                min_vol_ratio=min_vol_ratio,
                top_n=max_vol_spikes,
                lookback_days=lookback_days,
            ),
        )

        # Stage 2: unusual options on hot tickers
        hot_tickers = [s["ticker"] for s in vol_spikes]

        if not hot_tickers:
            logger.info("[hot_scan] No volume spikes found above %.1fx", min_vol_ratio)
            return {
                "hits": [],
                "volume_spikes": [],
                "summary": {
                    "universe_scanned": len(universe),
                    "vol_spike_found": 0,
                    "tickers_scanned": 0,
                    "tickers_with_hits": 0,
                    "total_hits": 0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "mixed_count": 0,
                },
                "scanned_at": datetime.now(timezone.utc).isoformat(),
            }

        logger.info(
            "[hot_scan] Stage-2 options scan: %d high-volume tickers",
            len(hot_tickers),
        )
        # Use a smaller concurrency cap - the tickers already have unusual
        # activity, so yfinance calls are fewer and we want to avoid hammering.
        auto_concurrent = min(6, len(hot_tickers))

        result: dict = await loop.run_in_executor(
            None,
            lambda: scan_unusual_options(
                tickers=hot_tickers,
                min_volume=min_volume,
                min_oi=min_oi,
                vol_oi_threshold=vol_oi_threshold,
                iv_spike_z=iv_spike_z,
                otm_pct=otm_pct,
                max_dte=max_dte,
                max_concurrent=auto_concurrent,
                top_n=top_n,
                min_premium=min_premium,
            ),
        )

        # Enrich each options hit with vol-spike metadata
        vol_spike_map = {s["ticker"]: s for s in vol_spikes}
        for hit in result.get("hits", []):
            tkr = hit.get("ticker", "")
            if tkr in vol_spike_map:
                hit["vol_spike"] = vol_spike_map[tkr]

        # Merge summary fields
        summary = result.get("summary", {})
        summary["universe_scanned"] = len(universe)
        summary["vol_spike_found"] = len(vol_spikes)
        result["summary"] = summary
        result["volume_spikes"] = vol_spikes

        logger.info(
            "[hot_scan] complete - %d vol spikes -> %d options hits",
            len(vol_spikes),
            len(result.get("hits", [])),
        )
        return result

    except Exception as exc:
        logger.exception("[hot_scan] failed")
        raise HTTPException(status_code=500, detail=f"hot options scan failed: {exc}")  # noqa: B904


@app.get("/api/sentiment")
async def api_sentiment(ticker: str | None = None, force: int = 0):
    """Social sentiment, Inverse Cramer, and options flow."""
    symbol = (ticker or cfg.ticker).upper().strip()
    try:
        result = await get_sentiment(symbol, force_refresh=bool(force))
        return result
    except Exception as e:
        logger.exception("Sentiment fetch failed for %s", symbol)
        raise HTTPException(status_code=500, detail=f"sentiment failed: {e}")  # noqa: B904


@app.get("/api/sentiment/global")
async def api_global_sentiment(force: int = 0):
    """Market-wide social sentiment."""
    try:
        result = await fetch_global_market_sentiment(force_refresh=bool(force))
        return result
    except Exception as e:
        logger.exception("Global sentiment fetch failed")
        raise HTTPException(status_code=500, detail=f"global sentiment failed: {e}")  # noqa: B904


@app.get("/api/news")
async def api_news(ticker: str | None = None, limit: int = 20):
    """Recent financial news headlines."""
    symbol = (ticker or "").upper().strip()
    return await fetch_news(symbol, limit)


@app.get("/api/fear-greed")
async def api_fear_greed():
    """CNN Fear & Greed Index."""
    return await fetch_fear_greed()


@app.get("/api/macro")
async def api_macro(force: int = 0):
    """Macroeconomic indicators (CPI, yields, unemployment, etc.)."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, fetch_macro_indicators, bool(force))
        return result
    except Exception as e:
        logger.exception("Macro indicators fetch failed")
        raise HTTPException(status_code=500, detail=f"macro fetch failed: {e}")  # noqa: B904


@app.get("/api/options/gex")
@limiter.limit("20/minute")
async def api_options_gex(request: Request, ticker: str):
    """Gamma-exposure profile, max pain, and call/put walls for one ticker."""
    t = (ticker or "").upper().strip()
    if not t:
        raise HTTPException(status_code=400, detail="ticker required")
    try:
        loop = asyncio.get_running_loop()
        flow = await loop.run_in_executor(None, fetch_options_flow, t)
        return flow.to_dict()
    except Exception as e:
        logger.exception("GEX fetch failed")
        raise HTTPException(status_code=500, detail=f"gex fetch failed: {e}")  # noqa: B904


@app.get("/api/scan/watchlists")
async def api_scan_watchlists():
    """Return available watchlist names and their tickers."""
    return dict(WATCHLISTS.items())


@app.post("/api/scan")
@limiter.limit("5/minute")
async def api_scan(request: Request, req: ScanRequest, api_key: str | None = Header(None)):
    """Scan tickers for breakouts and breakdowns."""
    _require_api_key(api_key)
    if req.tickers:
        tickers = [t.upper().strip() for t in req.tickers if t.strip()]
    else:
        tickers = get_watchlist(req.watchlist or "sp500_large")

    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(tickers) > 200:
        raise HTTPException(status_code=400, detail="Max 200 tickers per scan")

    interval = req.interval or "1d"
    lookback = req.lookback or 60
    extended = req.extended if req.extended is not None else cfg.extended
    concurrent = min(req.max_concurrent or 8, 20)

    try:
        report = await scan_tickers(
            tickers=tickers,
            interval=interval,
            lookback=lookback,
            extended=extended,
            max_concurrent=concurrent,
            min_score_abs=req.min_score_abs or 0.0,
        )
    except Exception as e:
        logger.exception("Scan failed")
        raise HTTPException(status_code=500, detail=f"scan failed: {e}")  # noqa: B904

    # Attach trade setup to every scan result
    def _enrich(items: list) -> list:
        enriched = []
        for r in items:
            try:
                r["trade_setup"] = trade_setup_from_scan(r)
            except Exception as ex:
                r["trade_setup"] = {"valid": False, "side": "none", "reason": str(ex)}
            enriched.append(r)
        return enriched

    report["breakouts"] = _enrich(report.get("breakouts", []))
    report["breakdowns"] = _enrich(report.get("breakdowns", []))
    report["neutral"] = _enrich(report.get("neutral", []))
    report["all"] = _enrich(report.get("all", []))

    return report


@app.post("/api/zone-scan")
@limiter.limit("5/minute")
async def api_zone_scan(request: Request, req: ScanRequest, api_key: str | None = Header(None)):
    """Scan for demand/supply zone setups."""
    _require_api_key(api_key)
    if req.tickers:
        tickers = [t.upper().strip() for t in req.tickers if t.strip()]
    else:
        tickers = get_watchlist(req.watchlist or "sp500_large")

    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")
    if len(tickers) > 200:
        raise HTTPException(status_code=400, detail="Max 200 tickers per scan")

    interval = req.interval or "1d"
    lookback = req.lookback or 120  # need more bars for EMA200
    extended = req.extended if req.extended is not None else cfg.extended
    concurrent = min(req.max_concurrent or 8, 20)
    min_score = float(req.min_score_abs or 0.0)

    try:
        report = await zone_scan_tickers(
            tickers=tickers,
            interval=interval,
            lookback=lookback,
            extended=extended,
            max_concurrent=concurrent,
            min_score=min_score,
        )
    except Exception as e:
        logger.exception("Zone scan failed")
        raise HTTPException(status_code=500, detail=f"zone scan failed: {e}")  # noqa: B904

    return report


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.clients.add(ws)
    logger.debug("WS client connected (%d total)", len(state.clients))
    if state.last_result:
        with suppress(Exception):
            await ws.send_json(state.last_result)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state.clients.discard(ws)
        logger.debug("WS client disconnected (%d total)", len(state.clients))
    except Exception as e:
        state.clients.discard(ws)
        logger.warning("WS error: %s", e)


@app.websocket("/ws/news")
async def ws_news(ws: WebSocket):
    """Live news WebSocket."""
    await ws.accept()
    logger.debug("[ws/news] client connected")
    try:
        while True:
            msg = await ws.receive_json()
            ticker = (msg.get("ticker") or "").upper().strip()
            if not ticker:
                continue
            init_items = await news_stream.subscribe(ws, ticker)
            await ws.send_json(
                {
                    "type": "init",
                    "ticker": ticker,
                    "items": init_items,
                }
            )
    except WebSocketDisconnect:
        logger.debug("[ws/news] client disconnected")
    except Exception as exc:
        logger.warning("[ws/news] error: %s", exc)
    finally:
        await news_stream.unsubscribe(ws)


@app.get("/api/sentiment/sector-cramer")
async def api_sector_cramer(ticker: str | None = None, sector: str | None = None):
    """Sector-level Cramer sentiment fallback."""
    resolved_sector = sector
    if not resolved_sector and ticker:
        resolved_sector = get_ticker_sector(ticker.upper())
    if not resolved_sector:
        return {
            "available": False,
            "article_count": 0,
            "cramer_signal": "unknown",
            "inverse_signal": "WAIT",
            "inverse_score": 0.0,
            "confidence": "low",
            "articles": [],
            "source_label": "Sector Cramer Signal",
            "reason": "unknown sector",
        }
    result = await _cramer_for_sector(resolved_sector)
    result["sector"] = resolved_sector
    return result


@app.get("/api/insider-activity")
async def api_insider_activity(ticker: str | None = None, days: int = 30):
    """Recent insider Form 4 filings."""
    symbol = (ticker or cfg.ticker).upper().strip()
    return await fetch_insider_activity(symbol, days)
