"""
api/server.py — FastAPI application: routes, WebSocket, poll loop.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import logging
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from datetime import date as _date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional, Set

import httpx
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import VALID_INTERVALS, VALID_MC_MODELS, cfg
from core import analyse, _df_to_candles
from core.backtest import walk_forward
from core.fetcher import fetch_candles
from core.indicators import compute_indicators
from core.regime import detect_regime
from core.scanner import scan_tickers, get_watchlist, WATCHLISTS
from core.signal import compute_signal
from core.store import SignalStore
from core.trade_setup import trade_setup_from_analysis, trade_setup_from_scan
from core.sentiment import fetch_global_market_sentiment, get_sentiment
from core.zone_scanner import zone_scan_tickers
from core.zones import detect_zones
from core.volume_profile import compute_volume_profile
from core.options_flow import fetch_options_flow
from core.hawkes import analyse_hawkes
from core.hmm_regime import analyse_hmm, blend_zone_probability
from core.macro import fetch_macro_indicators   # ← NEW: macroeconomic indicators

from .models import BacktestRequest, ConfigUpdate, ScanRequest


# ── News sentiment & category helpers ─────────────────────────────────────────
# Lightweight keyword-based approach — no NLP library required.
# Sentiment: score each headline by counting positive vs negative finance terms.
# Category: classify by dominant topic keywords.

_SENTIMENT_POSITIVE = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "gain", "gains",
    "growth", "grew", "profit", "profits", "record", "upgrade", "upgraded",
    "bullish", "outperform", "strong", "strength", "boom", "booming",
    "breakthrough", "positive", "rise", "rises", "rose", "higher", "upbeat",
    "recovery", "recover", "momentum", "accelerate", "accelerates", "expansion",
    "exceeds", "exceed", "topped", "tops", "above expectations", "above forecast",
}

_SENTIMENT_NEGATIVE = {
    "miss", "misses", "drop", "drops", "dropped", "fall", "falls", "fell",
    "loss", "losses", "decline", "declines", "declined", "bearish", "downgrade",
    "downgraded", "underperform", "weak", "weakness", "recession", "crash",
    "crashing", "fear", "risk", "risks", "cut", "cuts", "layoff", "layoffs",
    "below expectations", "below forecast", "disappoints", "disappointing",
    "concern", "concerns", "warning", "warns", "slump", "slumps", "plunge",
    "plunges", "correction", "sell-off", "selloff", "contraction", "default",
    "bankruptcy", "debt", "inflation", "stagflation", "tariff", "tariffs",
}

_MACRO_KEYWORDS = {
    "fed", "federal reserve", "fomc", "rate", "rates", "inflation", "cpi",
    "ppi", "pce", "gdp", "unemployment", "jobs", "payroll", "treasury",
    "yield", "yields", "recession", "economy", "economic", "interest rate",
    "bls", "bea", "ism", "pmi", "debt ceiling", "fiscal", "monetary",
    "jerome powell", "powell", "central bank", "quantitative", "tapering",
}

_SECTOR_KEYWORDS = {
    "tech": {"technology", "software", "chip", "semiconductor", "ai", "cloud", "nvidia",
              "apple", "google", "microsoft", "meta", "amazon", "tesla"},
    "energy": {"oil", "gas", "energy", "opec", "crude", "refinery", "exxon", "chevron",
                "lng", "pipeline", "coal", "renewables", "solar", "wind"},
    "financials": {"bank", "banking", "jpmorgan", "goldman", "morgan stanley", "credit",
                   "loan", "lending", "fintech", "insurance", "brokerage"},
    "healthcare": {"fda", "drug", "pharma", "biotech", "clinical", "trial", "approval",
                   "vaccine", "healthcare", "hospital", "medical"},
    "macro": _MACRO_KEYWORDS,
}


def _score_sentiment(title: str, summary: str = "") -> str:
    """
    Keyword-based sentiment classifier.
    Returns "Positive", "Negative", or "Neutral".
    """
    text = (title + " " + summary).lower()
    pos = sum(1 for w in _SENTIMENT_POSITIVE if w in text)
    neg = sum(1 for w in _SENTIMENT_NEGATIVE if w in text)
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def _classify_category(title: str, ticker: str = "") -> str:
    """
    Classify article into: "Company", "Macro", "Sector", or "General".
    Company match is based on whether the ticker appears in the headline.
    """
    text = title.lower()
    # 1. Company — headline mentions the current ticker symbol
    if ticker and ticker.lower() in text:
        return "Company"
    # 2. Macro — macro/economic policy keywords
    if any(kw in text for kw in _MACRO_KEYWORDS):
        return "Macro"
    # 3. Sector — broad sector keywords
    for sector, keywords in _SECTOR_KEYWORDS.items():
        if sector == "macro":
            continue
        if any(kw in text for kw in keywords):
            return "Sector"
    return "General"

logger = logging.getLogger(__name__)

# ─── State ──────────────────────────────────────────────────────────────────
clients:       Set[WebSocket]            = set()
_last_result:  dict                      = {}
_poll_task:    Optional[asyncio.Task]    = None
_store:        Optional[SignalStore]     = None
_analysis_lock: Optional[asyncio.Lock]  = None   # guards _run_analysis — set in lifespan


# ─── Lifespan (replaces deprecated @app.on_event) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _poll_task, _store, _analysis_lock
    _store = SignalStore(cfg.db_path)
    _analysis_lock = asyncio.Lock()          # must be created inside the running loop
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
async def update_config(update: ConfigUpdate, api_key: Optional[str] = Header(None)):
    """Apply config changes; restart poll loop; return fresh analysis."""
    if cfg.api_key and (not api_key or api_key != cfg.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
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

    # Immediate fresh analysis — include full result in response so the
    # frontend can update the chart instantly without waiting for WS broadcast.
    # We run this BEFORE restarting the poll loop so the loop's first pass hits
    # the fetcher's TTL cache instead of launching a duplicate network request.
    result = await _run_analysis()
    if "error" not in result:
        await _broadcast(result)

    # Restart the poll loop only after the analysis completes (cache is warm).
    _poll_task = asyncio.create_task(_poll_loop())

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


@app.get("/api/metrics/accuracy")
async def api_accuracy(ticker: Optional[str] = None, limit: int = 200):
    """
    Directional hit-rate computed from stored signal history.

    For each consecutive pair of recorded signals the next price is used as
    the realised outcome.  Returns hit_rate (%), n_calls evaluated, and
    average prob_up for Buy vs Sell calls.
    """
    if _store is None:
        return {"n_calls": 0, "hit_rate": None,
                "avg_prob_up_on_buys": None, "avg_prob_up_on_sells": None}
    return _store.accuracy_window(ticker=ticker, limit=max(10, min(limit, 5000)))


@app.post("/api/store/prune")
async def api_prune(days: int = 30):
    """
    Delete signal records older than `days` days (default 30).
    Returns the number of rows deleted.
    """
    if _store is None:
        return {"deleted": 0}
    days = max(1, min(days, 3650))
    deleted = _store.prune(days=days)
    return {"deleted": deleted, "days": days}


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


# ─── REST: portfolio price proxy ─────────────────────────────────────────────

@app.get("/api/portfolio/price/historical")
async def portfolio_price_historical(ticker: str, date: str):
    """
    Return the closing price for `ticker` on `date` (YYYY-MM-DD).
    Used by the portfolio tracker to avoid CORS issues with Yahoo Finance.
    """
    try:
        target = _date.fromisoformat(date)
        # Fetch a window around the target date to handle weekends/holidays
        start = (target - timedelta(days=5)).isoformat()
        end   = (target + timedelta(days=2)).isoformat()
        df = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: yf.download(ticker.upper(), start=start, end=end, progress=False, auto_adjust=True)
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/price/live")
async def portfolio_price_live(ticker: str):
    """
    Return the latest market price for `ticker`.
    Used by the portfolio tracker to avoid CORS issues with Yahoo Finance.
    """
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
        raise HTTPException(status_code=500, detail=str(e))


# ─── REST: market structure ──────────────────────────────────────────────────

@app.get("/api/market-structure")
async def api_market_structure(ticker: Optional[str] = None):
    """
    Run all four structural analysis models and return a unified result.

    Models:
      1. Volume Profile  — POC, HVN, LVN, Value Area from OHLCV
      2. Options Flow    — Max Pain, GEX, Call/Put Wall, Gamma Flip (yfinance)
      3. Hawkes Process  — Zone-touch excitation probabilities
      4. HMM Regime      — Probabilistic hidden market state

    Also returns blended zone-reaction probabilities that combine
    HMM regime priors + Hawkes excitation + zone strength.
    
    Timeout: 40 seconds total to prevent indefinite hanging.
    """
    symbol = (ticker or cfg.ticker).upper().strip()
    loop   = asyncio.get_running_loop()

    try:
        # Wrap entire analysis with timeout to prevent hanging
        result = await asyncio.wait_for(
            _api_market_structure_impl(symbol, loop),
            timeout=40.0,  # 40 second timeout
        )
        return result

    except asyncio.TimeoutError:
        logger.error("market-structure timeout after 40s for %s", symbol)
        raise HTTPException(status_code=504, detail="Market structure analysis timed out (>40s)")
    except Exception as exc:
        logger.exception("market-structure failed for %s", symbol)
        raise HTTPException(status_code=500, detail=f"market structure failed: {exc}")


async def _api_market_structure_impl(symbol: str, loop):
    """Market structure analysis with parallelised sub-tasks."""
    try:
        # ── 1. Fetch OHLCV data ───────────────────────────────────────────
        df = await loop.run_in_executor(
            None, fetch_candles, symbol, cfg.interval,
            max(cfg.lookback, cfg.chart_bars), cfg.extended,
        )

        spot        = float(df["close"].iloc[-1]) if not df.empty else None
        log_returns = df["close"].pct_change().dropna().values.tolist()

        # ── 2. Run volume-profile, zones, options-flow (and optionally HMM) in parallel ─
        # HMM is gated by cfg.hmm_enabled (default False) — adds ~5 s.
        # Hawkes is gated by cfg.hawkes_enabled (default False) — adds ~3 s.
        # Both are still available here because market-structure is the dedicated
        # heavy-analysis endpoint (has its own loading spinner in the UI).
        _ms_tasks = [
            loop.run_in_executor(None, compute_volume_profile, df),
            loop.run_in_executor(None, detect_zones, df),
            loop.run_in_executor(None, fetch_options_flow, symbol, spot),
        ]
        if cfg.hmm_enabled:
            _ms_tasks.append(loop.run_in_executor(None, analyse_hmm, log_returns))

        _ms_raw = await asyncio.gather(*_ms_tasks, return_exceptions=True)

        if cfg.hmm_enabled:
            vp_raw, zone_raw, of_raw, hmm_raw = _ms_raw
        else:
            vp_raw, zone_raw, of_raw = _ms_raw
            hmm_raw = None

        # Unpack volume profile
        vp_dict = (
            vp_raw.to_dict()
            if not isinstance(vp_raw, BaseException) and vp_raw
            else {"error": str(vp_raw) if isinstance(vp_raw, BaseException) else "vp_failed"}
        )

        # Unpack zones
        if isinstance(zone_raw, BaseException):
            logger.warning("market-structure zone detect: %s", zone_raw)
            zones_data, zone_list = {}, []
        else:
            zones_data = zone_raw.to_dict()
            zone_list = [
                {"level": z.level, "zone_type": "demand", "strength": z.strength}
                for z in zone_raw.demand_zones
            ] + [
                {"level": z.level, "zone_type": "supply", "strength": z.strength}
                for z in zone_raw.supply_zones
            ]

        # Unpack options flow
        of_dict = (
            of_raw.to_dict()
            if not isinstance(of_raw, BaseException)
            else {"error": str(of_raw)}
        )

        # Unpack HMM
        if hmm_raw is None:
            hmm_result, hmm_dict = None, {"disabled": True}
        elif isinstance(hmm_raw, BaseException):
            logger.warning("market-structure HMM failed: %s", hmm_raw)
            hmm_result, hmm_dict = None, {"error": str(hmm_raw)}
        else:
            hmm_result = hmm_raw
            hmm_dict   = hmm_raw.to_dict()

        # ── 3. Hawkes process (needs zone_list from step 2) ───────────────
        # Gated by cfg.hawkes_enabled — runs ~3 s; disabled by default.
        if cfg.hawkes_enabled and zone_list:
            hawkes_result = await loop.run_in_executor(
                None, analyse_hawkes, log_returns, zone_list,
            )
            hawkes_dict = hawkes_result.to_dict()
        else:
            hawkes_result = None
            hawkes_dict   = {"disabled": True}

        # ── 4. Blended zone-reaction probabilities ────────────────────────
        blended_zones = []
        for z in zone_list:
            hk_probs = None
            if hawkes_result is not None:
                for hr in hawkes_result.zone_reactions:
                    if abs(hr.level - z["level"]) < 0.01:
                        hk_probs = {
                            "bounce":      hr.bounce_prob,
                            "break":       hr.break_prob,
                            "consolidate": hr.consolidate_prob,
                        }
                        break
            if hmm_result is not None:
                blended = blend_zone_probability(
                    hmm=hmm_result,
                    hawkes_probs=hk_probs,
                    zone_strength=z.get("strength", 0.5),
                )
                blended_zones.append({
                    "level":            round(z["level"], 4),
                    "zone_type":        z["zone_type"],
                    "strength":         round(z.get("strength", 0.5), 3),
                    "bounce_prob":      blended["bounce"],
                    "break_prob":       blended["break"],
                    "consolidate_prob": blended["consolidate"],
                })

        return {
            "ticker":          symbol,
            "interval":        cfg.interval,
            "current_price":   round(spot or 0.0, 4),
            "volume_profile":  vp_dict,
            "options_flow":    of_dict,
            "hawkes":          hawkes_dict,
            "hmm":             hmm_dict,
            "blended_zones":   blended_zones,
            "zones":           zones_data,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }

    except asyncio.TimeoutError:
        raise
    except Exception as exc:
        logger.exception("_api_market_structure_impl failed for %s", symbol)
        raise


# ─── REST: sentiment ─────────────────────────────────────────────────────────

@app.get("/api/sentiment")
async def api_sentiment(ticker: Optional[str] = None, force: int = 0):
    """
    Aggregate social sentiment + Inverse Cramer + options flow for `ticker`.
    Pass force=1 to bypass the 5-minute cache and always fetch fresh data.
    """
    symbol = (ticker or cfg.ticker).upper().strip()
    try:
        result = await get_sentiment(symbol, force_refresh=bool(force))
        return result
    except Exception as e:
        logger.exception("Sentiment fetch failed for %s", symbol)
        raise HTTPException(status_code=500, detail=f"sentiment failed: {e}")


@app.get("/api/sentiment/global")
async def api_global_sentiment(force: int = 0):
    """
    Market-wide social sentiment — no specific ticker.
    Covers Reddit hot posts + Google News market headlines + Cramer market view.
    """
    try:
        result = await fetch_global_market_sentiment(force_refresh=bool(force))
        return result
    except Exception as e:
        logger.exception("Global sentiment fetch failed")
        raise HTTPException(status_code=500, detail=f"global sentiment failed: {e}")


# ─── REST: financial news aggregator ────────────────────────────────────────

@app.get("/api/news")
async def api_news(ticker: Optional[str] = None, limit: int = 20):
    """
    Aggregate recent financial news headlines from multiple free sources:
      - Yahoo Finance / yfinance news for the ticker (or general if no ticker)
      - Google News RSS for the ticker or broad market terms
      - VIX level from yfinance (^VIX)
    Returns articles sorted newest-first, deduplicated by title similarity.
    """
    articles = []
    loop     = asyncio.get_running_loop()
    symbol   = (ticker or "").upper().strip()

    # ── 1. Yahoo Finance news via yfinance ──────────────────────────────────
    try:
        def _yf_news():
            t = yf.Ticker(symbol if symbol else "SPY")
            return t.news or []
        yf_items = await loop.run_in_executor(None, _yf_news)
        cutoff   = datetime.now(timezone.utc) - timedelta(days=30)
        for item in yf_items[:15]:
            pub = item.get("providerPublishTime") or item.get("publish_time")
            if pub:
                try:
                    dt = datetime.fromtimestamp(int(pub), tz=timezone.utc)
                except Exception:
                    dt = None
            else:
                dt = None
            if dt and dt < cutoff:
                continue
            articles.append({
                "title":     item.get("title", ""),
                "url":       item.get("link") or item.get("url", ""),
                "source":    item.get("publisher", "Yahoo Finance"),
                "published": dt.isoformat() if dt else "",
                "img":       (item.get("thumbnail") or {}).get("resolutions", [{}])[0].get("url", "") if item.get("thumbnail") else "",
            })
    except Exception as exc:
        logger.warning("yfinance news failed: %s", exc)

    # ── 2. Google News RSS (market + ticker) ────────────────────────────────
    try:
        query = f"{symbol} stock" if symbol else "stock market"
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            ns   = {"dc": "http://purl.org/dc/elements/1.1/"}
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            for item in (root.find("channel") or []):
                if item.tag != "item":
                    continue
                title   = (item.findtext("title") or "").strip()
                url     = (item.findtext("link")  or "").strip()
                pub_str = (item.findtext("pubDate") or "").strip()
                source  = (item.findtext("source") or "Google News").strip()
                try:
                    dt = parsedate_to_datetime(pub_str).astimezone(timezone.utc) if pub_str else None
                except Exception:
                    dt = None
                if dt and dt < cutoff:
                    continue
                if not title or not url:
                    continue
                articles.append({
                    "title":     title,
                    "url":       url,
                    "source":    source,
                    "published": dt.isoformat() if dt else "",
                    "img":       "",
                })
                if len(articles) >= 40:
                    break
    except Exception as exc:
        logger.warning("Google News RSS failed: %s", exc)

    # ── 3. Deduplicate by title prefix (first 60 chars) ─────────────────────
    seen   = set()
    unique = []
    for a in articles:
        key = a["title"][:60].lower()
        h   = hashlib.md5(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(a)

    # Sort newest first (empty dates go last)
    unique.sort(key=lambda x: x["published"] or "0000", reverse=True)

    # ── 3b. Attach sentiment + category to each article ──────────────────────
    # Uses lightweight keyword matching — no external NLP service required.
    for a in unique:
        a["sentiment"] = _score_sentiment(a["title"])
        a["category"]  = _classify_category(a["title"], ticker=symbol)

    # ── 4. Fetch VIX level ───────────────────────────────────────────────────
    vix = None
    try:
        def _vix():
            v = yf.Ticker("^VIX")
            info = v.fast_info
            return getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        vix = await loop.run_in_executor(None, _vix)
        if vix:
            vix = round(float(vix), 2)
    except Exception:
        pass

    return {
        "ticker":   symbol or "MARKET",
        "articles": unique[:limit],
        "vix":      vix,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── REST: Fear & Greed Index ─────────────────────────────────────────────────

@app.get("/api/fear-greed")
async def api_fear_greed():
    """
    Fetch CNN Fear & Greed Index from their public data endpoint.
    Returns current score (0–100), label, and previous values.
    Falls back to VIX-derived estimate if CNN endpoint is unreachable.
    """
    loop = asyncio.get_running_loop()

    # Try CNN F&G first
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://edition.cnn.com/"},
            )
        if resp.status_code == 200:
            j    = resp.json()
            fg   = j.get("fear_and_greed", {})
            score = float(fg.get("score", 50))
            return {
                "score":       round(score, 1),
                "label":       fg.get("rating", _fg_label(score)),
                "previous_close": float(fg.get("previous_close", score)),
                "one_week_ago":   float(fg.get("one_week_ago",  score)),
                "one_month_ago":  float(fg.get("one_month_ago", score)),
                "source":      "cnn",
                "fetched_at":  datetime.now(timezone.utc).isoformat(),
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
            "score":   score,
            "label":   _fg_label(score),
            "source":  "vix_proxy",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("VIX fallback failed: %s", exc)
        return {"score": 50, "label": "Neutral", "source": "default", "fetched_at": datetime.now(timezone.utc).isoformat()}


def _fg_label(score: float) -> str:
    if score >= 75: return "Extreme Greed"
    if score >= 55: return "Greed"
    if score >= 45: return "Neutral"
    if score >= 25: return "Fear"
    return "Extreme Fear"


# ─── REST: macroeconomic indicators ─────────────────────────────────────────

@app.get("/api/macro")
async def api_macro(force: int = 0):
    """
    Return a structured set of macroeconomic indicators that influence stock prices.

    Indicators returned (each with current value, previous value, trend arrow,
    bullish/bearish/neutral impact badge, unit, and a plain-English description):
      - CPI          (Consumer Price Index, YoY %)
      - PPI          (Producer Price Index, YoY %)
      - Core PCE     (Fed's preferred inflation gauge, YoY %)
      - Fed Rate     (Federal Funds Rate / ^IRX proxy)
      - 10Y Yield    (10-Year Treasury Yield / ^TNX)
      - GDP Growth   (Real GDP, QoQ Annualised %)
      - Unemployment (Civilian Unemployment Rate)
      - ISM Mfg PMI  (Manufacturing survey, 50 = neutral)

    Data source priority:
      1. FRED API if FRED_API_KEY env var is set (recommended — free key at fred.stlouisfed.org)
      2. BLS public API for CPI / PPI / Unemployment
      3. yfinance ^TNX / ^IRX for yields and rate proxy
      4. World Bank API for GDP
      5. null for indicators that cannot be fetched without a key

    Results are cached for 4 hours (macro data changes monthly/quarterly).
    Pass force=1 to bypass the cache and force a fresh fetch.
    """
    try:
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, fetch_macro_indicators, bool(force)
        )
        return result
    except Exception as e:
        logger.exception("Macro indicators fetch failed")
        raise HTTPException(status_code=500, detail=f"macro fetch failed: {e}")


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
    logger.debug("WS client connected (%d total)", len(clients))
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
        logger.debug("WS client disconnected (%d total)", len(clients))
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


# ─── Multi-timeframe helpers ─────────────────────────────────────────────────

_HTF_MAP = {
    "1m":  "15m",
    "2m":  "15m",
    "5m":  "1h",
    "15m": "1h",
    "30m": "4h",
    "1h":  "1d",
    "4h":  "1d",
    "1d":  "1d",  # already daily — skip HTF
}


async def _htf_confirmation(ticker: str, base_interval: str,
                            extended: bool, loop) -> dict:
    """
    Fetch one timeframe higher than base_interval and compute a lightweight
    regime + signal snapshot.  Returns a compact dict for the dashboard.
    """
    htf = _HTF_MAP.get(base_interval)
    if not htf or htf == base_interval:
        return {"available": False, "reason": "no higher timeframe"}
    try:
        df  = await loop.run_in_executor(
            None, fetch_candles, ticker, htf, 60, extended
        )
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(
            None, detect_regime, df, ind.adx, ind.obv_slope
        )
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)
        return {
            "available":    True,
            "interval":     htf,
            "regime":       reg.regime,
            "trend_score":  reg.trend_score,
            "potential_up": reg.potential_up,
            "potential_down": reg.potential_down,
            "signal_label": sig.label,
            "composite":    sig.composite,
            "confidence":   sig.confidence,
            "rsi":          ind.rsi,
            "adx":          ind.adx,
            "ema_cross":    ind.ema_cross,
            # Alignment: does HTF agree with the base-TF direction?
            # Caller computes this after receiving HTF + base signal.
        }
    except Exception as e:
        logger.debug("HTF confirmation failed (%s %s): %s", ticker, htf, e)
        return {"available": False, "reason": str(e)}


# ─── Analysis ────────────────────────────────────────────────────────────────

async def _run_analysis() -> dict:
    global _last_result

    # ── Deduplication guard ───────────────────────────────────────────────────
    # If another coroutine is already in the middle of a full analysis, return
    # the cached result immediately rather than launching a redundant fetch.
    # This eliminates the duplicate that arises when update_config runs
    # _run_analysis() and then the newly-restarted poll loop fires within
    # milliseconds — both would otherwise miss the fetcher TTL cache.
    if _analysis_lock is not None and _analysis_lock.locked():
        if _last_result:
            logger.debug("[server] _run_analysis skipped — already in flight, returning cached result")
            return _last_result
        # No cached result yet — wait for the in-flight call to finish then return
        async with _analysis_lock:
            return _last_result

    lock_ctx = _analysis_lock if _analysis_lock is not None else asyncio.Lock()
    async with lock_ctx:
        try:
            loop = asyncio.get_running_loop()

            # Fetch enough bars to cover both display history and MC analysis window.
            display_bars = max(cfg.lookback, cfg.chart_bars)
            df_full = await loop.run_in_executor(
                None, fetch_candles, cfg.ticker, cfg.interval, display_bars, cfg.extended
            )

            # Slice to the analysis window (lookback candles) for MC + indicators.
            df = df_full.tail(cfg.lookback).copy()
            df.attrs = df_full.attrs

            # ── Phase 1: broadcast candles immediately ────────────────────────
            # Data is already fetched (~2-3 s). Push candles to all clients NOW
            # so the chart renders before the slow MC/zones analysis begins.
            # The frontend handles type="partial" by rendering only the chart
            # and showing "Analyzing…" on signal cards.
            try:
                await _broadcast({
                    "type":          "partial",
                    "ticker":        cfg.ticker,
                    "interval":      cfg.interval,
                    "current_price": round(float(df_full["close"].iloc[-1]), 4),
                    "candles":       _df_to_candles(df_full),
                    "updated_at":    datetime.now(timezone.utc).isoformat(),
                })
            except Exception as _pe:
                logger.debug("partial broadcast failed: %s", _pe)

            # Pre-compute returns for HMM now (fast, avoids duplicate work in executor)
            _returns_for_hmm = df["close"].pct_change().dropna().values.tolist()

            # ── Phase 2: run MC + zones concurrently ──────────────────────────
            # analyse() calls compute_volume_profile internally — do NOT add it
            # here again (that was a duplicate costing ~2-3 s every cycle).
            # HMM is gated by cfg.hmm_enabled (default False) — it adds ~5 s.
            _gather_tasks = [
                loop.run_in_executor(None, analyse, df, cfg.n_sim, cfg.n_forward, cfg.mc_model),
                loop.run_in_executor(None, detect_zones, df),
                _htf_confirmation(cfg.ticker, cfg.interval, cfg.extended, loop),
            ]
            if cfg.hmm_enabled:
                # Insert before HTF so the unpack order is: mc, zones, hmm, htf
                _gather_tasks.insert(2, loop.run_in_executor(None, analyse_hmm, _returns_for_hmm))

            raw = await asyncio.gather(*_gather_tasks, return_exceptions=True)

            if cfg.hmm_enabled:
                result_raw, zone_raw, hmm_raw, htf_data = raw
            else:
                result_raw, zone_raw, htf_data = raw
                hmm_raw = None

            # ── Unpack analyse result (must succeed) ──────────────────────────
            if isinstance(result_raw, BaseException):
                raise result_raw
            result = result_raw

            # ── Unpack zone result ────────────────────────────────────────────
            if isinstance(zone_raw, BaseException):
                logger.warning("zone detect failed: %s", zone_raw)
                zones_data = {"demand_zones": [], "supply_zones": [],
                              "nearest_demand": None, "nearest_supply": None,
                              "price_context": "unknown", "atr": 0.0}
            else:
                zones_data = zone_raw.to_dict()

            # ── Volume profile — reuse what analyse() already computed ─────────
            # analyse() embeds the VP result in its return dict under
            # "volume_profile". Pull it out here; no second call needed.
            _vp_inner = result.get("volume_profile") if isinstance(result, dict) else None
            if _vp_inner is None:
                vp_data = None
            elif isinstance(_vp_inner, dict):
                vp_data = _vp_inner
            elif hasattr(_vp_inner, "to_dict"):
                vp_data = _vp_inner.to_dict()
            else:
                vp_data = None

            # ── Unpack HMM result ─────────────────────────────────────────────
            if hmm_raw is None:
                hmm_data = None
            elif isinstance(hmm_raw, BaseException):
                logger.debug("hmm in _run_analysis failed: %s", hmm_raw)
                hmm_data = None
            else:
                hmm_data = hmm_raw.to_dict() if hmm_raw else None

            # ── Unpack HTF confirmation ───────────────────────────────────────
            if isinstance(htf_data, BaseException):
                logger.debug("HTF confirmation failed: %s", htf_data)
                htf_data = {"available": False, "reason": str(htf_data)}

            # ── Override candles with full display history ────────────────────
            if len(df_full) > len(df):
                result["candles"] = _df_to_candles(df_full)

            # ── Trade setup (needs analyse result — runs after gather) ────────
            mc_paths_full = result.pop("_mc_paths_full", None)
            try:
                trade_setup = trade_setup_from_analysis(
                    cfg.ticker, result, interval=cfg.interval, df=df,
                    mc_paths_full=mc_paths_full,
                )
            except Exception as e:
                logger.warning("trade_setup failed: %s", e)
                trade_setup = {"valid": False, "side": "none", "reason": str(e)}

            # ── HTF alignment (needs both analyse + HTF results) ──────────────
            if htf_data.get("available"):
                base_comp = float(result.get("signal", {}).get("composite", 0.0))
                htf_comp  = float(htf_data.get("composite", 0.0))
                if base_comp > 0.05 and htf_comp > 0.05:
                    htf_data["alignment"] = "confirm_bullish"
                elif base_comp < -0.05 and htf_comp < -0.05:
                    htf_data["alignment"] = "confirm_bearish"
                elif base_comp * htf_comp < 0:
                    htf_data["alignment"] = "conflict"
                else:
                    htf_data["alignment"] = "neutral"

            result.update({
                "ticker":         cfg.ticker,
                "interval":       cfg.interval,
                "extended":       cfg.extended,
                "mc_model":       cfg.mc_model,
                "trade_setup":    trade_setup,
                "zones":          zones_data,
                "volume_profile": vp_data,
                "hmm":            hmm_data,
                "htf":            htf_data,
                "config": {
                    "n_sim":        cfg.n_sim,
                    "n_forward":    cfg.n_forward,
                    "lookback":     cfg.lookback,
                    "chart_bars":   cfg.chart_bars,
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
        # Brief yield so the caller (update_config or lifespan) that just ran
        # _run_analysis() can populate the fetcher TTL cache before we fire.
        # The deduplication guard in _run_analysis() handles the race, but this
        # small sleep makes the common case a clean cache-hit instead of a wait.
        await asyncio.sleep(1)
        while True:
            result = await _run_analysis()
            if "error" not in result:
                await _broadcast(result)
            await asyncio.sleep(cfg.poll_seconds)
    except asyncio.CancelledError:
        logger.info("Poll loop cancelled")
        raise
