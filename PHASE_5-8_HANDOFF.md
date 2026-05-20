# Handoff — Phases 5-8

Read this top-to-bottom before touching code. It captures every decision from
the May 2026 refactor session so a fresh chat can pick up exactly where we left
off. Phases 1-4 are done; Phases 5-8 are described below in execute-ready detail.

---

## 0. Project shape (so a new chat doesn't have to re-explore)

```
Monte_Carlo_Predict_Stock/
├── main.py                        uvicorn entry (python main.py → :8000)
├── config.py                      Config dataclass + .env parser
├── pyproject.toml                 ruff + pytest config (added Phase 2)
├── CHANGELOG.md                   Phase 2/3/4 entries — read first
├── api/
│   ├── server.py                  FastAPI routes + WS + poll loop
│   └── models.py                  Pydantic request bodies
├── core/                          analysis pipeline (19 Python modules)
│   ├── sentiment.py               Reddit + Inverse Cramer + options_flow + UA
│   ├── hmm_regime.py              pure-NumPy Gaussian HMM
│   ├── hawkes.py                  Hawkes self-excitation
│   ├── zones.py                   demand/supply zone detector
│   ├── trade_setup.py             entry/SL/TP/RR + zone scenarios
│   ├── backtest.py                walk-forward
│   ├── macro.py                   FRED + BLS + yfinance macro indicators
│   └── … (see README.md for full list)
├── templates/dashboard.html       ~7800 lines, the entire frontend
├── static/
│   ├── css/dashboard.css          extracted Phase 2
│   ├── css/portfolio.css          extracted Phase 2
│   └── js/
│       ├── index.js               module barrel + status flag
│       ├── scanner.js             extracted Phase 2
│       └── tabs/
│           ├── options.js         extracted Phase 3 (Unusual Activity)
│           └── market-structure.js extracted Phase 4 (error boundaries)
└── tests/                         pytest suite
```

**Stack**: Python 3.10+ / FastAPI / uvicorn / yfinance / pandas / numpy / scipy /
SQLite. The frontend is vanilla HTML + inline JS + (now) some external JS
modules — **no React, no build step, no npm**.

---

## 1. Decisions already captured (don't re-ask)

These came from a `AskUserQuestion` block at the start of the session:

| Phase | Decision |
|---|---|
| 2 (refactor) | **Stay vanilla, refactor in spirit.** Pull CSS/JS into `/static/`, keep classic `<script>` tags so existing `onclick=…` handlers work, defer per-tab JS extraction into the feature phases. |
| 3-B (options flow direction) | **Show disclaimer only.** yfinance doesn't expose aggressor side; the spec's "Buy Calls / Sell Calls / Buy Puts / Sell Puts" metrics aren't computable. Frontend renders a visible amber disclaimer. |
| 5 (live news feed) | **Server-poll + WS push.** Backend polls RSS every 15-30s, diffs against prior cycle, broadcasts new items via a new WS channel. From the frontend POV it's true push (no client polling). |
| 8 (linting) | **ruff for Python only.** ESLint requires a Node toolchain this project doesn't have. Verification leans on `ruff check`, `pytest -q`, and browser manual test. |

---

## 2. State after Phases 1-4 (what's already done)

**Phase 1** — Exploration complete. Findings reported.

**Phase 2** — Refactor:
- CSS extracted to `static/css/dashboard.css` + `static/css/portfolio.css`.
  Inline `<style>` blocks in `dashboard.html` are bracketed by HTML comments
  (`<!-- legacy-css-1 … end-legacy-css-1 -->` etc.) for diff auditability.
- Scanner JS extracted to `static/js/scanner.js`. Original inline block disabled
  via `type="text/x-disabled-legacy"`.
- `pyproject.toml` adds ruff (ruleset: E/F/W/I/B/C4/UP/SIM/RUF, with sensible
  per-file ignores) and pytest config.
- `static/js/index.js` is a manifest barrel exposing `window.__mc_trader_modular__`
  with `version`, `extracted`, `inline` arrays — useful for spot-checking in
  DevTools console.

**Phase 3** — Options tab:
- Backend `core/sentiment.py`: new helper `_scan_unusual_activity(calls_df,
  puts_df, exp_str, today, min_vol=500, vol_oi_mult=1.5)`. `_options_flow_sync`
  scans up to 8 expirations (covers near-term + LEAPS), returns
  `options_flow.unusual_activity[]` sorted by Vol/OI desc, capped at 50.
  Each row has `{expiry, dte, strike, type, volume, oi, vol_oi, premium, flow,
  pct_change, iv, in_money, leaps}`. Response also carries
  `flow_direction_available: false`.
- Frontend HTML: amber "Flow direction unavailable from this source" banner at
  the top of `#opts-results`. `title=` tooltips on Calls Vol / Puts Vol / P/C
  ratio KPIs. New 🚨 Unusual Activity section with DTE filter pills
  (All / 0-7 / 8-30 / 31-60 / 60+ · LEAPS) and a sortable 11-column table.
- New module `static/js/tabs/options.js` (~155 lines) owns the Unusual Activity
  render + DTE filter + column sort. Exposes `window.renderUnusualActivity(data)`
  (called from the inline `_renderSentimentPanel`) and `window.setUnusualFilter`.

**Phase 4** — Market Structure:
- Backend `_api_market_structure_impl` rewritten. HMM + Hawkes now ALWAYS run
  in this endpoint (the cfg flags only gate the live poll loop). New state
  contract on each sub-result:
  `state ∈ {"ok", "error", "insufficient_data", "no_zones"}` plus
  `error_reason`, `min_bars_required`, `bars_available`. Blended zones gain a
  `blend_source` field. Pipeline-step `logger.debug('[ms] …')` at every step.
- New module `static/js/tabs/market-structure.js` (~265 lines). Monkey-patches
  `_renderHMM`, `_renderHawkes`, `_renderBlendedZones`, `_renderMarketStructure`
  after DOMContentLoaded. Each section in its own try/catch error boundary.
  State-aware empty/error renderers with **↻ Retry** buttons.

CHANGELOG.md has the full writeup of every change.

---

## 3. Phase 5 — Social Sentiment Tab Overhaul

### Goal

The Social Sentiment tab currently duplicates the news feed already in News &
Macro. Replace the news block with a **live streaming news feed** (server-poll
+ WS push, frontend treats as push), keep the social signals + Inverse Cramer,
improve Inverse Cramer to show sector-peer Cramer mentions when the current
ticker has no recent coverage.

### What to keep (top half)

The "Social signals" + Inverse Cramer rows already in the tab:
- Reddit sentiment score + WSB/r/stocks/r/investing breakdown
- X/Twitter: post count, call mentions, put mentions, text bias
- Overall sentiment badge

### What to replace (bottom half)

Current state: the bottom of `#sentiment-scan-panel` (lines ~1715-1789 of
`templates/dashboard.html`) has a "📰 News · VIX · Fear & Greed" sub-section
followed by "🌍 Global Market Pulse". The first sub-section duplicates News &
Macro and should be **replaced** with a live streaming feed.

The Global Market Pulse stays put (it's broader-market mood, not per-ticker news).

### How — Backend

1. **New module** `core/news_stream.py`:
   - Runs a background asyncio task started from `api/server.py`'s lifespan.
   - Polls Yahoo Finance via yfinance + Google News RSS every 20s.
   - Maintains a per-ticker dict of recent items (max 50 per ticker, oldest
     dropped). Use `hashlib.md5(title[:60].lower())` as the dedup key (same as
     existing `/api/news`).
   - When a new item arrives, broadcasts to a new WS channel.
   - Each item shape: `{ticker, title, url, source, published_iso, sentiment, category}`.
   - Sentiment + category use the existing helpers in `api/server.py`
     (`_score_sentiment`, `_classify_category`) — extract them into
     `core/news_stream.py` or import from there.

2. **New WS endpoint** in `api/server.py`:
   ```python
   @app.websocket("/ws/news")
   async def ws_news(ws: WebSocket):
       await ws.accept()
       news_clients.add(ws)
       # Send initial buffer for current ticker
       try:
           await ws.send_json({"type": "init", "items": news_stream.recent(ticker)})
           while True:
               msg = await ws.receive_json()  # client can send {"ticker": "AAPL"} to switch
               news_stream.subscribe(ws, msg.get("ticker", "").upper())
       except WebSocketDisconnect:
           news_clients.discard(ws)
   ```

3. **Subscription model**: each WS client subscribes to ONE ticker (the
   currently-active one in the dashboard). When the ticker changes in
   `currentConfig`, the frontend sends `{"ticker": "NEW"}` over the existing
   WS, the backend updates the client's subscription, sends an init burst,
   and from then on only broadcasts items for that ticker.

4. **Lifespan integration**: in `api/server.py:lifespan`, kick off
   `asyncio.create_task(news_stream.run_loop())` alongside `_poll_task`.
   Cancel it cleanly on shutdown.

5. **Resource notes**: scraping per-ticker every 20s is OK for a handful of
   active connections. If many users connect simultaneously, dedupe poll work
   by maintaining a single per-ticker poll loop and fan-out to subscribers.

### How — Frontend

1. **Extract a new module** `static/js/tabs/sentiment.js`. It owns:
   - The live news feed renderer
   - The WS connection + reconnect logic
   - The buffer (max 50 items, oldest dropped)
   - Pause button state (when paused, items still arrive over WS but don't
     replace the visible buffer — they go to a "pending" array)
   - Filter pills (All / Company / Macro / Sector / Positive / Negative)
   - Live "X seconds ago" timestamps (updated by `setInterval(1000)`)
   - Sector Cramer fallback for the Inverse Cramer card when ticker has no
     coverage

2. **HTML changes** in `templates/dashboard.html` (`#sentiment-scan-panel`):
   - Replace the 📰 News · VIX · Fear & Greed sub-section (lines ~1715-1789)
     with a new "🔴 Live News Feed" sub-section containing:
     - Status row (live dot + "Connected" / "Disconnected" / Reconnecting)
     - Pause/Resume button (`⏸ Pause` / `▶ Resume`)
     - Filter pills (re-use the existing `.news-filter-btn` styles)
     - Scrollable feed div `<div id="live-news-list">` with max-height ~520px
   - The Inverse Cramer card already exists (lines ~1628-1673). Add a
     `_renderSectorCramer(data)` path: when `cramer.articles.length === 0`,
     fetch `/api/sentiment/sector-cramer?sector=<inferred>` (new endpoint —
     see backend below) and render those instead, labelled "Sector Cramer
     Signal".

3. **Wire the new module's WS into the page**:
   - On Sentiment tab open, open `new WebSocket('/ws/news')`.
   - Send `{"ticker": currentConfig.ticker}` on open and whenever ticker
     changes.
   - On message: parse, dedup by md5 of title prefix, prepend to buffer,
     trim to 50.
   - On disconnect: exponential backoff reconnect (1s → 2s → 4s → 8s, cap 30s).
   - On tab close / page unload: `ws.close()`.

4. **Inverse Cramer improvement (sector fallback)**:
   - Add a small Python lookup table mapping ticker → sector (e.g.
     `_TICKER_SECTOR_MAP` in `core/sentiment.py`). yfinance's `Ticker(t).info`
     has a `sector` field but it's slow and unreliable.
   - When `_cramer_for_ticker` returns no articles in past 30 days, fall back
     to `_cramer_for_sector` which returns the last 5 Cramer mentions for any
     ticker in the same sector. Label the UI section "Sector Cramer Signal".
   - Tooltip helper: "Inverse Cramer flips Jim Cramer's bullish/bearish call.
     Empirically, his short-term picks have underperformed; the contrarian
     trade has outperformed."

### Files to touch (Phase 5)

| File | Action |
|---|---|
| `core/news_stream.py` | NEW — background polling + dedup + WS broadcast |
| `core/sentiment.py` | Add `_cramer_for_sector` fallback + ticker→sector map |
| `api/server.py` | New `/ws/news` endpoint; lifespan kicks off news_stream task; ensure cleanup |
| `templates/dashboard.html` | Replace News/VIX/F&G sub-section with Live News Feed UI; load `sentiment.js` |
| `static/js/tabs/sentiment.js` | NEW — feed renderer, WS client, pause, filters, sector Cramer |
| `static/js/index.js` | Update extracted manifest |

### Verification

1. Start the server, open dashboard, switch to Social Sentiment tab.
2. DevTools → Network → WS — should see `/ws/news` connection alive.
3. After ~20s a new headline should appear at the top of the feed without
   any manual refresh.
4. Click Pause → new items pile up but don't shift the visible list. Click
   Resume → they appear.
5. Filter pills filter the buffer client-side.
6. Switch ticker in header → feed clears and re-populates for the new ticker.
7. Find a ticker with no Cramer mentions (e.g. an illiquid small-cap) and
   confirm the sector-Cramer fallback fires.
8. Refresh the page → reconnect logic re-establishes WS.

---

## 4. Phase 6 — News & Macro Tab Cleanup

### Goal

Remove the duplicate news list (now lives in Social Sentiment as the live
feed). Keep only macro indicators, VIX + Fear & Greed, add an Earnings Calendar
and a Fed Calendar.

### What to remove

In `templates/dashboard.html`, `#news-panel`:
- The entire News Feed section (lines ~2118-2167 — `#news-feed-list`, the
  filter pill row, the `#news-fetched-at` timestamp).
- The associated JS: `fetchNewsData()`, `setNewsFilter()`, `renderNewsFeed()`
  in the inline block. Keep them around if they're still called elsewhere
  (they are — Phase 5's live feed replaces this path entirely once it ships,
  so safe to remove after Phase 5).

### What to keep

- **Macroeconomic Indicators grid** (`.macro-grid` with 8 cards: CPI, PPI,
  Core PCE, Fed Rate, 10Y Yield, GDP, Unemployment, ISM PMI) — already
  populated from `/api/macro` (uses `core/macro.py`).
- **VIX badge** (`#news-vix-badge`) — leave as-is.

### What to add

#### A) Earnings Calendar widget

Shows next 7 days of earnings for the current ticker + its top 5 sector peers.

**Backend** — new endpoint `GET /api/earnings-calendar?ticker=<t>`:
- Use yfinance: `Ticker(t).calendar` for upcoming earnings dates.
- For sector peers, use the same sector map from Phase 5.
- For each ticker, return `{ticker, date_iso, time_bmo_amc, eps_est, revenue_est, options_iv_rank}`.
- IV rank: compute from current ATM IV vs 1-year range of ATM IV using
  yfinance options chain. Or skip if too slow — return null with a comment.

**Frontend** — new HTML block in `#news-panel`:
```html
<div style="margin-top:20px;">
  <div style="font-size:13px;font-weight:700;margin-bottom:10px;">
    📅 Earnings Calendar — next 7 days
  </div>
  <table id="earn-table">
    <thead>
      <tr><th>Ticker</th><th>Date</th><th>Time</th><th>Est. EPS</th>
          <th>Est. Revenue</th><th>IV Rank</th></tr>
    </thead>
    <tbody id="earn-tbody"></tbody>
  </table>
</div>
```

Rendering JS: highlight rows where `iv_rank > 70` (earnings-IV-crush plays)
in amber.

#### B) Fed Calendar row

Single-row widget showing next FOMC meeting date, current Fed Funds rate,
and market-implied probability of next move.

**Backend** — new endpoint `GET /api/fed-calendar`:
- Hardcoded next FOMC dates list (e.g. fetched once per quarter from
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).
  Actually: parse the page once per day, cache.
- Current rate: from FRED (FEDFUNDS series) via `core/macro.py`.
- Probability of next move: from CME FedWatch
  (https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html).
  If their JSON endpoint is reachable, fetch it; otherwise return null and
  fall back to a "Probability unavailable" line in the UI.

**Frontend** — new row above the macro indicator grid:
```html
<div class="fed-row" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:14px;">
  <div class="macro-card">Next FOMC: <span id="fed-next-date">—</span></div>
  <div class="macro-card">Current Rate: <span id="fed-rate">—</span></div>
  <div class="macro-card">P(hike|cut|hold): <span id="fed-prob">—</span></div>
</div>
```

### Files to touch (Phase 6)

| File | Action |
|---|---|
| `api/server.py` | New `/api/earnings-calendar` + `/api/fed-calendar` endpoints |
| `core/macro.py` | Helper for next FOMC date + FedWatch probability |
| `templates/dashboard.html` | Remove News Feed section from `#news-panel`; add Earnings + Fed Calendar widgets |
| `static/js/tabs/news.js` | NEW — render Earnings Calendar + Fed Calendar |
| `static/js/index.js` | Update extracted manifest |

### Verification

1. News & Macro tab now shows only: macro grid, VIX/F&G, Earnings Calendar,
   Fed Calendar. No news headlines (those moved to Sentiment).
2. Pick a ticker like AAPL — Earnings Calendar shows next 5-7 days of
   earnings for AAPL + sector peers (MSFT, GOOG, …).
3. Fed Calendar shows the next FOMC date and current Fed Funds rate.
4. If FedWatch is unreachable, the probability cell shows "—" not an error.

---

## 5. Phase 7 — Right Panel Audit

### Goal

Audit every card in the right sidebar (`#right-side`, lines 2348-2710 of
`templates/dashboard.html`). Keep / Fix / Replace per the original brief.

### KEEP (no changes needed)

| Card | Lines | Notes |
|---|---|---|
| Next-N candles probabilities (Up/Flat/Down + P10/Median/P90) | 2351-2366 | working |
| AI signal indicator bars (RSI/Slope/Momentum/MACD/Bollinger/ADX/Volatility) | 2370-2407 | working |
| Support & Resistance zones | 2544-2588 | working |
| Price Targets (Fib + S/R, Max High / Max Downside) | 2589-2626 | working |
| Drift & Risk panel | 2628-2657 | working |

### FIX

#### A) Confidence score color scale

Current: AI signal card shows confidence as just "23%" with a fixed blue bar.

Required: color scale by confidence value:
- `< 30%` → grey (`var(--muted)`)
- `30–50%` → yellow (`var(--amber)`)
- `> 50%` → green (`var(--green)`)

**Where**: the inline JS that paints `#conf-val` and `#conf-bar` (find with
`grep "conf-bar" templates/dashboard.html`). Should be in the right-panel
render path.

**How**: extend the inline update to also set `conf-val.style.color` and
`conf-bar.style.background` based on the thresholds. When extracting to
`static/js/right-panel.js`, take it with you.

#### B) Trade Setup dynamic

Current: Trade Setup card always shows "No Entry Right Now" / "Loading…"
banner regardless of signal.

Required: if `confidence > 40%` AND direction is clear (composite > 0.1 for
long, < -0.1 for short), show a suggested entry with SL/TP from the existing
`trade_setup` field on the response. Otherwise show "No Edge".

**Backend**: `/api/signal` and the WS broadcast already include
`result.trade_setup` (via `trade_setup_from_analysis` in
`core/trade_setup.py`). It returns `{valid: bool, side, entry, sl_atr,
sl_pct, tp1, tp2, rr_atr, rr_pct, prob_tp1, ...}`.

**Frontend**: find the inline render path for `#ts-banner`, `#ts-levels-wrap`,
`#ts-entry`, etc. (around the `_renderTradeSetup` inline function). Fix the
gating: render levels iff `trade_setup.valid && trade_setup.confidence > 0.40`.
Otherwise show "No Edge".

#### C) Walk-Forward Backtest

Current: button + result fields exist, `runBacktest()` already wired to
`POST /api/backtest`.

Verify it actually works. Look in inline JS for `runBacktest`. If broken,
fix the response handler to paint the four cells (`#bt-hit`, `#bt-brier`,
`#bt-ll`, `#bt-corr`).

If running the backtest is too compute-heavy or breaks for any reason after
inspection: **remove the entire card**. The brief explicitly says "do NOT
keep a broken UI element".

### REMOVE (after audit confirms)

Find anything in the right panel that always shows "—" regardless of ticker.
Likely candidates after a quick eyeball:
- The Microstructure card (`#ms-card`) — only shown when `mc_model ===
  'microstructure'`. Keep but verify the conditional display still works.

The audit should be: run the dashboard, switch tickers a few times, note any
card that doesn't change. Remove those.

### Extract

Move all right-panel rendering into `static/js/right-panel.js`. The inline
JS has a render function per card; pull them out and re-expose as
`window.renderRightPanel(data)`. Call it from the existing WS message handler
in dashboard.html.

### Files to touch (Phase 7)

| File | Action |
|---|---|
| `templates/dashboard.html` | Remove inline right-panel render JS; fix Trade Setup gating; fix confidence color scale; possibly remove broken cards |
| `core/trade_setup.py` | Already returns the right data; no changes expected |
| `static/js/right-panel.js` | NEW — all sidebar card rendering |
| `static/js/index.js` | Update extracted manifest |

### Verification

1. Switch tickers and observe: every visible card updates. No card shows "—"
   indefinitely.
2. Find a ticker with confidence < 30% → confidence pill is grey.
3. Find one with confidence > 50% → green. Mid range → amber.
4. When the signal is clear (e.g. AAPL on a trending day), Trade Setup shows
   actual Entry / SL / TP1 / TP2. When the signal is mush, it shows "No Edge".
5. Click Run on the backtest — fills in Hit Rate, Brier, Log Loss, ρ.

---

## 6. Phase 8 — Final Checks

### Tasks

1. **Run ruff**
   ```
   pip install ruff
   ruff check .
   ruff format .
   ```
   Fix all errors. Ignore warnings the existing config relaxes
   (see `pyproject.toml`).

2. **Run pytest**
   ```
   pytest -q
   ```
   Existing tests should pass. CHANGES.md mentions `tests/test_zones.py`,
   `tests/test_montecarlo.py`, `tests/test_backtest.py` but only
   `test_indicators.py`, `test_signal.py`, `test_store.py`, `test_api.py`
   exist. Add the missing ones if time permits, or note the gap.

3. **Browser console check**: open the dashboard, click through all 6 tabs,
   confirm no errors in the console for any tab switch. Look specifically
   for:
   - 404s on `/static/css/…` or `/static/js/…`
   - "Uncaught" exceptions
   - WS connection failures

4. **Ticker propagation**: change ticker in the header quick-ticker control.
   Confirm ALL 6 tabs reflect the new ticker (sentiment, options,
   market-structure, news, scanner, right panel).

5. **WS cleanup on unmount**: open DevTools → Network → WS. Open the
   sentiment tab (Phase 5's WS connects). Navigate away from the page →
   confirm both `/ws` and `/ws/news` close cleanly. If they leak, find the
   cleanup path in `sentiment.js` and wire it to `beforeunload` or
   `pagehide`.

6. **CHANGELOG**: append a Phase 5/6/7/8 section. The existing entries are
   in `CHANGELOG.md` — keep the same shape ("Root cause / Backend / Frontend /
   Files / Verify").

### Acceptance criteria (the brief's exact wording)

- `ruff check .` exits 0 (or with only pre-existing warnings).
- No console errors on tab switch for any of the 6 tabs.
- Ticker changes propagate to ALL tabs simultaneously (single source of
  truth in `currentConfig`).
- Live news WS properly closed on unmount (no memory leaks).
- CHANGELOG.md describes every structural change, every bug fixed, every
  section added or removed.

---

## 7. Conventions a new chat should follow

- **Don't move inline JS wholesale.** The "refactor in spirit" approach is to
  pull out tab-specific functions when you touch them, monkey-patching the
  inline versions on DOMContentLoaded (see `market-structure.js` for the
  pattern). This keeps each phase's diff small.
- **Keep IDs stable.** Every `id="…"` in the HTML is referenced from JS
  somewhere. Don't rename them without grepping first.
- **Backend additions go in `core/` modules**, not in `api/server.py`. The
  server file is route plumbing; the logic lives in `core/`.
- **Update `static/js/index.js`** when you add a new module. The
  `window.__mc_trader_modular__.extracted` array is the on-page indicator
  of refactor progress.
- **Defensive guards.** Every new feature should fail gracefully when the
  data source is missing. Existing pattern:
  `if (typeof window.foo === 'function') { try { window.foo(data); } catch (e) { console.warn(...); } }`.
- **No new external dependencies** unless absolutely needed. The project
  doesn't have a Node toolchain; adding one is out of scope.

---

## 8. Quick command reference

```bash
# Run the server
python main.py
# → http://localhost:8000

# Activate venv (Windows)
.venv\Scripts\activate

# Lint
ruff check .
ruff format .

# Tests
pytest -q

# Inspect refactor state in browser console
window.__mc_trader_modular__
```

---

End of handoff. CHANGELOG.md has the full Phase 2/3/4 history; this doc has
the forward-looking plan for Phases 5/6/7/8.
