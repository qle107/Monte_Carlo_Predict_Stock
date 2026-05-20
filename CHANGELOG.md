# CHANGELOG

## Phase 4 — Market Structure Fix

### Root cause

The Market Regime card and the Zone Reaction Probabilities table were
silently empty for a counter-intuitive reason: the heavy-analysis helpers
(`analyse_hmm`, `analyse_hawkes`) are gated behind `cfg.hmm_enabled` and
`cfg.hawkes_enabled` — both `False` by default in `config.py` to keep the
live poll loop's latency budget tight. When those flags were off, the
`/api/market-structure` endpoint returned `{"disabled": true}` for both
sections, and the frontend (`_renderHMM`, `_renderHawkes`) treated the
truthy-but-empty dict as a valid response and rendered blank cards.

The blended-zone loop also bailed out (`if hmm_result is not None`) so
"No demand/supply zones detected" was shown even when zones DID exist —
the message was actually "no HMM data, so I can't blend".

### Backend — `api/server.py` (rewrote `_api_market_structure_impl`)

- **HMM and Hawkes now ALWAYS run** in this endpoint, regardless of
  `cfg.hmm_enabled` / `cfg.hawkes_enabled`. Those flags stay in place to
  gate the live poll loop, but the dedicated `/api/market-structure`
  endpoint has its own loading spinner and is invoked by an explicit user
  click — the heavy work belongs here.
- New **state contract** on each sub-result:
  ```
  state ∈ {"ok", "error", "insufficient_data", "no_zones"}
  error_reason, min_bars_required, bars_available
  ```
  The frontend uses this to render meaningful empty/error states instead
  of blank cards.
- **Blended zones now fall back gracefully** when one (or both) of HMM /
  Hawkes is missing — every row also carries `blend_source ∈ {"hmm+hawkes",
  "hmm", "hawkes", "fallback"}` so the UI can caveat low-confidence rows.
  Previously the table was empty whenever HMM was missing; now it renders
  the zones with neutral 40/30/30 priors and a `fallback` tag.
- **Pipeline-step debug logs** at `data-fetch → preprocess → model-fit →
  classify → render-ready` (visible at log level DEBUG).
- HMM init order was audited and is correct: `pi`, `A`, `mus`, `sigs` are
  all initialised before the first `_forward` / `_backward` call inside
  `_baum_welch`. No bug there.

### Frontend — `static/js/tabs/market-structure.js` (NEW, ~265 lines)

Sits alongside the inline rendering code and monkey-patches the four
top-level functions (`_renderHMM`, `_renderHawkes`, `_renderBlendedZones`,
`_renderMarketStructure`) on `DOMContentLoaded`. Originals are preserved
as `*_original` for the happy path; the new code wraps every section in
a try/catch error boundary and routes the new `state` contract to a
state-aware empty/error renderer.

- **Insufficient data** → "📊 Market Regime — Need more candles … HMM
  requires at least 40 bars; 22 available. Switch to a longer timeframe."
- **Error** → "⚠ Market Regime — Error: Baum-Welch fit failed: ..."
- **No zones** → "🔍 Price Activity — No zones. Hawkes excitation needs
  demand/supply zones to score reactions at."
- Every error state includes a **↻ Retry** button that re-invokes
  `runMarketStructure()`.
- Each section is its own try/catch — a crash in HMM rendering can't
  break Volume Profile.
- `console.debug('[ms] …', payload)` at every render step (browser
  DevTools "Verbose" level shows them).

### Files touched

- `api/server.py` — rewrote `_api_market_structure_impl`. The endpoint
  signature and the `/api/market-structure` route are unchanged; only the
  response payload gains the new `state` / `error_reason` /
  `min_bars_required` / `blend_source` / `bars_available` fields.
- `templates/dashboard.html` — added the new `<script>` tag.
- `static/js/tabs/market-structure.js` — **new**.
- `static/js/index.js` — updated extracted/version markers.

### How to verify

1. `python main.py`, open `http://localhost:8000`.
2. Click the 🔬 Market Structure tab → click **Analyse**.
3. On a normal ticker (AAPL daily): **Market Regime** should show one of
   Trending / Ranging / Volatile with a probability breakdown; **Price
   Activity** should show Quiet / Normal / High Activity; **Zone Reaction
   Probabilities** should populate from the detected demand/supply zones.
4. Switch to a 1m or 2m interval with `lookback=30` (Settings) and re-run
   — you should see "📊 Market Regime — Need more candles" with min-bars
   hint and a Retry button.
5. Switch ticker to one without options (e.g. a small-cap with no chain)
   and confirm the GEX section shows its existing "Options data
   unavailable" error, while HMM/Hawkes still render normally.
6. Open browser DevTools console (set verbosity to Verbose) and confirm
   you see `[ms] data-fetch`, `[ms] preprocess`, etc. on each fetch.

---

## Phase 3 — Options Tab Fix

### A — Volume display

- Call Volume and Put Volume stay as plain positive numbers (already were, no
  sign flip). Re-confirmed the colour coding is by **type** (green = call,
  red = put), not by directionality.
- Added an explicit `title=` tooltip on each `.sent-kpi` cell:
  > Volume = total contracts traded. High put volume can mean buyers (bearish)
  > OR sellers (bullish covered puts). See Flow for direction.
- Added a matching tooltip on the call cell and on P/C Ratio Vol / P/C Ratio OI.

### B — Flow direction disclaimer

`yfinance` does not expose aggressor side (buy- vs sell-initiated trades), so the
spec-prescribed "Buy Calls / Sell Calls / Buy Puts / Sell Puts" metrics aren't
computable from this data source. A visible amber banner now sits at the top of
`#opts-results`:

> ⓘ **Flow direction unavailable from this source.** yfinance reports total
> contracts traded, not buy- vs sell-initiated trades. High volume on a strike
> can mean buyers *or* sellers — colour-coding below indicates contract *type*
> (call vs put), not directionality.

The backend response now carries `options_flow.flow_direction_available = false`
so a future paid feed (Tradier / Polygon / Unusual Whales) can flip this flag
and the UI can hide the banner / show real B/S splits without further changes.

### C — Unusual Options Activity detector

**Backend (`core/sentiment.py`).** New helper `_scan_unusual_activity(calls_df,
puts_df, exp_str, today)` flags any contract where
`volume > open_interest × 1.5` AND `volume > 500`. The detector runs against
the **full chain** for the first 3 expiries and an additional 5 longer-dated
expiries (up to 8 total) so LEAPS-type unusual flow is caught. Each row carries
`{expiry, dte, strike, type, volume, oi, vol_oi, premium, flow, pct_change, iv,
in_money, leaps}`. The response field `options_flow.unusual_activity` is sorted
by `vol_oi` descending and capped at 50 rows so the JSON payload stays small.

**Frontend (`templates/dashboard.html` + `static/js/tabs/options.js`).**
A new "🚨 Unusual Activity" section sits below "Hot Strikes & Price Levels"
inside the Options tab. It has:

- A DTE filter row (All / 0–7 / 8–30 / 31–60 / 60+ days · LEAPS), client-side
  filtering on the cached `unusual_activity` array — no extra network call.
- A sortable table with columns Strike, Type, Expiry, DTE, Volume, OI,
  **Vol/OI**, Premium, Flow $, Chg%, Sentiment.
- Default sort: Vol/OI descending. Click any column header to re-sort; text
  columns default to ascending on first click, numeric columns to descending.
- LEAPS rows (DTE > 60) get a purple `LEAPS` chip next to the DTE column so
  the eye picks them out even when "All" is selected.
- The Sentiment column uses **type-based** labels (📞 Call / 🔻 Put) with
  intensity from Vol/OI ratio (Heavy at ≥3×, Extreme at ≥5×). It deliberately
  does NOT say bullish / bearish — the disclaimer at the top of the tab
  explains why. ITM contracts get a trailing dot.

The new JS module lives at `/static/js/tabs/options.js` and exposes
`window.setUnusualFilter` (for the inline `onclick=…`) and
`window.renderUnusualActivity` (called from the inline `_renderSentimentPanel`
in dashboard.html). The script is loaded with the other extracted modules
right after the three CSS extraction comments.

### Files touched

- `core/sentiment.py` — added `_scan_unusual_activity`; extended
  `_options_flow_sync` with deeper expiry scan + new return fields.
- `templates/dashboard.html` — added disclaimer banner, KPI tooltips,
  Unusual Activity section, options.js script tag, shim call in
  `_renderSentimentPanel`.
- `static/js/tabs/options.js` — **new** (~150 lines).
- `static/js/index.js` — updated extracted/version markers.

### How to verify

1. `python main.py`, open `http://localhost:8000`.
2. Click the 📈 Options tab → click **Fetch Options**.
3. Confirm the amber **Flow direction unavailable** banner shows at the top.
4. Hover the Calls Vol / Puts Vol KPI cells → tooltip should appear.
5. Scroll to **🚨 Unusual Activity** — verify the table populates for any
   ticker with active options (try AAPL, NVDA, TSLA).
6. Click DTE filter pills and column headers — list updates in place.
7. Confirm LEAPS chips appear on rows with DTE > 60.

---

## Phase 2 — Frontend Refactor (in progress)

**Decision recap.** This project is a FastAPI backend serving a single 7019-line
`templates/dashboard.html` (vanilla HTML + inline CSS + inline JS — no React,
no build tool, no npm). The Phase 2 brief was written for a JS-framework
project. After confirming with the user we chose **"stay vanilla, refactor in
spirit"**: pull CSS and JS into `/static/` files served by FastAPI's existing
`StaticFiles` mount, keep classic `<script>` tags so the existing `onclick=…`
handlers continue to work, and defer per-tab JS extraction into Phases 3-7
(each tab pulls out its own functions when it's touched).

### Structural changes

**New files**

- `static/css/dashboard.css` — main dashboard stylesheet (~700 selectors,
  was 3 inline `<style>` blocks at HTML lines 9-745, 851-994 (portfolio
  drawer was kept separate), and 2743-2985).
- `static/css/portfolio.css` — portfolio drawer styles, scoped under
  `#portfolio-drawer` so they cannot leak.
- `static/js/scanner.js` — Breakout / Breakdown scanner logic (~385 lines,
  was lines 2987-3372 inline). Contains `runScanner`, `processScanResult`,
  `buildTopCards`, `setScanSort`, `filterScan`, `renderScanTable`,
  `loadTicker`, and the `_tsSortAdapters` map.
- `static/js/index.js` — manifest barrel for the front-end module layer.
  Currently a marker module; will become the real bootstrap as Phases 3-7
  pull more JS out of the HTML file.
- `pyproject.toml` — adds `ruff` (lint + format) and `pytest` config.

**`templates/dashboard.html` edits**

- Inline `<style>` block 1 (head, lines 9-745) → replaced by
  `<link rel="stylesheet" href="/static/css/dashboard.css">`. The original
  block is wrapped in an HTML comment (`<!-- legacy-css-1 … end-legacy-css-1 -->`)
  so the diff is auditable. The user can delete the commented region once
  the refactor is verified.
- Inline `<style>` block 2 (inside `#portfolio-drawer`) → replaced by
  `<link rel="stylesheet" href="/static/css/portfolio.css">`, original
  wrapped in `legacy-css-2` comment.
- Inline `<style>` block 3 (body, lines 2743-2985) → merged into
  `dashboard.css` and original wrapped in `legacy-css-3` comment.
- Inline scanner `<script>` (block 1, ~385 lines) → replaced by two
  external script tags: `<script src="/static/js/index.js">` and
  `<script src="/static/js/scanner.js">`. The original inline block now has
  `type="text/x-disabled-legacy"` so the browser parses but doesn't execute
  it — also auditable, also deletable once verified.

### Behaviour

- Page-load order: external `scanner.js` runs before the still-inline main
  JS, which is what the existing code expects (`scanner.js` references
  `currentConfig`, `applySettings`, and `log` — all of which are still
  defined in the main inline block).
- `onclick="runScanner()"`, `onclick="loadTicker(...)"`, etc. continue to
  resolve because `scanner.js` runs in the global scope (it's a classic
  script, not an ES module).
- No build step required. Refresh the page in the browser and FastAPI's
  `StaticFiles` mount serves the new files at `/static/css/…` and
  `/static/js/…`.

### Deferred to later phases

These were in the original Phase 2 brief but make more sense to do
incrementally:

- **Per-tab JS modules** (`tabs/sentiment.js`, `tabs/options.js`, etc.):
  the main inline JS is one ~4000-line block whose functions reference each
  other heavily. Pulling each tab out atomically would require analysing
  the whole block in one pass. Instead, each tab gets extracted in its
  feature phase: Phase 3 pulls out the Options tab, Phase 4 pulls out
  Market Structure, Phase 5 pulls out Social Sentiment, Phase 6 pulls out
  News & Macro, Phase 7 pulls out the right-side panel cards.
- **Lazy-load each tab on first switch**: depends on the per-tab extraction
  above. Once each tab is in its own file we change `switchScannerTab()`
  to do a dynamic `<script>` injection on first open.
- **Portfolio tracker JS** (~365 lines, still inline): will be extracted
  when the portfolio code is next touched.

### Linting

`ruff` is now configured in `pyproject.toml` with a pragmatic ruleset
(pyflakes + pycodestyle + isort + bugbear + comprehensions + pyupgrade +
simplify, with the usual relaxations: E501 line length, E731 lambdas,
E741 ambiguous math names in `core/montecarlo.py`/`signal.py`/`indicators.py`).
Run `ruff check .` and `ruff format .`.

JS linting was skipped intentionally — adding ESLint requires a Node
toolchain that this project does not have. The proper verification is
loading the dashboard in a browser and checking the JS console.
