# MC Trader — Next.js frontend

A Next.js (App Router + TypeScript + Tailwind) frontend for the Monte Carlo
trader, talking to the existing FastAPI backend. Five panels are live —
**Chart, Options Flow, GEX/Max Pain, Scanner, and Signal & Monte Carlo** — with
News, Backtest, and a settings drawer still to migrate off the legacy
`templates/dashboard.html`.

---

## Run it

### One command (recommended)

From the repo root, `main.py` launches **both** the FastAPI backend and this
frontend (`npm run dev`), auto-installing npm packages on first run:

```bash
python main.py
# Backend  : http://localhost:8000   (legacy dashboard at /)
# Frontend : http://localhost:3000   (this app)
```

- `python main.py --no-frontend` (or `NO_FRONTEND=1`) — backend only
- `FRONTEND_PORT=4000` — run the dev server on another port

### Manual (two terminals)

```bash
# 1) Backend
uvicorn api.server:app --reload --port 8000
# 2) Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

`next.config.mjs` proxies `/api/*` and `/ws/*` to `http://localhost:8000`
(override with `BACKEND_URL`), so the browser sees one origin — no CORS to set
up. Every panel falls back to a **sample layout** when the backend is
unreachable, so the UI renders standalone; refresh once the API is up for live
data.

> **Requirements:** Node 18.18+ (works on Node 24). `.nvmrc` pins Node 20 LTS as
> the tested version. React Strict Mode is intentionally **off** in
> `next.config.mjs` so the slow scan endpoints aren't fetched twice in dev.

---

## Panels

| Route | Panel |
|---|---|
| `/` | Overview cards |
| `/chart` | Candlestick chart |
| `/flow` | Options flow feed (sweeps & blocks) |
| `/gex` | GEX / Max Pain |
| `/scanner` | Breakout / breakdown scanner |
| `/signal` | Signal & Monte Carlo |

### `/chart`
SVG candlestick chart with a volume pane, client-computed **EMA 9/21/200**,
Bollinger, and VWAP overlays (toggleable), a crosshair with an OHLCV tooltip,
and Entry/Stop/Target lines from the trade setup. EMAs are seeded from the first
bar so even EMA200 renders on a ~200-candle series.

### `/flow`
The sweeps & blocks feed: Time, Value, Ticker, Spot, Strike, PC, Exp., X
(Ask/Bid), Type (Sweep/Block), Price, Size, SigScore bar, Peak Return, Δ, Volume.
Sortable columns, watchlist picker, and a manual/auto refresh. Server filters on
`/api/options/unusual`: `min_sweep_premium=50_000`, `min_block_premium=100_000`,
`exclude_bid_side=true`, `exclude_high_volume_etfs=true`. Black–Scholes **Δ** is
computed client-side. Scans are wrapped in an `AbortController` so a superseded
scan is cancelled instead of resetting the socket. Defaults to the **momentum**
watchlist — `all optionable` (~500 tickers) is marked *(slow)*.

### `/gex`
Stat cards (spot, max pain, γ-flip, call/put walls, net GEX with
vol-damping/amplifying regime) and a **vertical net-GEX-by-strike** bar chart
(green positive / red negative) with Spot, γ-flip, and Max-pain markers and a
hover tooltip. Backed by `GET /api/options/gex`.

### `/scanner`
`POST /api/scan` over a watchlist: summary strip, All/Breakouts/Breakdowns/Neutral
tabs, and a sortable table with a centered signed **score bar** (number shown
beside it), direction, regime, signal, confidence, P(up), RSI/ADX/ATR%/Hurst.
Click a ticker to set it on the backend and jump to the chart.

### `/signal`
Signal + Monte Carlo. Stat strip, a **composite gauge** with confidence and the
sub-score breakdown, a regime potential bar, and a redesigned Monte Carlo card:
- an **MC model selector** (Gaussian, Student-t, GARCH, Bootstrap, Jump,
  Ensemble, Microstructure) that re-runs analysis,
- a **timeframe selector** (1m…1d) in the header,
- a smooth gradient **fan chart** (P10–P90 / P25–P75 bands, glowing median) with
  labeled **up / median / down price nodes** at the horizon edge,
- a probability split, percentile ladder, CVaR, and
- **horizon projection nodes** — 1 Day / 3 Days / 1 Week — projecting price and a
  ±0.67σ range by extrapolating the selected model's drift and vol.

---

## Project layout

```
frontend/
  app/
    layout.tsx          # shell + nav
    page.tsx            # overview
    chart/page.tsx
    flow/page.tsx
    gex/page.tsx
    scanner/page.tsx
    signal/page.tsx
    globals.css
  components/
    Nav.tsx
    PriceChart.tsx · ChartPanel.tsx
    FlowFeed.tsx
    GexChart.tsx · GexPanel.tsx
    ScannerPanel.tsx
    SignalPanel.tsx · McFanChart.tsx
  lib/
    types.ts            # options flow types + DEFAULT_FILTERS
    api.ts              # fetchFlow() + watchlists
    analysisTypes.ts    # /api/signal payload types
    analysisApi.ts      # fetchSignal, setConfigAndAnalyze, MC_MODELS, INTERVALS
    scannerTypes.ts · scannerApi.ts
    gexTypes.ts · gexApi.ts
    blackScholes.ts     # normCdf + bsDelta
    indicators.ts       # ema / sma / bollinger / vwap
    format.ts           # value/expiry/return formatters
    sample*.ts          # offline fallback data per panel
  next.config.mjs       # API/WS proxy to FastAPI, strict-mode off
  tailwind.config.ts    # dark theme tokens
  .nvmrc                # Node 20 LTS
```

---

## Data caveats (inherited from the snapshot source)

yfinance returns end-of-interval **chain snapshots**, not the per-trade tape, so
on the flow panel:

- **Sweep vs Block** is approximated from volume/OI.
- **X (Ask/Bid)** is a last-vs-bid/ask lean, not an exchange trade condition.
- **Time** is the scan timestamp, shared by all rows in a scan.
- **Size** = contract day volume; **Volume** column shows open interest.
- The **horizon projections** on `/signal` extrapolate MC drift/vol beyond the
  simulated window — for true simulated horizons, raise `MC_FORWARD_CANDLES` or
  run on the `1d` interval.

For a true per-print tape (real sweep/block tags, exchange side, trade times),
swap in a flow data provider (Polygon options trades, CBOE, Unusual Whales) by
replacing `fetchFlow()` in `lib/api.ts` and the backend endpoint.

---

## Migration checklist

- [x] Price chart + indicators → `/chart`
- [x] Options flow (sweeps & blocks) → `/flow`
- [x] Options GEX / Max Pain → `/gex` *(added `/api/options/gex`)*
- [x] Breakout/breakdown scanner → `/scanner`
- [x] Signal + Monte Carlo → `/signal`
- [ ] News & sentiment stream (websocket `/ws/news`) → `/news`
- [ ] Backtest (`POST /api/backtest`) → `/backtest`
- [ ] Config / settings drawer (`POST /api/config`)

Keep the legacy dashboard served at `/` on the backend until parity is reached,
then flip the default.
