# MC Trader - Next.js frontend

Next.js (App Router + TypeScript + Tailwind) frontend for the Monte Carlo trader.
It talks to the existing FastAPI backend. Five panels are live: Chart, Options
Flow, GEX/Max Pain, Scanner, and Signal & Monte Carlo. News, Backtest, and a
settings drawer are still to migrate off the legacy `templates/dashboard.html`.

---

## Run it

### One command (recommended)

From the repo root, `main.py` launches both the FastAPI backend and this
frontend (`npm run dev`), auto-installing npm packages on first run:

```bash
python main.py
# Backend  : http://localhost:8000   (legacy dashboard at /)
# Frontend : http://localhost:3000   (this app)
```

- `python main.py --no-frontend` (or `NO_FRONTEND=1`) - backend only
- `FRONTEND_PORT=4000` - custom frontend port

### Manual (two terminals)

```bash
# 1) Backend
uvicorn api.server:app --reload --port 8000
# 2) Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

`next.config.mjs` proxies `/api/*` and `/ws/*` to `http://localhost:8000`
(override with `BACKEND_URL`), so the browser sees one origin and no CORS setup
is required. Panels fall back to sample data when the backend is unreachable.

| Panel | Loads on open? |
|---|---|
| `/chart`, `/signal` | Yes — fetches `/api/signal` |
| `/gex` | Yes — loads default ticker (`PLTR`) |
| `/flow`, `/scanner` | No — click **Scan** to run |

After the backend comes up, use **Refresh** (chart/signal/gex) or **Scan**
(flow/scanner) for live data.

> **Requirements:** Node 18.18+ (works on Node 24). `.nvmrc` pins Node 20 LTS.
> React Strict Mode is off in `next.config.mjs` so slow scan endpoints are not
> fetched twice in dev.

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
SVG candlestick chart with a volume pane, client-computed EMA 9/21/200,
Bollinger, and VWAP overlays (toggleable), a crosshair with an OHLCV tooltip,
and Entry/Stop/Target lines from the trade setup.

### `/flow`
Sweeps and blocks feed with sortable columns and watchlist picker. Click Scan to
fetch from `/api/options/unusual`. Black-Scholes delta is computed client-side.

### `/gex`
Stat cards (spot, max pain, gamma flip, call/put walls, net GEX) and a vertical
net-GEX-by-strike bar chart. Backed by `GET /api/options/gex`.

### `/scanner`
`POST /api/scan` over a watchlist with summary strip, tabs, and a sortable table.
Click **Scan** to run; click a ticker to set it on the backend and jump to the chart.

### `/signal`
Signal and Monte Carlo with model/timeframe selectors, fan chart, probability
split, percentile ladder, CVaR, and horizon projection cards.

Shared UI: `components/Select.tsx` and `lib/display.ts`.

---

## Project layout

```
frontend/
  app/
    layout.tsx
    page.tsx
    chart/page.tsx
    flow/page.tsx
    gex/page.tsx
    scanner/page.tsx
    signal/page.tsx
    globals.css
  components/
    Nav.tsx
    Select.tsx, StatusBadge.tsx
    PriceChart.tsx, ChartPanel.tsx
    FlowFeed.tsx
    GexChart.tsx, GexPanel.tsx
    ScannerPanel.tsx
    SignalPanel.tsx, McFanChart.tsx
  lib/
    types.ts, api.ts
    analysisTypes.ts, analysisApi.ts
    scannerTypes.ts, scannerApi.ts
    gexTypes.ts, gexApi.ts
    blackScholes.ts, indicators.ts
    display.ts, format.ts
    sample*.ts
  next.config.mjs
  tailwind.config.ts
  .nvmrc
```

---

## Data caveats

yfinance returns chain snapshots, not a per-trade tape. On the flow panel
(manual **Scan** only):

- Sweep vs Block is approximated from volume/OI.
- Ask/Bid side is a last-vs-bid/ask lean, not an exchange trade condition.
- Time is the scan timestamp, shared by all rows in a scan.
- Size is contract day volume; the Volume column shows open interest.
- Horizon projections on `/signal` extrapolate MC drift/vol beyond the simulated
  window. For true simulated horizons, raise `MC_FORWARD_CANDLES` or run on `1d`.

---

## Migration checklist

- [x] Price chart + indicators -> `/chart`
- [x] Options flow -> `/flow`
- [x] Options GEX / Max Pain -> `/gex`
- [x] Breakout/breakdown scanner -> `/scanner`
- [x] Signal + Monte Carlo -> `/signal`
- [ ] News & sentiment stream -> `/news`
- [ ] Backtest -> `/backtest`
- [ ] Config / settings drawer

Keep the legacy dashboard at `/` on the backend until parity is reached.
