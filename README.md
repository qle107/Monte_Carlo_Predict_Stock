# Monte Carlo Predict Stock

A self-hosted FastAPI service that turns OHLCV candles into a regime-aware
directional signal, runs a Monte Carlo simulation under one of seven models,
and serves a live dashboard with trade setup, scanner, and backtest — all in
a single process.

**Research / paper-trading only. Not a broker integration. Not a recommendation engine.**

---

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                   # dashboard at http://localhost:8000
```

Copy `.env.example` to `.env` and set at minimum:

```
TICKER=AAPL
CANDLE_INTERVAL=15m
MC_MODEL=garch
```

For real-time bars set `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`. The fetcher
falls through Alpaca → Polygon → yfinance automatically.

```bash
pytest -q          # 58 tests
ruff check .       # 0 errors
```

---

## Architecture

```
main.py
 └─ api/server.py          FastAPI app — routes, WebSocket, poll loop
     └─ core/__init__.py   analyse(df) → JSON dict
         ├─ indicators.py  RSI, EMA, MACD, BB, ADX, OBV, VWAP, kurtosis
         ├─ regime.py      Hurst / R² / Donchian / ADX composite → 8 labels
         ├─ signal.py      Regime-weighted composite score + confidence
         ├─ zones.py       Demand/supply zone detection (pivot → cluster → score)
         ├─ montecarlo.py  7 path models (see below)
         ├─ trade_setup.py Entry / SL / TP / RR + Kelly sizing
         ├─ backtest.py    Walk-forward harness with per-trade stats
         └─ store.py       SQLite signal log
```

Optional enrichments (called on-demand, not in the poll loop):

| Module | Endpoint |
|---|---|
| `hmm_regime.py` | `/api/market-structure` |
| `hawkes.py` | `/api/market-structure` |
| `options_flow.py` | `/api/sentiment` |
| `sentiment.py` | `/api/sentiment`, `/ws/news` |
| `macro.py` | `/api/macro` |

---

## Monte Carlo models

Select via `MC_MODEL` env var or `POST /api/config`.

| Model | Innovation | Best for |
|---|---|---|
| `gaussian` | GBM / Normal | Calm regimes, baseline |
| `student_t` | Student-t (df from kurtosis) | Fat-tailed returns |
| `garch` | GARCH(1,1) σ-path | Volatility clustering (default) |
| `bootstrap` | Resampled historical returns | Unknown distribution shape |
| `jump` | Merton jump-diffusion | Earnings / event tails |
| `ensemble` | GARCH + bootstrap + jump blend | Most-robust default |
| `microstructure` | GARCH + vol profile + CVD + Hurst | Level-aware path generation |

---

## Key configuration

All tunables live in `config.py` as a `Config` dataclass; everything can be
overridden via `.env` or `POST /api/config` at runtime.

| Var | Default | Notes |
|---|---|---|
| `TICKER` | `PLTR` | |
| `CANDLE_INTERVAL` | `15m` | 1m 2m 5m 15m 30m 1h 4h 1d |
| `MC_MODEL` | `garch` | |
| `MC_SIMULATIONS` | `2000` | 10000 for research |
| `MC_FORWARD_CANDLES` | `5` | Forecast horizon in bars |
| `LOOKBACK` | `50` | History bars fed to analysis |
| `POLL_SECONDS` | `120` | |
| `HMM_ENABLED` | `False` | Adds ~3–10 s per call |
| `HAWKES_ENABLED` | `False` | Adds ~3–10 s per call |
| `API_KEY` | _(unset)_ | If set, all `/api/*` routes require `X-Api-Key` header |

---

## HTTP API

```
GET  /                         Dashboard
GET  /api/health
GET  /api/signal               Force fresh analysis
GET  /api/config
POST /api/config               Update ticker, interval, model, etc.
POST /api/backtest
GET  /api/history
GET  /api/metrics
GET  /api/metrics/accuracy
GET  /api/export.csv
POST /api/scan                 Breakout/breakdown scanner
POST /api/zone-scan            Zone + EMA scanner
GET  /api/market-structure     HMM + Hawkes + blended zones
GET  /api/sentiment
GET  /api/sentiment/global
GET  /api/news
GET  /api/fear-greed
GET  /api/macro
WS   /ws                       Server-push — new analysis on every poll
```

---

## Frontend

Single-page dashboard at `/` — no build step, no npm. Static assets in
`static/css/` and `static/js/` are served by FastAPI's `StaticFiles` mount.

Extracted JS modules (all IIFEs, loaded as classic `<script>` tags):

| File | Owns |
|---|---|
| `static/js/scanner.js` | Breakout/breakdown scanner UI |
| `static/js/tabs/options.js` | Options flow + unusual activity table |
| `static/js/tabs/market-structure.js` | HMM / Hawkes error boundaries + retry |
| `static/js/tabs/sentiment.js` | `/ws/news` feed with backoff reconnect |
| `static/js/right-panel.js` | Confidence colour, trade-setup gate, backtest KPIs |

---

## Project layout

```
├── main.py
├── config.py
├── requirements.txt
├── pyproject.toml           ruff + pytest config
├── CHANGELOG.md
├── api/
│   ├── server.py
│   └── models.py
├── core/                    Analysis pipeline (see Architecture above)
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── dashboard.html
└── tests/                   58 tests, pytest -q
```

---

## Disclaimer

For educational and research purposes only. Monte Carlo simulation describes
a hypothetical distribution under modelling assumptions — it does not predict
the future. Paper trade before risking real capital.
