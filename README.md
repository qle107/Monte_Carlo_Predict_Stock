# Monte Carlo Predict Stock

Stock research API: regime signal, Monte Carlo paths, options flow, GEX,
scanner, and backtest. FastAPI backend with WebSocket streaming.

```bash
python main.py   # http://localhost:8000/docs
```

Research only. Not investment advice.

## Quick start

Python 3.10+.

```bash
git clone https://github.com/qle107/Monte_Carlo_Predict_Stock.git
cd Monte_Carlo_Predict_Stock

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env

python main.py
```

Set `CORS_ORIGINS` in `.env` for browser clients on another origin.

## Configuration

Copy `.env.example` to `.env`. Default ticker is `DEFAULT_TICKER` in `config.py`.
Price data: Alpaca, Polygon, then yfinance. Runtime changes via `POST /api/config`.

## Layout

```
main.py              uvicorn launcher
api/                 routes, websockets, analysis orchestration
core/                signal, MC, flow, scanner, backtest
docs/math.md         model notes
tests/               pytest
portfolio_tracker.html  standalone HTML client
```

## Monte Carlo models

| Model | Notes |
|---|---|
| `gaussian` | GBM baseline |
| `student_t` | Fat tails from kurtosis |
| `garch` | Default; vol clustering |
| `bootstrap` | Resampled returns |
| `jump` | Merton jumps |
| `ensemble` | Blend of several |
| `microstructure` | GARCH + volume profile + CVD |

Details in [docs/math.md](docs/math.md).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/signal` | Run analysis |
| GET/POST | `/api/config` | Read/update settings |
| GET | `/api/options/unusual` | Options flow scan |
| GET | `/api/options/gex` | GEX profile |
| GET | `/api/options/contract` | Contract premium history |
| POST | `/api/scan` | Breakout scanner |
| POST | `/api/backtest` | Backtest |
| WS | `/ws` | Live analysis push |
| WS | `/ws/news` | News stream |

Protected routes need `X-Api-Key` when `API_KEY` is set.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `ruff check .`, `ruff format --check .`,
and `pytest -v` before opening a PR.
