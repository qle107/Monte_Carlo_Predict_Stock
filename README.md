# Monte Carlo Predict Stock

A self-hosted Python service that turns a stream of OHLCV candles into a
forecast: it computes a regime-aware composite signal, runs a Monte Carlo
simulation under one of seven innovation models, scores the result against
demand/supply zones, and serves the whole thing — chart, trade setup,
scanner, backtest — over a single FastAPI process.

This is a research / paper-trading tool. It is not a recommendation
engine and it is not connected to a broker. See the **Disclaimer** at the
bottom.

## Status

The analysis pipeline was audited and refactored in May 2026. The
short version of what changed and why is in **[CHANGES.md](CHANGES.md)** —
read that first if you used an earlier version, because the demand/supply
zone scorer, the Monte Carlo engine, the backtest, and the trade-setup TP
logic all had material bugs that are now fixed. The public APIs
(`core.analyse`, `core.zones.detect_zones`, `core.montecarlo.run`,
`core.backtest.walk_forward`, `core.trade_setup.compute_trade_setup`) and
the HTTP routes are unchanged, so the dashboard still works without
modification.

## Quick start

The service needs Python 3.10 or newer and the dependencies listed in
`requirements.txt` (uvicorn, fastapi, pandas, numpy, scipy, yfinance,
alpaca-py, httpx, python-dotenv, websockets).

```bash
cd Monte_Carlo_Predict_Stock
python -m venv .venv
.venv\Scripts\activate            # Windows; on Unix: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The dashboard is served at `http://localhost:8000`. Override anything via
a `.env` file in the project root — at minimum you can set `TICKER`,
`CANDLE_INTERVAL`, `MC_MODEL`, and `MC_SIMULATIONS`; the full list of
tunables is the `Config` dataclass in `config.py`.

For real-time bars instead of the default delayed yfinance feed, set
`ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in the `.env` file. The fetcher
in `core/fetcher.py` tries Alpaca first when keys are present, falls
through to Polygon if `POLYGON_API_KEY` is set, and finally to yfinance.

To run the test suite:

```bash
pytest -q
```

## Architecture

`main.py` boots uvicorn and points it at `api:app`. The FastAPI app in
`api/server.py` owns three things: the HTTP routes, a single WebSocket
endpoint that pushes a fresh analysis on every poll, and a background
poll loop that fetches new candles, runs `core.analyse`, persists the
result to SQLite, and broadcasts it.

The `core/` package is the analysis pipeline. Each module is a focused
stage:

| Module | Responsibility |
|---|---|
| `core/fetcher.py` | Candle fetching with Alpaca → Polygon → yfinance fallback, retry, and request coalescing. |
| `core/indicators.py` | RSI, EMA stack, MACD, Bollinger, ADX, OBV slope, VWAP distance, kurtosis/skew, RSI divergence, 52-week distance. |
| `core/regime.py` | Hurst (R/S), multi-window R², Donchian, swing pivot HH/HL counts, range compression — composited into one of seven regime labels. |
| `core/signal.py` | Composite score in `[-1, +1]` with regime-aware weight blending and entropy-aware confidence. Sets the drift bias for the Monte Carlo. |
| `core/zones.py` | Demand/supply zone detection (pivots → cluster → score → break filter). |
| `core/montecarlo.py` | Seven path-simulation models, GARCH MLE with caching, microstructure context (volume profile, CVD, Hurst regime). |
| `core/trade_setup.py` | Entry/SL/TP/RR using MC percentiles, ATR, and zones. Kelly and fixed-fractional position sizing. |
| `core/backtest.py` | Walk-forward backtest with cost-aware per-trade stats. |
| `core/scanner.py`, `core/zone_scanner.py` | Concurrent multi-ticker scanners. |
| `core/store.py` | SQLite signal log used by `/api/history` and `/api/metrics`. |
| `core/hawkes.py`, `core/hmm_regime.py`, `core/options_flow.py`, `core/sentiment.py`, `core/macro.py`, `core/volume_profile.py` | Optional enrichments surfaced by the `/api/market-structure`, `/api/sentiment`, `/api/macro` endpoints. |

`core/__init__.py` exposes `analyse(df, n_simulations, n_forward, mc_model)`,
which is the one function the HTTP layer actually calls. It returns a
JSON-serialisable dict containing indicators, regime, signal, MC result,
candles, and warnings.

## The signal

The signal answers a deliberately narrow question: over the next
`n_forward` candles, what is the directional bias and how confident are
we in it? `core/signal.py` computes it in three steps.

First, each indicator is mapped to a sub-score in `[-1, +1]` by a
hand-tuned function — for example `_score_rsi` saturates at ±0.60 at
extremes, `_score_macd` is a linear function of the histogram clipped to
±0.60, and `_score_adx` scales by both ADX strength and slope direction
so a high-ADX downtrend contributes negatively.

Second, the sub-scores are combined with **regime-aware weights**. There
are five weight maps (strong trend, weak trend, breakout, range-bound,
choppy) that sum to 1.0 each. In a trending regime, slope/MACD/ADX carry
~60% of the weight and RSI/Bollinger nearly nothing; in a range-bound
regime RSI and Bollinger dominate and slope is suppressed. The regime
also contributes its own `trend_score` blended in at 0.55 weight.

Third, the **confidence** is computed from the entropy of the active
sub-signals' sign agreement, multiplied by the average magnitude. A
high-confidence signal therefore requires both *agreement* across
indicators *and* meaningful magnitudes — a single strong-but-isolated
component cannot run away with the call.

The composite score and confidence then set the drift bias fed to the
Monte Carlo:

```
base_drift  = empirical mean per-candle return
signal_adj  = composite × confidence × (0.5 × std_return)
drift_bias  = clip(base_drift + signal_adj, ±2 × std_return)
```

The ±2σ cap is the "no 99.9% certain" guard — it stops a strong signal
from collapsing the simulation onto a single direction. On a gap day
(`|gap| > gap_threshold`, default 3%), `drift_bias` is set to zero and
volatility is inflated 1.6×, because the historical mean is meaningless
after a news-driven gap.

## Monte Carlo models

All seven models share the same shape (`n_sim` paths, `n_candles + 1`
columns including the spot at column 0) and return the same
`MCResult` schema, so the dashboard renders any of them identically. The
model is selected by the `MC_MODEL` env var or the runtime
`POST /api/config` call.

| Model | Innovation process | When it helps |
|---|---|---|
| `gaussian` | GBM with Normal innovations. | Baseline; calm regimes. |
| `student_t` | Student-t innovations rescaled to unit variance, df fit from observed excess kurtosis (`df = 4 + 6/κ`). | When recent returns are fat-tailed. |
| `garch` (default) | GARCH(1,1) σ-path with the α, β configurable; `_calibrate_garch` adapts them to the current vol regime. | Volatility clustering — gap days, post-news bars. |
| `bootstrap` | Resamples centred historical returns rescaled to match the target σ. | When the empirical distribution shape matters more than its parametric form. |
| `jump` | Merton jump-diffusion: Gaussian diffusion plus Poisson-triggered Gaussian jumps with the compensator `λ·μ_J` subtracted from drift. | Earnings, FOMC, anything with a known discrete-event tail. |
| `ensemble` | Data-driven blend of GARCH + bootstrap + jump; weights are functions of vol-of-vol (favours GARCH when regime is unstable) and excess kurtosis (favours jumps when tails are heavy). | The most-robust default when you don't know which model is right. |
| `microstructure` | GARCH + Student-t(df=4) with a per-step **gravity** field derived from the volume profile (POC, VAH, VAL, HVN, LVN), a CVD-driven drift bias, and a Hurst-based regime multiplier. | When level-aware path generation matters — e.g. evaluating a setup near POC. |

Two implementation notes from the May 2026 refactor that are worth
knowing if you are reading the code:

- The diffusive `MC_CLIP` cap (default 0.25) is widened to 0.5 for
  `jump`, `student_t`, and `ensemble`, because clipping at the diffusive
  level was silently truncating the tails those models exist to produce.
- The "flat" probability band in `_build_mc_result` now scales with the
  cross-path standard deviation (`0.25 × σ_horizon`, floored at 25 bps)
  instead of being a fixed 30 bp band. That makes `prob_flat`
  meaningful at long horizons.

The MC result includes per-step P25/P75 and P10/P90 bands (the inner and
outer confidence cones the dashboard draws), the median path, a sample
of 100 paths for the chart, the expected return and its 5% CVaR, and
the full `(n_sim, n_steps+1)` path matrix that `trade_setup.py` uses to
compute path-aware TP/SL probabilities. The microstructure model also
attaches `ms_regime`, `ms_hurst`, `ms_drift_bias`, and `ms_key_levels`
diagnostics.

## Demand and supply zones

`core/zones.py` was the most-broken part of the codebase before the May
2026 pass. The algorithm now is:

1. **Pivots.** Find swing highs and lows using a `±zone_pivot_window`
   look-around (default ±4 bars). A pivot low is a candidate demand
   origin; a pivot high is a candidate supply origin.

2. **Cluster.** Merge pivots that fall within `zone_cluster_atr × ATR`
   of each other into a single zone. Clustering anchors on the *seed*
   price of each cluster, not the running mean, so a long string of
   pivots cannot drift the cluster outward indefinitely.

3. **Score.** Each zone gets a strength in `[0, 1]`:

   ```
   strength =  0.35 · touch_score      # min(touches / 4, 1)
             + 0.25 · recency_score    # bar_idx / (n_bars - 1)
             + 0.20 · freshness_score  # 1 if never retested after formation
             + 0.20 · depth_score      # |close - open| / range at the pivot bar
   ```

   The depth term means a clean rejection candle (close far from open
   relative to range) outscores a doji at the same level. The freshness
   flag flips to `False` only when a bar *after* the formation bar
   re-enters the touch band.

4. **Filter broken zones.** A demand zone is broken if any subsequent
   bar's *low* — not close — traded more than `zone_break_atr × ATR`
   below the level. Supply uses subsequent highs. Using intraday extrema
   catches wick violations that close-only filters would miss.

5. **Select nearest.** Return up to `zone_max_demand` (default 5) and
   `zone_max_supply` zones sorted by strength, then pick the nearest
   meaningful zone on each side of price. "Meaningful" means strength
   above 0.30; if no zone clears that bar we fall back to the closest of
   any strength so the dashboard always has something to show.

The `ZoneResult` exposes the full sorted lists, the two nearest zones,
and a `price_context` of `at_demand`, `at_supply`, `between`, or
`unknown`. The trade-setup engine reads `price_context` to decide which
of four zone scenarios applies (demand bounce, supply break, supply
bounce, demand break).

## Regime detection

`core/regime.py` produces one of eight labels —
`strong_uptrend`, `weak_uptrend`, `strong_downtrend`, `weak_downtrend`,
`breakout_up`, `breakout_down`, `range_bound`, `choppy` — from a
weighted composite of:

- R² and slope of three regression windows (short, mid, long, default
  10/20/50 bars), each signed by the slope direction;
- the Hurst exponent (R/S estimator on log-prices, clamped to `[0, 1]`),
  signed by the long-window slope;
- Donchian position (where the current close sits between the N-bar
  high and low) plus breakout flags for the N-bar, 10-bar, and
  within-3 %-of-high consolidation cases;
- HH/HL/LH/LL pivot counts;
- ADX scaled and signed;
- OBV slope.

The label decision tree is in `_label_regime`. Two rules worth flagging:
a `strong ADX > 30` overrides `range_bound` (a stock with real
directional pressure is not range-bound even if recent spread is tight),
and the `range_compression` score is suppressed when a large recent gap
is present so post-breakout consolidation does not get mislabelled.

The regime also produces the `trend_score` blended into the composite
signal, and the verdict string the dashboard shows.

## Trade setup and backtest

`core/trade_setup.py` turns the analysis output into a concrete plan.
Entry is the current close. Stop-loss is computed two ways and the
tighter is recommended: an **ATR-based** stop using a regime-dependent
multiplier (1.5×–3.0×) capped by a per-timeframe maximum percentage
(3% for 1m, 12% for 1d), and a **fixed-percentage** stop whose base is
also regime-dependent and is widened to cover at least 1.5× ATR. Targets
prefer the MC P75/P90 percentiles, falling back to an ATR projection
when the MC distribution sits on the wrong side of entry — this replaces
the pre-refactor bug where TP1 was inflated to a flat `entry × 1.015`
whenever P75 came in tight.

Zone-aware targets are computed alongside the MC targets: when the
`price_context` is `at_demand` the zone scenario is `demand_bounce`,
giving a zone SL just below the zone low and zone TPs at the nearest
supply zones above; similar logic for the three other quadrants. The
output dataclass `TradeSetup` carries both target sets so the dashboard
can render them side by side.

Position sizing is computed two ways: a half-Kelly using
`prob_tp1` as the win probability and the best R:R as the payoff
ratio, and a 1%-of-equity fixed-fractional sizing using the tighter of
the two stops.

`core/backtest.py` is a walk-forward harness: it slides through history,
recomputes the full pipeline on each bar's prefix, runs the MC, and
compares to the realised `n_forward`-bar move. The May 2026 pass fixed
four things in this module — the "hit" threshold is now consistently
`cfg.backtest_band_pct` everywhere (previously three different values
were in use), trades no longer overlap by default (a 10-bar Buy signal
used to be counted as 10 trades), the Sharpe annualisation respects the
bar interval instead of always using `sqrt(252)`, and max drawdown is
computed on the wealth curve `cumprod(1 + r)` instead of `cumsum(r)`
which blew up when the curve passed through zero.

Reported metrics include hit rate, Brier score, log-loss, expected-vs-
realised correlation, calibration over five `prob_up` buckets,
annualised Sharpe, max drawdown, average win/loss, win/loss ratio,
profit factor, and maximum consecutive losses.

## Dashboard

The dashboard is a single-page app served at `/`. It shows the live
candlestick chart with the MC confidence cone (P25–P75 inner band,
P10–P90 outer band), the median path, the regime banner, the
indicator breakdown, the MC probability bars, the trade-setup card
(MC targets plus zone targets), the walk-forward backtest summary,
and tabs for the two scanners (breakout and zone+EMA). The gear icon
lets you change ticker, timeframe, MC model, `n_sim`, `n_forward`,
`lookback`, and `poll_seconds` without restarting the server — the
poll loop picks up the new config on the next cycle.

There is also a portfolio tracker overlay that stores positions in the
browser's `localStorage` and refreshes prices through the
`/api/portfolio/price/*` proxy endpoints. The tracker is purely
front-end state; nothing is persisted server-side.

## HTTP API

```
GET  /                                                dashboard HTML
GET  /api/health                                      liveness probe
GET  /api/signal                                      force a fresh analysis cycle
GET  /api/config                                      current config + valid choices
POST /api/config                                      update any config field (validated)
POST /api/backtest                                    walk-forward backtest
GET  /api/history                                     recent persisted signals
GET  /api/metrics                                     per-ticker accuracy stats
GET  /api/metrics/accuracy                            per-bucket calibration
POST /api/store/prune                                 trim signal store
GET  /api/export.csv                                  CSV dump of signal history
POST /api/scan                                        breakout/breakdown scanner
GET  /api/scan/watchlists                             list available watchlists
POST /api/zone-scan                                   zone + EMA strategy scanner
GET  /api/market-structure                            HMM regime + Hawkes excitation + zone blend
GET  /api/sentiment                                   per-ticker news sentiment
GET  /api/sentiment/global                            broad-market sentiment
GET  /api/news                                        news feed with classification
GET  /api/fear-greed                                  CNN fear/greed proxy
GET  /api/macro                                       FRED-sourced macro indicators
GET  /api/portfolio/price/historical?ticker=&date=    historical close for the portfolio tracker
GET  /api/portfolio/price/live?ticker=                live price for the portfolio tracker
WS   /ws                                              server-push: new analysis on every poll
```

Representative request bodies:

```jsonc
// POST /api/config — all fields optional
{
  "ticker": "AAPL",
  "interval": "15m",
  "mc_model": "ensemble",
  "n_sim": 10000,
  "n_forward": 10,
  "lookback": 50,
  "poll_seconds": 60
}

// POST /api/backtest
{ "history_bars": 200, "n_forward": 10, "n_sim": 500, "mc_model": "garch" }

// POST /api/scan
{ "watchlist": "tech", "interval": "1d", "lookback": 60, "max_concurrent": 8 }

// POST /api/zone-scan
{ "watchlist": "sp500_large", "interval": "1d", "lookback": 120 }
```

If `API_KEY` is set in the environment, every `/api/*` route requires
an `X-API-Key` header that matches.

## Configuration

The full set of tunables lives in the `Config` dataclass at the top of
`config.py`. Anything not set in the environment uses a defensible
default; out-of-range values fall back to the default with a warning so
a malformed `.env` cannot crash the server.

The fields that matter most in day-to-day use:

| Env var | Default | Purpose |
|---|---|---|
| `TICKER` | `PLTR` | Active ticker for the dashboard's poll loop. |
| `CANDLE_INTERVAL` | `15m` | One of 1m, 2m, 5m, 15m, 30m, 1h, 4h, 1d. |
| `MC_MODEL` | `garch` | Default Monte Carlo innovation model. |
| `MC_SIMULATIONS` | `2000` | Paths per simulation. ~2000 is the sweet spot for a live dashboard; 10000 for research. |
| `MC_FORWARD_CANDLES` | `5` | Forecast horizon in bars. |
| `LOOKBACK` | `50` | Bars of history fed into the analysis. |
| `POLL_SECONDS` | `120` | Background refresh cadence. |
| `GARCH_ALPHA`, `GARCH_BETA` | `0.10`, `0.85` | GARCH(1,1) base parameters (α+β<1 is enforced). |
| `JUMP_INTENSITY`, `JUMP_SIGMA_MULT` | `0.03`, `3.0` | Merton jump knobs. |
| `MC_CLIP` | `0.25` | Per-step return cap for diffusive models. |
| `ZONE_*` | various | Pivot window, cluster ATR, touch ATR, break ATR, max zones per side, zone width. |
| `MIN_*` | various | Trade-setup gates (score, ADX, confidence, MC probability, RR). |
| `BACKTEST_BAND_PCT`, `BACKTEST_COMMISSION`, `BACKTEST_SLIPPAGE` | `0.003`, `0.001`, `0.0005` | Walk-forward thresholds and costs. |
| `HMM_ENABLED`, `HAWKES_ENABLED` | `False`, `False` | Heavy enrichments — disabled by default because each adds 3–10 s per analysis. Available on demand via `/api/market-structure`. |

## Project layout

```
Monte_Carlo_Predict_Stock/
├── main.py                       # uvicorn entry point
├── config.py                     # Config dataclass + env parsing + cross-field validation
├── requirements.txt
├── CHANGES.md                    # May 2026 refactor notes — read first
├── api/
│   ├── __init__.py               # exports `app`
│   ├── server.py                 # routes, WS, poll loop
│   └── models.py                 # Pydantic request models
├── core/
│   ├── __init__.py               # public analyse()
│   ├── fetcher.py                # Alpaca → Polygon → yfinance fallback
│   ├── indicators.py             # full indicator set
│   ├── signal.py                 # composite + regime-aware weighting
│   ├── regime.py                 # Hurst / R² / Donchian composite
│   ├── zones.py                  # demand/supply zone detector
│   ├── montecarlo.py             # seven path models + microstructure
│   ├── trade_setup.py            # entry/SL/TP/RR + zone scenarios + sizing
│   ├── backtest.py               # walk-forward backtest
│   ├── store.py                  # SQLite signal log
│   ├── scanner.py                # breakout/breakdown scanner
│   ├── zone_scanner.py           # zone + EMA scanner
│   ├── volume_profile.py         # POC/VAH/VAL/HVN/LVN derivation
│   ├── hmm_regime.py             # HMM regime detector (optional)
│   ├── hawkes.py                 # Hawkes self-excitation (optional)
│   ├── options_flow.py           # GEX/DEX (optional)
│   ├── sentiment.py              # keyword-based news sentiment
│   └── macro.py                  # FRED-backed macro indicators
├── templates/
│   └── dashboard.html
├── tests/
│   ├── conftest.py
│   ├── test_indicators.py
│   ├── test_signal.py
│   ├── test_zones.py             # added in the May 2026 pass
│   ├── test_montecarlo.py
│   ├── test_backtest.py
│   ├── test_store.py
│   └── test_api.py
└── resource_image/               # screenshots embedded in this README
```

## Testing

`pytest -q` runs the suite. The `tests/conftest.py` fixtures build
deterministic synthetic OHLCV (`synth_df`, `trend_up_df`, `trend_down_df`)
so the regime-, signal-, and MC-level tests are reproducible. The new
`tests/test_zones.py` covers the bug classes fixed in the May 2026 pass
— touch score, body-depth scoring, intraday break filter, and the
`ZoneResult` contract.

The Monte Carlo tests include a performance regression guard
(`test_microstructure_perf_n_sim_2000`) that fails if the vectorised
microstructure path slips above ~5 seconds for 2000 paths × 10 steps on
a typical CPU. If you are profiling, that test is the easiest fingerprint.

## Known limitations

The walk-forward backtest is per-trade and assumes one position at a
time; it does not model portfolio-level constraints (correlation,
margin, concurrent exposure). If you want multi-position simulation you
need to wrap `walk_forward` calls yourself.

The fetcher's "extended hours" path is best-effort — Alpaca returns
extended-hours bars when `EXTENDED_HOURS=true`, but the yfinance
fallback does not, so behaviour during pre/post market can differ
between providers.

`core/sentiment.py` is keyword-based, not ML-based; it is useful as a
coarse gauge but should not be relied on for nuanced text. The
`core/options_flow.py` GEX calculation uses the displayed open interest
without dealer-positioning adjustments, which is the standard caveat
for retail-side GEX dashboards.

The microstructure MC's `final_drift = base_drift + cvd_bias × drift_mult`
zeros the CVD drift bias entirely in mean-reverting regimes
(`drift_mult = 0`). This is deliberate — mean-reverting regimes by
construction discount sustained directional flow — but it is a design
choice rather than an empirical finding, so if you have a different
view on how CVD should propagate in those regimes, that is the line to
edit.

## Disclaimer

This software is for **educational and research purposes only**. It is
not a trading system, not a recommendation engine, and not connected to
any broker. Monte Carlo simulation describes a hypothetical distribution
under modelling assumptions; it does not predict the future. Past
volatility, correlation, and regime patterns do not extend reliably
forward. Paper trade before risking real money, and assume every metric
in this codebase is wrong about something — that is what backtesting is
for.
