import type { ScanReport, ScanResult } from "./scannerTypes";

function mk(p: Partial<ScanResult>): ScanResult {
  return {
    ticker: "",
    price: 0,
    score: 0,
    direction: "neutral",
    strength: "weak",
    regime: "range_bound",
    signal_label: "neutral",
    confidence: 0.3,
    prob_up: null,
    prob_down: null,
    rsi: 50,
    adx: 18,
    macd_hist: 0,
    bb_position: 0,
    obv_slope: 0,
    atr_pct: 2,
    vol_regime: "normal",
    donchian_pos: 0,
    hurst: 0.5,
    trend_score: 0,
    potential_up: 33,
    potential_down: 33,
    rsi_divergence: 0,
    ema200_dist: 0,
    price_vs_52w: 0,
    reasoning: "",
    elapsed_ms: 120,
    trade_setup: { valid: false, side: "none" },
    ...p,
  };
}

const rows: ScanResult[] = [
  mk({ ticker: "NVDA", price: 1180.4, score: 0.78, direction: "breakout_up", strength: "strong", regime: "breakout_up", signal_label: "strong buy", confidence: 0.72, prob_up: 64.2, prob_down: 24.1, rsi: 67, adx: 34, atr_pct: 3.1, trend_score: 0.8, hurst: 0.63, reasoning: "breakout above donchian high; obv rising; adx strong." }),
  mk({ ticker: "PLTR", price: 62.1, score: 0.46, direction: "trending_up", strength: "moderate", regime: "weak_uptrend", signal_label: "bullish", confidence: 0.61, prob_up: 58.4, prob_down: 29.4, rsi: 59, adx: 24, atr_pct: 2.4, trend_score: 0.46, hurst: 0.57, reasoning: "ema9>ema21, macd rising, above vwap." }),
  mk({ ticker: "META", price: 632.6, score: 0.33, direction: "trending_up", strength: "moderate", regime: "strong_uptrend", signal_label: "bullish", confidence: 0.55, prob_up: 55.1, prob_down: 31.0, rsi: 61, adx: 28, atr_pct: 2.0, trend_score: 0.6, hurst: 0.59, reasoning: "higher highs, trend intact." }),
  mk({ ticker: "AAPL", price: 214.7, score: 0.12, direction: "neutral", strength: "weak", regime: "range_bound", signal_label: "neutral", confidence: 0.34, prob_up: null, prob_down: null, rsi: 52, adx: 16, atr_pct: 1.4, trend_score: 0.1, hurst: 0.5, reasoning: "range-bound near mid-band." }),
  mk({ ticker: "INTC", price: 114.6, score: -0.41, direction: "trending_down", strength: "moderate", regime: "weak_downtrend", signal_label: "bearish", confidence: 0.58, prob_up: 33.0, prob_down: 55.4, rsi: 41, adx: 23, atr_pct: 2.8, trend_score: -0.44, hurst: 0.56, reasoning: "ema9<ema21, obv falling, below vwap." }),
  mk({ ticker: "BA", price: 178.2, score: -0.69, direction: "breakdown", strength: "strong", regime: "breakout_down", signal_label: "strong sell", confidence: 0.66, prob_up: 26.0, prob_down: 62.3, rsi: 33, adx: 31, atr_pct: 3.4, trend_score: -0.72, hurst: 0.61, reasoning: "broke donchian low; momentum negative." }),
];

export const SAMPLE_SCAN: ScanReport = {
  scanned: 50,
  succeeded: 48,
  failed: 2,
  elapsed_ms: 8420,
  interval: "1d",
  lookback: 60,
  breakouts: rows.filter((r) => r.score > 0.2),
  breakdowns: rows.filter((r) => r.score < -0.2),
  neutral: rows.filter((r) => Math.abs(r.score) <= 0.2),
  all: [...rows].sort((a, b) => b.score - a.score),
  errors: [{ ticker: "XYZ", error: "no data" }],
};
