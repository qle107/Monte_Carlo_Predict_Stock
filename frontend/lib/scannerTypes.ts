// Mirrors core/scanner.py ScanResult + scan_tickers() report.

export interface ScanResult {
  ticker: string;
  price: number;
  score: number; // signed -1..1
  direction: string;
  strength: string; // strong | moderate | weak
  regime: string;
  signal_label: string;
  confidence: number; // 0..1
  prob_up: number | null;
  prob_down: number | null;
  rsi: number;
  adx: number;
  macd_hist: number;
  bb_position: number;
  obv_slope: number;
  atr_pct: number;
  vol_regime: string;
  donchian_pos: number;
  hurst: number;
  trend_score: number;
  potential_up: number;
  potential_down: number;
  rsi_divergence: number;
  ema200_dist: number;
  price_vs_52w: number;
  reasoning: string;
  error?: string | null;
  elapsed_ms: number;
  trade_setup?: { valid: boolean; side: string; reason?: string; [k: string]: unknown };
}

export interface ScanReport {
  scanned: number;
  succeeded: number;
  failed: number;
  elapsed_ms: number;
  interval: string;
  lookback: number;
  breakouts: ScanResult[];
  breakdowns: ScanResult[];
  neutral: ScanResult[];
  all: ScanResult[];
  errors: { ticker: string; error: string }[];
}

export interface ScanParams {
  watchlist: string;
  interval: string;
  lookback: number;
  min_score_abs: number;
}

export const DEFAULT_SCAN: ScanParams = {
  watchlist: "sp500_large",
  interval: "1d",
  lookback: 60,
  min_score_abs: 0.2,
};
