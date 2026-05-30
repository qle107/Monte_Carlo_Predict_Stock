// Mirrors the dict returned by api/analysis.py _run_analysis() / GET /api/signal.

export interface Signal {
  composite: number; // -1..1
  confidence: number; // 0..1
  drift_bias: number;
  base_drift: number;
  signal_adj: number;
  vol_adj: number;
  label: string;
  reasoning: string;
  gap_warning: string;
  sub_scores: Record<string, number>;
  weights: Record<string, number>;
}

export interface Regime {
  regime: string;
  verdict: string;
  potential_up: number; // 0..100
  potential_down: number;
  potential_flat: number;
  trend_score: number; // -1..1
  range_score: number; // 0..1
  breakout_up: boolean;
  breakout_down: boolean;
  hurst: number;
  donchian_pos: number; // -1..1
  donchian_high: number;
  donchian_low: number;
  hh_count: number;
  hl_count: number;
  lh_count: number;
  ll_count: number;
  range_compression: number;
  components: Record<string, number>;
}

export interface MC {
  prob_up: number;
  prob_flat: number;
  prob_down: number;
  median_price: number;
  p10_price: number;
  p90_price: number;
  p25_price: number;
  p75_price: number;
  expected_price: number;
  expected_return: number; // %
  cvar_5: number; // %
  upper_band: number[]; // p75 path
  lower_band: number[]; // p25 path
  p90_band: number[];
  p10_band: number[];
  paths: number[][];
  median_path: number[];
  model: string;
  prob_up_se: number;
  prob_down_se: number;
  cvar_5_se: number;
  ms_regime?: string | null;
}

export interface TradeSetup {
  valid: boolean;
  side: string;
  reason?: string;
  entry?: number;
  stop?: number;
  target?: number;
  rr?: number;
  [k: string]: unknown;
}

export interface Indicators {
  rsi: number;
  adx: number;
  macd_hist: number;
  atr_pct: number;
  bb_position: number;
  obv_slope: number;
  ema200_dist: number;
  vol_regime: string;
  [k: string]: unknown;
}

export interface Candle {
  t: string; // ISO timestamp
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export interface AnalysisResult {
  ticker: string;
  interval: string;
  mc_model: string;
  current_price: number;
  updated_at: string;
  signal: Signal;
  regime: Regime;
  mc: MC;
  indicators: Indicators;
  trade_setup: TradeSetup;
  candles: Candle[];
  warnings: string[];
  error?: string;
}
