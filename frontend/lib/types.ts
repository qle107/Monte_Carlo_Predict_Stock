export type OptionType = "call" | "put";
export type TradeStyle = "sweep" | "block";
export type ExecSide = "ask" | "bid" | "mid";
export type Sentiment = "bullish" | "bearish" | "mixed";

export interface UnusualOption {
  ticker: string;
  expiry: string; // YYYY-MM-DD
  strike: number;
  option_type: OptionType;
  volume: number;
  open_interest: number;
  vol_oi_ratio: number;
  implied_vol: number; // percent
  avg_chain_iv: number; // percent
  in_the_money: boolean;
  percent_change: number;
  sector: string;
  premium_per_contract: number; // mid * 100 (dollars)
  total_premium: number;
  unusual_score: number; // 0..1
  flags: string[];
  sentiment: Sentiment;
  spot: number;
  days_to_expiry: number;
  trade_style: TradeStyle;
  exec_side: ExecSide;
}

export interface ScanSummary {
  tickers_scanned: number;
  tickers_with_hits: number;
  total_hits: number;
  bullish_count: number;
  bearish_count: number;
  mixed_count: number;
  sweep_count: number;
  block_count: number;
  delisted_count: number;
  delisted_tickers: string[];
  excluded_etf_count: number;
  filters: {
    min_sweep_premium: number;
    min_block_premium: number;
    exclude_bid_side: boolean;
    exclude_high_volume_etfs: boolean;
  };
}

export interface ScanResponse {
  hits: UnusualOption[];
  summary: ScanSummary;
  scanned_at: string; // ISO-8601
}

export interface FlowRow extends UnusualOption {
  _delta: number;
  _time: string; // HH:MM:SS
}

export interface FlowFilters {
  watchlist: string;
  min_sweep_premium: number;
  min_block_premium: number;
  exclude_bid_side: boolean;
  exclude_high_volume_etfs: boolean;
  max_dte: number;
  vol_oi_threshold: number;
  top_n: number;
}

export const DEFAULT_FILTERS: FlowFilters = {
  watchlist: "momentum",
  min_sweep_premium: 50_000,
  min_block_premium: 100_000,
  exclude_bid_side: true,
  exclude_high_volume_etfs: true,
  max_dte: 60,
  vol_oi_threshold: 3,
  top_n: 100,
};
