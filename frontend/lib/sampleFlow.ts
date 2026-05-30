import type { ScanResponse, UnusualOption } from "./types";

// Shown when the FastAPI backend is unreachable, so the layout is visible
// without a running server. Mirrors the reference flow screenshot.
function mk(p: Partial<UnusualOption>): UnusualOption {
  return {
    ticker: "",
    expiry: "2026-06-18",
    strike: 0,
    option_type: "call",
    volume: 0,
    open_interest: 0,
    vol_oi_ratio: 0,
    implied_vol: 30,
    avg_chain_iv: 30,
    in_the_money: false,
    percent_change: 0,
    sector: "Other",
    premium_per_contract: 0,
    total_premium: 0,
    unusual_score: 0.5,
    flags: [],
    sentiment: "bullish",
    spot: 0,
    days_to_expiry: 20,
    trade_style: "block",
    exec_side: "ask",
    ...p,
  };
}

export const SAMPLE_FLOW: ScanResponse = {
  scanned_at: new Date().toISOString(),
  summary: {
    tickers_scanned: 48,
    tickers_with_hits: 9,
    total_hits: 16,
    bullish_count: 12,
    bearish_count: 4,
    mixed_count: 0,
    sweep_count: 7,
    block_count: 9,
    delisted_count: 0,
    delisted_tickers: [],
    excluded_etf_count: 31,
    filters: {
      min_sweep_premium: 50000,
      min_block_premium: 100000,
      exclude_bid_side: true,
      exclude_high_volume_etfs: true,
    },
  },
  hits: [
    mk({ ticker: "HYG", spot: 80.32, strike: 55, option_type: "call", expiry: "2026-07-17", trade_style: "block", premium_per_contract: 2640, total_premium: 792000, volume: 300, open_interest: 575, unusual_score: 0.53, implied_vol: 22, days_to_expiry: 48 }),
    mk({ ticker: "MSFT", spot: 449.53, strike: 500, option_type: "call", expiry: "2028-01-21", trade_style: "block", premium_per_contract: 6415, total_premium: 737700, volume: 115, open_interest: 869, unusual_score: 0.45, implied_vol: 25, days_to_expiry: 601 }),
    mk({ ticker: "HOOD", spot: 94.35, strike: 90, option_type: "call", expiry: "2027-06-17", trade_style: "block", premium_per_contract: 2920, total_premium: 292000, volume: 100, open_interest: 289, unusual_score: 0.4, implied_vol: 55, days_to_expiry: 383 }),
    mk({ ticker: "MSFT", spot: 450.05, strike: 440, option_type: "call", expiry: "2026-06-26", trade_style: "sweep", premium_per_contract: 2110, total_premium: 213100, volume: 101, open_interest: 13851, unusual_score: 0.54, implied_vol: 24, days_to_expiry: 27, percent_change: 0 }),
    mk({ ticker: "JPM", spot: 299.45, strike: 300, option_type: "call", expiry: "2026-06-05", trade_style: "block", premium_per_contract: 380, total_premium: 190000, volume: 500, open_interest: 3711, unusual_score: 0.56, implied_vol: 18, days_to_expiry: 6 }),
    mk({ ticker: "INTC", spot: 114.67, strike: 124, option_type: "put", expiry: "2026-06-05", trade_style: "block", premium_per_contract: 1160, total_premium: 290000, volume: 250, open_interest: 1711, unusual_score: 0.51, implied_vol: 42, days_to_expiry: 6, sentiment: "bearish" }),
    mk({ ticker: "INTC", spot: 114.7, strike: 100, option_type: "put", expiry: "2026-06-05", trade_style: "sweep", premium_per_contract: 100.7, total_premium: 86400, volume: 858, open_interest: 3506, unusual_score: 0.69, implied_vol: 40, days_to_expiry: 6, percent_change: 0.3, sentiment: "bearish" }),
    mk({ ticker: "GLW", spot: 181.43, strike: 175, option_type: "call", expiry: "2027-03-19", trade_style: "sweep", premium_per_contract: 5109.6, total_premium: 766400, volume: 150, open_interest: 160, unusual_score: 0.66, implied_vol: 30, days_to_expiry: 293, percent_change: 0.05 }),
    mk({ ticker: "INTC", spot: 114.65, strike: 110, option_type: "put", expiry: "2026-06-05", trade_style: "block", premium_per_contract: 325, total_premium: 154100, volume: 474, open_interest: 8065, unusual_score: 0.6, implied_vol: 39, days_to_expiry: 6, sentiment: "bearish" }),
    mk({ ticker: "INTC", spot: 114.65, strike: 110, option_type: "call", expiry: "2026-06-05", trade_style: "sweep", premium_per_contract: 830, total_premium: 415000, volume: 500, open_interest: 2672, unusual_score: 0.69, implied_vol: 38, days_to_expiry: 6 }),
    mk({ ticker: "WSM", spot: 203.68, strike: 200, option_type: "put", expiry: "2026-06-18", trade_style: "block", premium_per_contract: 565, total_premium: 256500, volume: 454, open_interest: 622, unusual_score: 0.54, implied_vol: 36, days_to_expiry: 19, sentiment: "bearish" }),
    mk({ ticker: "MU", spot: 972.35, strike: 210, option_type: "call", expiry: "2026-06-18", trade_style: "block", premium_per_contract: 76205, total_premium: 15240000, volume: 200, open_interest: 200, unusual_score: 0.43, implied_vol: 50, days_to_expiry: 19 }),
    mk({ ticker: "MSFT", spot: 449.62, strike: 500, option_type: "call", expiry: "2026-07-17", trade_style: "sweep", premium_per_contract: 545, total_premium: 58900, volume: 108, open_interest: 16655, unusual_score: 0.57, implied_vol: 24, days_to_expiry: 48, percent_change: 0.92 }),
    mk({ ticker: "META", spot: 632.63, strike: 760, option_type: "call", expiry: "2028-12-15", trade_style: "sweep", premium_per_contract: 13550, total_premium: 1355000, volume: 100, open_interest: 105, unusual_score: 0.57, implied_vol: 33, days_to_expiry: 929 }),
    mk({ ticker: "MSFT", spot: 449.85, strike: 500, option_type: "call", expiry: "2026-07-17", trade_style: "sweep", premium_per_contract: 545, total_premium: 50100, volume: 92, open_interest: 16547, unusual_score: 0.57, implied_vol: 24, days_to_expiry: 48, percent_change: 0.92 }),
    mk({ ticker: "META", spot: 632.61, strike: 635, option_type: "call", expiry: "2026-06-12", trade_style: "block", premium_per_contract: 1490, total_premium: 166900, volume: 112, open_interest: 399, unusual_score: 0.44, implied_vol: 31, days_to_expiry: 13, percent_change: 0.2 }),
  ],
};
