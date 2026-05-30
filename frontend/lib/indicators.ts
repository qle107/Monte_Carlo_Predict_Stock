import type { Candle } from "./analysisTypes";

// Client-side overlay math computed from the candle series. Mirrors the backend
// indicator intent for visualization only (the authoritative values come from
// /api/signal). NaN is used for the warm-up region so lines start cleanly.

export function ema(values: number[], period: number): number[] {
  // Seeded EMA computed from the first bar so the line is visible even when the
  // series is shorter than `period` (e.g. EMA200 on ~200 candles). Early values
  // are an approximation — these overlays are for display only.
  const out = new Array(values.length).fill(NaN);
  if (values.length === 0) return out;
  const k = 2 / (period + 1);
  let prev = values[0];
  out[0] = prev;
  for (let i = 1; i < values.length; i++) {
    prev = values[i] * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}

export function sma(values: number[], period: number): number[] {
  const out = new Array(values.length).fill(NaN);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}

export interface Bollinger {
  mid: number[];
  upper: number[];
  lower: number[];
}

export function bollinger(values: number[], period = 20, k = 2): Bollinger {
  const mid = sma(values, period);
  const upper = new Array(values.length).fill(NaN);
  const lower = new Array(values.length).fill(NaN);
  for (let i = period - 1; i < values.length; i++) {
    let s = 0;
    for (let j = i - period + 1; j <= i; j++) s += (values[j] - mid[i]) ** 2;
    const sd = Math.sqrt(s / period);
    upper[i] = mid[i] + k * sd;
    lower[i] = mid[i] - k * sd;
  }
  return { mid, upper, lower };
}

// Cumulative VWAP over the visible window (typical price weighted by volume).
export function vwap(candles: Candle[]): number[] {
  const out = new Array(candles.length).fill(NaN);
  let cumPV = 0;
  let cumV = 0;
  for (let i = 0; i < candles.length; i++) {
    const tp = (candles[i].h + candles[i].l + candles[i].c) / 3;
    cumPV += tp * candles[i].v;
    cumV += candles[i].v;
    out[i] = cumV > 0 ? cumPV / cumV : NaN;
  }
  return out;
}
