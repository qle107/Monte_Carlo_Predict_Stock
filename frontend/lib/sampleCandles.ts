import type { Candle } from "./analysisTypes";

// Deterministic synthetic OHLCV series for the offline chart fallback.
export function sampleCandles(n = 120, start = 55): Candle[] {
  let seed = 1337;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const out: Candle[] = [];
  let price = start;
  const now = Date.now();
  const stepMs = 15 * 60 * 1000; // 15m candles
  for (let i = 0; i < n; i++) {
    const drift = 0.04 + Math.sin(i / 14) * 0.18;
    const o = price;
    const ch = drift + (rnd() - 0.5) * 1.4;
    const c = Math.max(1, o + ch);
    const h = Math.max(o, c) + rnd() * 0.7;
    const l = Math.min(o, c) - rnd() * 0.7;
    const v = Math.round(200000 + rnd() * 800000 + (Math.abs(ch) > 1 ? 600000 : 0));
    out.push({
      t: new Date(now - (n - i) * stepMs).toISOString(),
      o: +o.toFixed(2),
      h: +h.toFixed(2),
      l: +l.toFixed(2),
      c: +c.toFixed(2),
      v,
    });
    price = c;
  }
  return out;
}
