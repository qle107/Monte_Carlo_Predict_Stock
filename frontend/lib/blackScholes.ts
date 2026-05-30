import type { UnusualOption } from "./types";

// Standard normal CDF (Abramowitz & Stegun 7.1.26 approximation).
export function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}

// Black–Scholes delta from a chain snapshot. r = 5% risk-free.
export function bsDelta(o: UnusualOption, r = 0.05): number {
  const sigma = (o.implied_vol || 0) / 100;
  const T = Math.max(o.days_to_expiry || 0, 0.5) / 365;
  const S = o.spot || 0;
  const K = o.strike || 0;
  if (!(S > 0 && K > 0 && sigma > 0 && T > 0)) {
    return o.option_type === "put" ? -0.5 : 0.5;
  }
  const d1 =
    (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const nd1 = normCdf(d1);
  return o.option_type === "put" ? nd1 - 1 : nd1;
}
