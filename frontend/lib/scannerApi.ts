import type { ScanParams, ScanReport } from "./scannerTypes";

export async function runScan(p: ScanParams): Promise<ScanReport> {
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      watchlist: p.watchlist,
      interval: p.interval,
      lookback: p.lookback,
      min_score_abs: p.min_score_abs,
    }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const b = await res.json();
      detail = b.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const SCAN_WATCHLISTS: { value: string; label: string }[] = [
  { value: "sp500_large", label: "s&p 500 large" },
  { value: "momentum", label: "momentum" },
  { value: "tech", label: "tech" },
  { value: "biotech", label: "biotech" },
  { value: "energy", label: "energy" },
  { value: "defense", label: "defense" },
  { value: "crypto", label: "crypto" },
  { value: "financials", label: "financials" },
  { value: "ev", label: "ev" },
  { value: "etfs", label: "etfs" },
];

export const SCAN_INTERVALS = ["15m", "30m", "1h", "4h", "1d"];
