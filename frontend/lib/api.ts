import type { FlowFilters, ScanResponse } from "./types";

// Calls the FastAPI /api/options/unusual endpoint (proxied via next.config.mjs).
export async function fetchFlow(f: FlowFilters, signal?: AbortSignal): Promise<ScanResponse> {
  const params = new URLSearchParams({
    top_n: String(f.top_n),
    max_dte: String(f.max_dte),
    vol_oi_threshold: String(f.vol_oi_threshold),
    min_premium: "0",
    min_sweep_premium: String(f.min_sweep_premium),
    min_block_premium: String(f.min_block_premium),
    exclude_bid_side: String(f.exclude_bid_side),
    exclude_high_volume_etfs: String(f.exclude_high_volume_etfs),
  });
  if (f.watchlist) params.set("watchlist", f.watchlist);

  const res = await fetch(`/api/options/unusual?${params.toString()}`, {
    cache: "no-store",
    signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const WATCHLISTS: { value: string; label: string }[] = [
  { value: "momentum", label: "momentum" },
  { value: "tech", label: "tech" },
  { value: "biotech", label: "biotech" },
  { value: "energy", label: "energy" },
  { value: "defense", label: "defense" },
  { value: "crypto", label: "crypto" },
  { value: "financials", label: "financials" },
  { value: "ev", label: "ev" },
  { value: "", label: "all optionable (slow)" },
];
