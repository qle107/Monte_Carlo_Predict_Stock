import type { OptionsFlow } from "./gexTypes";

// GET /api/options/gex?ticker=XXX
export async function fetchGex(ticker: string): Promise<OptionsFlow> {
  const res = await fetch(`/api/options/gex?ticker=${encodeURIComponent(ticker.toUpperCase().trim())}`, {
    cache: "no-store",
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
  const data = (await res.json()) as OptionsFlow;
  if (data.error) throw new Error(data.error);
  return data;
}
