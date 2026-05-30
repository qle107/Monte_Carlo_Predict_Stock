import type { AnalysisResult } from "./analysisTypes";

// GET /api/signal triggers a fresh analysis with the server's current config.
export async function fetchSignal(): Promise<AnalysisResult> {
  const res = await fetch("/api/signal", { cache: "no-store" });
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
  const data = (await res.json()) as AnalysisResult;
  if (data.error) throw new Error(data.error);
  return data;
}

// POST /api/config with an arbitrary patch, then re-run analysis.
export async function setConfigAndAnalyze(patch: Record<string, unknown>): Promise<AnalysisResult> {
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
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
  // Config change kicks the poll loop; fetch a fresh result.
  return fetchSignal();
}

export function setTickerAndAnalyze(ticker: string): Promise<AnalysisResult> {
  return setConfigAndAnalyze({ ticker: ticker.toUpperCase().trim() });
}

// Mirrors VALID_INTERVALS in config.py.
export const INTERVALS = ["1m", "2m", "5m", "15m", "30m", "1h", "4h", "1d"];

// Mirrors VALID_MC_MODELS in config.py.
export const MC_MODELS: { value: string; label: string }[] = [
  { value: "gaussian", label: "Gaussian GBM" },
  { value: "student_t", label: "Student-t" },
  { value: "garch", label: "GARCH(1,1)" },
  { value: "bootstrap", label: "Bootstrap" },
  { value: "jump", label: "Jump-diffusion" },
  { value: "ensemble", label: "Ensemble" },
  { value: "microstructure", label: "Microstructure" },
];
