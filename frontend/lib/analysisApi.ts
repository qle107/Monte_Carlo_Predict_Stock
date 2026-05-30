import type { AnalysisResult } from "./analysisTypes";

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
  return fetchSignal();
}

export function setTickerAndAnalyze(ticker: string): Promise<AnalysisResult> {
  return setConfigAndAnalyze({ ticker: ticker.toUpperCase().trim() });
}

export const INTERVALS = ["1m", "2m", "5m", "15m", "30m", "1h", "4h", "1d"];

export const MC_MODELS: { value: string; label: string }[] = [
  { value: "gaussian", label: "gaussian" },
  { value: "student_t", label: "student-t" },
  { value: "garch", label: "garch" },
  { value: "bootstrap", label: "bootstrap" },
  { value: "jump", label: "jump" },
  { value: "ensemble", label: "ensemble" },
  { value: "microstructure", label: "microstructure" },
];
