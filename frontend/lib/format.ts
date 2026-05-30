// Display formatters for the flow feed.

export function fmtValue(v: number): string {
  if (v >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return "$" + (v / 1e3).toFixed(1) + "K";
  return "$" + (v || 0).toFixed(0);
}

export function fmtExpiry(s: string): string {
  // "2026-07-17" -> "26-07-17"
  if (!s) return "—";
  const p = s.split("-");
  return p.length === 3 ? `${p[0].slice(2)}-${p[1]}-${p[2]}` : s;
}

export function fmtStrike(k: number): string {
  return k % 1 ? k.toFixed(2) : k.toFixed(0);
}

export function fmtPeakReturn(pc: number | null | undefined): string {
  if (pc === null || pc === undefined) return "N/A";
  const dp = pc !== 0 && Math.abs(pc) < 1 ? 2 : 0;
  return (pc > 0 ? "+" : "") + pc.toFixed(dp) + "%";
}

export function hms(iso?: string): string {
  const d = iso ? new Date(iso) : new Date();
  return d.toTimeString().slice(0, 8);
}

// SigScore bar fill color by score.
export function sigColor(v: number): string {
  if (v >= 0.65) return "#84cc16";
  if (v >= 0.55) return "#eab308";
  return "#f59e0b";
}

export function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
