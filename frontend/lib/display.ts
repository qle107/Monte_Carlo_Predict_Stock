const ACRONYMS = new Set([
  "mc", "gex", "rsi", "adx", "macd", "ema", "vwap", "obv", "atr", "bb", "oi",
  "iv", "etf", "cvar", "rr", "hh", "hl", "lh", "ll", "garch", "dte", "sma",
  "cp", "us", "se", "ev", "atm", "itm", "otm", "pc", "ms", "chg", "conf", "exp", "gex",
]);

const STATUS_LABELS: Record<string, string> = {
  idle: "Idle",
  live: "Live",
  loading: "Loading",
  scanning: "Scanning",
  offline: "Offline",
};

function titleWord(w: string): string {
  const m = w.match(/^([a-zA-Z]+)(\d*)$/);
  if (m) {
    const alpha = m[1];
    const digits = m[2];
    if (ACRONYMS.has(alpha.toLowerCase())) return alpha.toUpperCase() + digits;
    return alpha.charAt(0).toUpperCase() + alpha.slice(1).toLowerCase() + digits;
  }
  return w.charAt(0).toUpperCase() + w.slice(1);
}

/** Title-case API / config strings (acronyms uppercased). */
export function label(s: string): string {
  const out = (s || "-")
    .replace(/%/g, " %")
    .replace(/_/g, " ")
    .replace(/\//g, " / ")
    .split(/\s+/)
    .filter(Boolean)
    .map(titleWord)
    .join(" ")
    .replace(/ %/g, "%");
  return out || "-";
}

/** Sentence-case prose from the API. */
export function displayText(s: string): string {
  const t = label((s || "-").replace(/_/g, " "));
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/** Human-readable connection / scan status. */
export function statusLabel(s: string): string {
  return STATUS_LABELS[s.toLowerCase()] ?? label(s);
}
