"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { runScan, SCAN_INTERVALS, SCAN_WATCHLISTS } from "@/lib/scannerApi";
import { DEFAULT_SCAN, type ScanParams, type ScanReport, type ScanResult } from "@/lib/scannerTypes";
import { SAMPLE_SCAN } from "@/lib/sampleScan";

type Tab = "all" | "breakouts" | "breakdowns" | "neutral";
type SortKey = "score" | "ticker" | "price" | "confidence" | "rsi" | "adx" | "atr_pct" | "hurst" | "prob_up";

export default function ScannerPanel() {
  const router = useRouter();
  const [params, setParams] = useState<ScanParams>(DEFAULT_SCAN);
  const [report, setReport] = useState<ScanReport | null>(null);
  const [status, setStatus] = useState<"idle" | "scanning" | "live" | "offline">("idle");
  const [banner, setBanner] = useState("");
  const [tab, setTab] = useState<Tab>("all");
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  async function scan() {
    setStatus("scanning");
    setBanner("");
    try {
      const r = await runScan(params);
      setReport(r);
      setStatus("live");
    } catch (err) {
      setReport(SAMPLE_SCAN);
      setStatus("offline");
      setBanner(`Couldn't reach the API (${(err as Error).message}). Showing sample scan — start FastAPI and scan again.`);
    }
  }

  async function loadTicker(t: string) {
    try {
      await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: t }),
      });
    } catch {
      /* navigate anyway */
    }
    router.push("/chart");
  }

  const base = useMemo<ScanResult[]>(() => {
    if (!report) return [];
    return tab === "breakouts" ? report.breakouts
      : tab === "breakdowns" ? report.breakdowns
      : tab === "neutral" ? report.neutral
      : report.all;
  }, [report, tab]);

  const rows = useMemo(() => {
    const copy = [...base];
    copy.sort((a, b) => {
      const x = a[sortKey];
      const y = b[sortKey];
      if (typeof x === "string" && typeof y === "string") return sortDir * x.localeCompare(y);
      return sortDir * ((Number(x) || 0) - (Number(y) || 0));
    });
    return copy;
  }, [base, sortKey, sortDir]);

  function th(key: SortKey, label: string, align: "l" | "r" = "r") {
    return (
      <th
        onClick={() => {
          if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1));
          else { setSortKey(key); setSortDir(key === "ticker" ? 1 : -1); }
        }}
        className={"sticky top-0 cursor-pointer select-none whitespace-nowrap border-b border-line bg-[#0d1217] px-2.5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-dim " + (align === "l" ? "text-left" : "text-right")}
      >
        {label}
        {sortKey === key && <span className="ml-1 text-[9px] text-blue">{sortDir < 0 ? "▼" : "▲"}</span>}
      </th>
    );
  }

  return (
    <div className="pt-3">
      <div className="mb-3 flex flex-wrap items-center gap-2.5">
        <h1 className="m-0 text-[17px] font-semibold tracking-tight">Scanner</h1>
        <div className="flex-1" />
        <span className="flex items-center gap-1.5 text-[12px] text-muted">
          <span className={"inline-block h-[7px] w-[7px] rounded-full " + (status === "live" ? "bg-up shadow-[0_0_7px_#3fb950]" : status === "scanning" ? "bg-gold" : "bg-dim")} />
          {status}
        </span>
        <select className="rounded-md border border-line bg-[#161c23] px-2.5 py-1.5 text-[12px]" value={params.watchlist} onChange={(e) => setParams((p) => ({ ...p, watchlist: e.target.value }))}>
          {SCAN_WATCHLISTS.map((w) => <option key={w.value} value={w.value}>{w.label}</option>)}
        </select>
        <select className="rounded-md border border-line bg-[#161c23] px-2.5 py-1.5 text-[12px]" value={params.interval} onChange={(e) => setParams((p) => ({ ...p, interval: e.target.value }))}>
          {SCAN_INTERVALS.map((i) => <option key={i} value={i}>{i}</option>)}
        </select>
        <label className="flex items-center gap-1.5 text-[12px] text-muted">
          min score
          <input type="number" step={0.05} min={0} max={1} value={params.min_score_abs}
            onChange={(e) => setParams((p) => ({ ...p, min_score_abs: Number(e.target.value) }))}
            className="w-16 rounded-md border border-line bg-[#161c23] px-2 py-1.5 text-[12px]" />
        </label>
        <button onClick={scan} className="rounded-md border border-[#1f6feb] bg-[#1f6feb] px-3.5 py-1.5 text-[12px] font-semibold text-white hover:bg-[#388bfd]">
          {status === "scanning" ? "Scanning…" : "Scan"}
        </button>
      </div>

      {banner && <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">{banner}</div>}

      {report && (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-4 text-[12px] text-muted">
            <span><b className="text-ink">{report.scanned}</b> scanned</span>
            <span className="text-up"><b>{report.breakouts.length}</b> breakouts</span>
            <span className="text-down"><b>{report.breakdowns.length}</b> breakdowns</span>
            <span><b className="text-ink">{report.succeeded}</b> ok</span>
            <span><b className="text-ink">{report.failed}</b> failed</span>
            <span className="text-dim">{(report.elapsed_ms / 1000).toFixed(1)}s</span>
          </div>

          <div className="mb-2 flex gap-1.5">
            {([
              ["all", `All (${report.all.length})`],
              ["breakouts", `Breakouts (${report.breakouts.length})`],
              ["breakdowns", `Breakdowns (${report.breakdowns.length})`],
              ["neutral", `Neutral (${report.neutral.length})`],
            ] as [Tab, string][]).map(([t, label]) => (
              <button key={t} onClick={() => setTab(t)}
                className={"rounded-md border px-3 py-1 text-[12px] transition-colors " + (tab === t ? "border-[#2c3a47] bg-[#161c23] text-ink" : "border-line text-dim")}>
                {label}
              </button>
            ))}
          </div>

          <div className="overflow-x-auto rounded-xl border border-line bg-panel">
            <table className="w-full tnum border-collapse text-[13px]">
              <thead>
                <tr>
                  {th("ticker", "Ticker", "l")}
                  {th("price", "Price")}
                  {th("score", "Score")}
                  <th className="sticky top-0 whitespace-nowrap border-b border-line bg-[#0d1217] px-2.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-dim">Direction</th>
                  <th className="sticky top-0 whitespace-nowrap border-b border-line bg-[#0d1217] px-2.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-dim">Regime</th>
                  <th className="sticky top-0 whitespace-nowrap border-b border-line bg-[#0d1217] px-2.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-dim">Signal</th>
                  {th("confidence", "Conf")}
                  {th("prob_up", "P(up)")}
                  {th("rsi", "RSI")}
                  {th("adx", "ADX")}
                  {th("atr_pct", "ATR%")}
                  {th("hurst", "Hurst")}
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr><td colSpan={12} className="p-10 text-center text-muted">No results in this view.</td></tr>
                ) : (
                  rows.map((r, i) => <Row key={i} r={r} onLoad={loadTicker} />)
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!report && (
        <div className="rounded-xl border border-line bg-panel p-10 text-center text-muted">
          Pick a watchlist and press <b>Scan</b>. A full scan fetches data for every ticker and can take a few seconds.
        </div>
      )}
    </div>
  );
}

function Row({ r, onLoad }: { r: ScanResult; onLoad: (t: string) => void }) {
  const pos = r.score >= 0;
  const w = Math.min(50, Math.abs(r.score) * 50);
  return (
    <tr className="border-b border-[#141a20] odd:bg-rowalt hover:bg-[#13202b]">
      <td className="px-2.5 py-[7px]">
        <button onClick={() => onLoad(r.ticker)} className="font-bold text-blue hover:underline" title={`Load ${r.ticker}`}>
          {r.ticker}
        </button>
      </td>
      <td className="px-2.5 py-[7px] text-right text-[#c9d4df]">${r.price.toFixed(2)}</td>
      <td className="px-2.5 py-[7px]">
        <div className="flex items-center justify-end gap-2">
          <div className="relative h-2.5 w-[84px] rounded bg-[#1c242c]">
            <span className="absolute top-0 h-full w-px bg-dim" style={{ left: "50%" }} />
            <span
              className="absolute top-0 h-full rounded"
              style={{ background: pos ? "#3fb950" : "#f0556d", left: pos ? "50%" : `${50 - w}%`, width: `${w}%` }}
            />
          </div>
          <span className={"w-10 text-right text-[11px] font-semibold tnum " + (pos ? "text-up" : "text-down")}>
            {(r.score > 0 ? "+" : "") + r.score.toFixed(2)}
          </span>
        </div>
      </td>
      <td className={"px-2.5 py-[7px] text-left font-medium " + dirColor(r.direction)}>{pretty(r.direction)}</td>
      <td className="px-2.5 py-[7px] text-left text-muted">{pretty(r.regime)}</td>
      <td className="px-2.5 py-[7px] text-left">{r.signal_label}</td>
      <td className="px-2.5 py-[7px] text-right text-muted">{(r.confidence * 100).toFixed(0)}%</td>
      <td className="px-2.5 py-[7px] text-right">{r.prob_up != null ? <span className={r.prob_up >= 50 ? "text-up" : "text-down"}>{r.prob_up.toFixed(0)}%</span> : <span className="text-dim">—</span>}</td>
      <td className={"px-2.5 py-[7px] text-right " + (r.rsi >= 70 ? "text-down" : r.rsi <= 30 ? "text-up" : "text-[#c9d4df]")}>{r.rsi.toFixed(0)}</td>
      <td className="px-2.5 py-[7px] text-right text-[#c9d4df]">{r.adx.toFixed(0)}</td>
      <td className="px-2.5 py-[7px] text-right text-muted">{r.atr_pct.toFixed(1)}</td>
      <td className="px-2.5 py-[7px] text-right text-muted">{r.hurst.toFixed(2)}</td>
    </tr>
  );
}

function dirColor(d: string): string {
  if (d.includes("up") || d === "bullish") return "text-up";
  if (d.includes("down") || d === "breakdown" || d === "bearish") return "text-down";
  return "text-muted";
}
function pretty(s: string): string {
  return (s || "—").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
