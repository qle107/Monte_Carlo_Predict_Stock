"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { displayText, label } from "@/lib/display";
import { runScan, SCAN_INTERVALS, SCAN_WATCHLISTS } from "@/lib/scannerApi";
import { DEFAULT_SCAN, type ScanParams, type ScanReport, type ScanResult } from "@/lib/scannerTypes";
import { SAMPLE_SCAN } from "@/lib/sampleScan";
import Select from "./Select";
import StatusBadge from "./StatusBadge";

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
      setBanner(`Could not reach API (${(err as Error).message}). Showing sample data.`);
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

  function th(key: SortKey, lbl: string, align: "l" | "r" = "r") {
    return (
      <th
        onClick={() => {
          if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1));
          else { setSortKey(key); setSortDir(key === "ticker" ? 1 : -1); }
        }}
        className={"sticky top-0 z-10 cursor-pointer select-none whitespace-nowrap border-b border-line bg-[#0b1015] px-2.5 py-2.5 text-[10.5px] font-semibold transition-colors hover:text-muted " + (sortKey === key ? "text-blue " : "text-dim ") + (align === "l" ? "text-left" : "text-right")}
      >
        {label(lbl)}
        {sortKey === key && <span className="ml-1 text-[9px] text-blue">{sortDir < 0 ? "v" : "^"}</span>}
      </th>
    );
  }

  return (
    <div className="pt-3">
      <div className="mb-3 flex flex-wrap items-center gap-2.5">
        <h1 className="m-0 text-[17px] font-semibold tracking-tight">Scanner</h1>
        <div className="flex-1" />
        <StatusBadge
          status={status}
          tone={
            status === "live"
              ? "bg-up shadow-[0_0_7px_#3fb950]"
              : status === "scanning"
              ? "bg-gold"
              : status === "offline"
              ? "bg-gold"
              : "bg-dim"
          }
        />
        <Select
          value={params.watchlist}
          onChange={(v) => setParams((p) => ({ ...p, watchlist: v }))}
          options={SCAN_WATCHLISTS.map((w) => ({ value: w.value, label: label(w.label) }))}
          title="watchlist"
          width={150}
        />
        <Select
          value={params.interval}
          onChange={(v) => setParams((p) => ({ ...p, interval: v }))}
          options={SCAN_INTERVALS.map((i) => ({ value: i, label: i }))}
          title="timeframe"
          width={84}
        />
        <label className="field flex cursor-text items-center gap-1.5" title="minimum absolute score">
          <span className="whitespace-nowrap text-dim">Min Score</span>
          <input type="number" step={0.05} min={0} max={1} value={params.min_score_abs}
            onChange={(e) => setParams((p) => ({ ...p, min_score_abs: Number(e.target.value) }))}
            className="w-12 bg-transparent text-right font-semibold tnum text-ink outline-none" />
        </label>
        <button onClick={scan} disabled={status === "scanning"} className="btn-primary">
          {status === "scanning" ? "Scanning..." : "Scan"}
        </button>
      </div>

      {banner && <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">{banner}</div>}

      {report && (
        <>
          <div className="mb-3 grid grid-cols-3 gap-2 sm:grid-cols-6">
            <ScanStat label="scanned" value={report.scanned} />
            <ScanStat label="breakouts" value={report.breakouts.length} tone="text-up" />
            <ScanStat label="breakdowns" value={report.breakdowns.length} tone="text-down" />
            <ScanStat label="ok" value={report.succeeded} />
            <ScanStat label="failed" value={report.failed} tone={report.failed > 0 ? "text-gold" : "text-ink"} />
            <ScanStat label="seconds" value={Number((report.elapsed_ms / 1000).toFixed(1))} />
          </div>

          <div className="mb-2 flex flex-wrap gap-1.5">
            {([
              ["all", "all", report.all.length],
              ["breakouts", "breakouts", report.breakouts.length],
              ["breakdowns", "breakdowns", report.breakdowns.length],
              ["neutral", "neutral", report.neutral.length],
            ] as [Tab, string, number][]).map(([t, lbl, count]) => (
              <button key={t} onClick={() => setTab(t)}
                className={"flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[12px] font-medium transition-colors " + (tab === t ? "border-line2 bg-[#161c23] text-ink" : "border-line text-dim hover:text-muted")}>
                {label(lbl)}
                <span className={"rounded px-1.5 py-px text-[10px] tnum " + (tab === t ? "bg-blue/15 text-blue" : "bg-[#11161c] text-dim")}>{count}</span>
              </button>
            ))}
          </div>

          <div className="overflow-x-auto rounded-xl border border-line bg-panel shadow-panel">
            <table className="w-full tnum border-collapse text-[13px]">
              <thead>
                <tr>
                  {th("ticker", "ticker", "l")}
                  {th("price", "price")}
                  {th("score", "score")}
                  <th className="sticky top-0 z-10 whitespace-nowrap border-b border-line bg-[#0b1015] px-2.5 py-2.5 text-left text-[10.5px] font-semibold text-dim">{label("direction")}</th>
                  <th className="sticky top-0 z-10 whitespace-nowrap border-b border-line bg-[#0b1015] px-2.5 py-2.5 text-left text-[10.5px] font-semibold text-dim">{label("regime")}</th>
                  <th className="sticky top-0 z-10 whitespace-nowrap border-b border-line bg-[#0b1015] px-2.5 py-2.5 text-left text-[10.5px] font-semibold text-dim">{label("signal")}</th>
                  {th("confidence", "conf")}
                  {th("prob_up", "p(up)")}
                  {th("rsi", "rsi")}
                  {th("adx", "adx")}
                  {th("atr_pct", "atr%")}
                  {th("hurst", "hurst")}
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
          Pick a watchlist and press scan.
        </div>
      )}
    </div>
  );
}

function ScanStat({ label: name, value, tone = "text-ink" }: { label: string; value: number; tone?: string }) {
  return (
    <div className="rounded-lg border border-line bg-[#0d1217] px-3 py-2">
      <div className="flex items-baseline gap-1.5">
        <span className={"text-[16px] font-bold tnum " + tone}>{value}</span>
        <span className="text-[11px] text-muted">{label(name)}</span>
      </div>
    </div>
  );
}

function Row({ r, onLoad }: { r: ScanResult; onLoad: (t: string) => void }) {
  const pos = r.score >= 0;
  const w = Math.min(50, Math.abs(r.score) * 50);
  return (
    <tr className="group border-b border-[#10161c] odd:bg-rowalt hover:bg-[#13202b]">
      <td className="px-2.5 py-[7px]">
        <button
          onClick={() => onLoad(r.ticker)}
          className="inline-flex items-center gap-1 font-bold text-blue transition-colors hover:text-[#7cc7ff]"
          title={`${label("load")} ${r.ticker} ${label("in chart")}`}
        >
          {r.ticker}
          <svg viewBox="0 0 24 24" className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
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
      <td className={"px-2.5 py-[7px] text-left font-medium " + dirColor(r.direction)}>{label(r.direction)}</td>
      <td className="px-2.5 py-[7px] text-left text-muted">{label(r.regime)}</td>
      <td className="px-2.5 py-[7px] text-left">{displayText(r.signal_label)}</td>
      <td className="px-2.5 py-[7px] text-right text-muted">{(r.confidence * 100).toFixed(0)}%</td>
      <td className="px-2.5 py-[7px] text-right">{r.prob_up != null ? <span className={r.prob_up >= 50 ? "text-up" : "text-down"}>{r.prob_up.toFixed(0)}%</span> : <span className="text-dim">-</span>}</td>
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
