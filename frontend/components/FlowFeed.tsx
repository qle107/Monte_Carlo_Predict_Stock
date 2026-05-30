"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchFlow, WATCHLISTS } from "@/lib/api";
import { bsDelta } from "@/lib/blackScholes";
import {
  cap,
  fmtExpiry,
  fmtPeakReturn,
  fmtStrike,
  fmtValue,
  hms,
  sigColor,
} from "@/lib/format";
import { SAMPLE_FLOW } from "@/lib/sampleFlow";
import {
  DEFAULT_FILTERS,
  type FlowFilters,
  type FlowRow,
  type ScanResponse,
  type ScanSummary,
} from "@/lib/types";

type SortKey =
  | "time"
  | "value"
  | "ticker"
  | "spot"
  | "strike"
  | "pc"
  | "exp"
  | "side"
  | "type"
  | "price"
  | "size"
  | "sig"
  | "ret"
  | "delta"
  | "vol";

interface Col {
  key: SortKey;
  label: string;
  align: "l" | "r";
  sortable?: boolean;
}

const COLS: Col[] = [
  { key: "time", label: "Time", align: "l", sortable: false },
  { key: "value", label: "Value", align: "r" },
  { key: "ticker", label: "Ticker", align: "l" },
  { key: "spot", label: "Spot", align: "r" },
  { key: "strike", label: "Strike", align: "r" },
  { key: "pc", label: "PC", align: "l" },
  { key: "exp", label: "Exp.", align: "r" },
  { key: "side", label: "X", align: "l" },
  { key: "type", label: "Type", align: "l" },
  { key: "price", label: "Price", align: "r" },
  { key: "size", label: "Size", align: "r" },
  { key: "sig", label: "SigScore", align: "r" },
  { key: "ret", label: "Peak Return", align: "r" },
  { key: "delta", label: "Δ", align: "r" },
  { key: "vol", label: "Volume", align: "r" },
];

function sortVal(r: FlowRow, k: SortKey): number | string {
  switch (k) {
    case "value": return r.total_premium;
    case "ticker": return r.ticker;
    case "spot": return r.spot;
    case "strike": return r.strike;
    case "pc": return r.option_type;
    case "exp": return r.expiry;
    case "side": return r.exec_side;
    case "type": return r.trade_style;
    case "price": return r.premium_per_contract;
    case "size": return r.volume;
    case "sig": return r.unusual_score;
    case "ret": return r.percent_change ?? -Infinity;
    case "delta": return r._delta;
    case "vol": return r.open_interest;
    default: return 0;
  }
}

export default function FlowFeed() {
  const [rows, setRows] = useState<FlowRow[]>([]);
  const [summary, setSummary] = useState<ScanSummary | null>(null);
  const [scannedAt, setScannedAt] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "scanning" | "live" | "offline">("idle");
  const [banner, setBanner] = useState<string>("");
  const [filters, setFilters] = useState<FlowFilters>(DEFAULT_FILTERS);
  const [autoSec, setAutoSec] = useState<number>(0);
  const [sortKey, setSortKey] = useState<SortKey>("value");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const ingest = useCallback((data: ScanResponse) => {
    const t = hms(data.scanned_at);
    const enriched: FlowRow[] = (data.hits || []).map((h) => ({
      ...h,
      _delta: bsDelta(h),
      _time: t,
    }));
    setRows(enriched);
    setSummary(data.summary);
    setScannedAt(t);
  }, []);

  const scan = useCallback(async () => {
    // Cancel any in-flight scan so we never have two heavy requests racing.
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setStatus("scanning");
    setBanner("");
    try {
      const data = await fetchFlow(filters, ctrl.signal);
      ingest(data);
      setStatus("live");
    } catch (err) {
      if ((err as Error).name === "AbortError") return; // superseded, ignore
      ingest(SAMPLE_FLOW);
      setStatus("offline");
      setBanner(
        `Couldn't reach the API (${(err as Error).message}). Showing sample layout — start FastAPI (uvicorn api.server:app) and Scan again for live flow.`
      );
    }
  }, [filters, ingest]);

  // initial scan
  useEffect(() => {
    scan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // auto-refresh
  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (autoSec > 0) {
      timerRef.current = setInterval(scan, autoSec * 1000);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      abortRef.current?.abort();
    };
  }, [autoSec, scan]);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const x = sortVal(a, sortKey);
      const y = sortVal(b, sortKey);
      if (typeof x === "string" && typeof y === "string") {
        return sortDir * x.localeCompare(y);
      }
      return sortDir * ((x as number) - (y as number));
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  function onSort(c: Col) {
    if (c.sortable === false) return;
    if (sortKey === c.key) {
      setSortDir((d) => (d === 1 ? -1 : 1));
    } else {
      setSortKey(c.key);
      setSortDir(c.key === "ticker" || c.key === "pc" || c.key === "type" || c.key === "side" ? 1 : -1);
    }
  }

  return (
    <div className="pt-3">
      {/* Header */}
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h1 className="m-0 text-[17px] font-semibold tracking-tight">
          Options Flow
          <span className="ml-2 text-[12px] font-normal text-muted">
            sweeps &amp; blocks · ask-side conviction
          </span>
        </h1>
        <div className="flex-1" />
        <span className="flex items-center gap-1.5 text-[12px] text-muted">
          <span
            className={
              "inline-block h-[7px] w-[7px] rounded-full " +
              (status === "live" ? "bg-up shadow-[0_0_7px_#3fb950]" : status === "scanning" ? "bg-gold" : "bg-dim")
            }
          />
          {status}
        </span>
        <select
          className="rounded-md border border-line bg-[#161c23] px-2.5 py-1.5 text-[12px]"
          value={filters.watchlist}
          onChange={(e) => setFilters((f) => ({ ...f, watchlist: e.target.value }))}
        >
          {WATCHLISTS.map((w) => (
            <option key={w.value} value={w.value}>{w.label}</option>
          ))}
        </select>
        <select
          className="rounded-md border border-line bg-[#161c23] px-2.5 py-1.5 text-[12px]"
          value={autoSec}
          onChange={(e) => setAutoSec(Number(e.target.value))}
        >
          <option value={0}>manual</option>
          <option value={30}>30s</option>
          <option value={60}>60s</option>
          <option value={120}>2m</option>
        </select>
        <button
          onClick={scan}
          className="rounded-md border border-[#1f6feb] bg-[#1f6feb] px-3.5 py-1.5 text-[12px] font-semibold text-white hover:bg-[#388bfd]"
        >
          Scan
        </button>
      </div>

      {/* Filter chips */}
      <div className="mb-3 flex flex-wrap gap-1.5">
        <Chip>Sweeps ≥ <b className="text-ink">{fmtValue(filters.min_sweep_premium)}</b></Chip>
        <Chip>Blocks ≥ <b className="text-ink">{fmtValue(filters.min_block_premium)}</b></Chip>
        {filters.exclude_high_volume_etfs && <Chip>Excl. high-volume ETFs</Chip>}
        {filters.exclude_bid_side && <Chip>Ask-side only <span className="text-dim">(bid hidden)</span></Chip>}
      </div>

      {/* Summary */}
      {summary && (
        <div className="mb-3 flex flex-wrap gap-4 text-[12px] text-muted">
          <span><b className="text-ink">{summary.total_hits}</b> prints</span>
          <span className="text-sweep"><b>{summary.sweep_count}</b> sweeps</span>
          <span className="text-blue"><b>{summary.block_count}</b> blocks</span>
          <span className="text-up"><b>{summary.bullish_count}</b> bullish</span>
          <span className="text-down"><b>{summary.bearish_count}</b> bearish</span>
          <span><b className="text-ink">{summary.tickers_scanned}</b> scanned</span>
          {summary.excluded_etf_count > 0 && (
            <span><b className="text-ink">{summary.excluded_etf_count}</b> ETFs excluded</span>
          )}
          <span className="text-dim">@ {scannedAt}</span>
        </div>
      )}

      {banner && (
        <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">
          {banner}
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-line bg-panel">
        <table className="w-full tnum border-collapse text-[13px]">
          <thead>
            <tr>
              {COLS.map((c) => (
                <th
                  key={c.key}
                  onClick={() => onSort(c)}
                  className={
                    "sticky top-0 whitespace-nowrap border-b border-line bg-[#0d1217] px-2.5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-dim " +
                    (c.align === "l" ? "text-left " : "text-right ") +
                    (c.sortable === false ? "cursor-default" : "cursor-pointer select-none")
                  }
                >
                  {c.label}
                  {sortKey === c.key && c.sortable !== false && (
                    <span className="ml-1 text-[9px] text-blue">{sortDir < 0 ? "▼" : "▲"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={COLS.length} className="p-10 text-center text-muted">
                  No qualifying sweeps or blocks.
                </td>
              </tr>
            ) : (
              sorted.map((r, i) => <Row key={i} r={r} />)
            )}
          </tbody>
        </table>
      </div>

      {/* Footnote */}
      <p className="mt-2.5 text-[11px] leading-relaxed text-dim">
        <b>Data notes.</b> Built on <code className="rounded bg-[#161c23] px-1 text-[#a9c7e8]">/api/options/unusual</code> (yfinance snapshot).{" "}
        <b>Sweep/Block</b> approximated from volume/OI; <b>X</b> (Ask/Bid) is a last-vs-bid/ask lean — bid-side filtered server-side.{" "}
        <b>Time</b> is the scan timestamp (snapshot, not per-trade tape). <b>Size</b> = day volume; <b>Volume</b> = open interest.{" "}
        <b>Δ</b> is Black–Scholes, computed client-side.
      </p>
    </div>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="whitespace-nowrap rounded-full border border-[#244] bg-[#10211c] px-2.5 py-1 text-[11.5px] text-[#9fe0bf]">
      {children}
    </span>
  );
}

function Row({ r }: { r: FlowRow }) {
  const isCall = r.option_type === "call";
  const big = r.total_premium >= 5e5;
  const sizeHot = (r.volume || 0) >= 150;
  const d = r._delta ?? 0;
  const score = r.unusual_score || 0;
  const sideColor = r.exec_side === "ask" ? "text-up" : r.exec_side === "bid" ? "text-down" : "text-muted";
  const ret = r.percent_change;
  const retColor = ret == null ? "text-dim" : ret > 0 ? "text-up" : "text-dim";

  return (
    <tr className="border-b border-[#141a20] odd:bg-rowalt hover:bg-[#13202b]">
      <td className="px-2.5 py-[7px] text-left text-[11px] text-muted">{r._time}</td>
      <td className={"px-2.5 py-[7px] text-right font-bold " + (big ? "text-gold" : "text-ink")}>
        {fmtValue(r.total_premium)}
      </td>
      <td className="px-2.5 py-[7px] text-left">
        <span
          className={
            "inline-block min-w-[48px] rounded-md border px-2 py-[3px] text-center text-[11.5px] font-bold tracking-wide " +
            (isCall
              ? "border-up/30 bg-up/10 text-[#56d364]"
              : "border-down/30 bg-down/10 text-[#ff7b93]")
          }
        >
          {r.ticker}
        </span>
      </td>
      <td className="px-2.5 py-[7px] text-right text-[#c9d4df]">{r.spot.toFixed(2)}</td>
      <td className="px-2.5 py-[7px] text-right">{fmtStrike(r.strike)}</td>
      <td className={"px-2.5 py-[7px] text-left font-semibold " + (isCall ? "text-up" : "text-down")}>
        {isCall ? "Call" : "Put"}
      </td>
      <td className="px-2.5 py-[7px] text-right text-muted">{fmtExpiry(r.expiry)}</td>
      <td className={"px-2.5 py-[7px] text-left font-semibold " + sideColor}>{cap(r.exec_side)}</td>
      <td className={"px-2.5 py-[7px] text-left font-semibold " + (r.trade_style === "sweep" ? "text-sweep" : "text-blue")}>
        {cap(r.trade_style)}
      </td>
      <td className="px-2.5 py-[7px] text-right text-[#c9d4df]">{(r.premium_per_contract / 100).toFixed(2)}</td>
      <td className={"px-2.5 py-[7px] text-right font-semibold " + (sizeHot ? "text-magenta" : "text-[#c9d4df]")}>
        {(r.volume || 0).toLocaleString()}
      </td>
      <td className="px-2.5 py-[7px] text-right">
        <span className="inline-flex flex-col items-end gap-[3px]">
          <span className="h-[7px] w-[74px] overflow-hidden rounded bg-[#1c242c]">
            <span
              className="block h-full rounded"
              style={{ width: `${Math.round(score * 100)}%`, background: sigColor(score) }}
            />
          </span>
          <span className="text-[10.5px] text-muted">{score.toFixed(2)}</span>
        </span>
      </td>
      <td className={"px-2.5 py-[7px] text-right font-medium " + retColor}>{fmtPeakReturn(ret)}</td>
      <td className={"px-2.5 py-[7px] text-right " + (d >= 0 ? "text-[#9fb2c2]" : "text-[#c98aa8]")}>
        {d.toFixed(3)}
      </td>
      <td className="px-2.5 py-[7px] text-right text-muted">{(r.open_interest || 0).toLocaleString()}</td>
    </tr>
  );
}
