"use client";

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchFlow, WATCHLISTS } from "@/lib/api";
import { bsDelta } from "@/lib/blackScholes";
import { label } from "@/lib/display";
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
import Select from "./Select";
import StatusBadge from "./StatusBadge";

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
  { key: "time", label: "time", align: "l", sortable: false },
  { key: "value", label: "value", align: "r" },
  { key: "ticker", label: "ticker", align: "l" },
  { key: "spot", label: "spot", align: "r" },
  { key: "strike", label: "strike", align: "r" },
  { key: "pc", label: "cp", align: "l" },
  { key: "exp", label: "exp", align: "r" },
  { key: "side", label: "side", align: "l" },
  { key: "type", label: "type", align: "l" },
  { key: "price", label: "price", align: "r" },
  { key: "size", label: "size", align: "r" },
  { key: "sig", label: "score", align: "r" },
  { key: "ret", label: "chg%", align: "r" },
  { key: "delta", label: "delta", align: "r" },
  { key: "vol", label: "oi", align: "r" },
];

// Render table rows in chunks so large scans do not block the main thread.
const CHUNK = 80;

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

function Chip({ children }: { children: React.ReactNode }) {
  return <span className="chip-good">{children}</span>;
}

function SummaryStat({
  label: name,
  value,
  sub,
  tone = "text-ink",
}: {
  label: string;
  value: number;
  sub?: string;
  tone?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-line bg-[#0d1217] px-3 py-2.5 text-center">
      <div className="flex items-baseline justify-center gap-1.5">
        <span className={"text-[17px] font-bold tnum " + tone}>{value}</span>
        <span className="text-[11px] text-muted">{label(name)}</span>
      </div>
      {sub && <div className="mt-0.5 text-[10px] text-dim">{sub}</div>}
    </div>
  );
}

export default function FlowFeed() {
  const [rows, setRows] = useState<FlowRow[]>([]);
  const [summary, setSummary] = useState<ScanSummary | null>(null);
  const [scannedAt, setScannedAt] = useState<string>("");
  const [status, setStatus] = useState<"idle" | "scanning" | "live" | "offline">("idle");
  const [banner, setBanner] = useState<string>("");
  const [filters, setFilters] = useState<FlowFilters>(DEFAULT_FILTERS);
  const [sortKey, setSortKey] = useState<SortKey>("value");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);
  const [visible, setVisible] = useState<number>(CHUNK);
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
      setBanner(`Could not reach API (${(err as Error).message}). Showing sample data.`);
    }
  }, [filters, ingest]);

  useEffect(() => () => abortRef.current?.abort(), []);

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

  // Reveal rows in chunks for large result sets.
  useEffect(() => {
    if (sorted.length <= CHUNK) {
      setVisible(sorted.length);
      return;
    }
    setVisible(CHUNK);
    let v = CHUNK;
    let raf = 0;
    const step = () => {
      v = Math.min(sorted.length, v + CHUNK);
      setVisible(v);
      if (v < sorted.length) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [sorted]);

  const shown = visible >= sorted.length ? sorted : sorted.slice(0, visible);
  const loadingFirst = status === "scanning" && rows.length === 0;

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
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <h1 className="m-0 text-[17px] font-semibold tracking-tight">Options Flow</h1>
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
        {/* unusual-options scan field: minimum volume / open-interest ratio */}
        <label className="field flex cursor-text items-center gap-1.5" title="minimum volume / open-interest ratio">
          <span className="whitespace-nowrap text-dim">Vol/OI &gt;=</span>
          <input
            type="number"
            min={0}
            step={0.5}
            value={filters.vol_oi_threshold}
            onChange={(e) => setFilters((f) => ({ ...f, vol_oi_threshold: Math.max(0, Number(e.target.value) || 0) }))}
            className="w-10 bg-transparent text-right font-semibold tnum text-ink outline-none"
          />
        </label>
        <Select
          value={filters.watchlist}
          onChange={(v) => setFilters((f) => ({ ...f, watchlist: v }))}
          options={WATCHLISTS}
          title="watchlist"
          width={156}
        />
        <button onClick={scan} disabled={status === "scanning"} className="btn-primary">
          {status === "scanning" ? "Scanning..." : "Scan"}
        </button>
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        <Chip>Sweeps &gt;= <b className="text-ink">{fmtValue(filters.min_sweep_premium)}</b></Chip>
        <Chip>Blocks &gt;= <b className="text-ink">{fmtValue(filters.min_block_premium)}</b></Chip>
        <Chip>Vol/OI &gt;= <b className="text-ink">{filters.vol_oi_threshold}x</b></Chip>
        <Chip>&lt;= <b className="text-ink">{filters.max_dte}</b> DTE</Chip>
        {filters.exclude_high_volume_etfs && <Chip>Excl. ETFs</Chip>}
        {filters.exclude_bid_side && <Chip>Ask side only</Chip>}
      </div>

      {summary && (
        <div className="mb-3 grid grid-cols-3 gap-2 sm:grid-cols-6">
          <SummaryStat label="prints" value={summary.total_hits} />
          <SummaryStat label="sweeps" value={summary.sweep_count} tone="text-sweep" />
          <SummaryStat label="blocks" value={summary.block_count} tone="text-blue" />
          <SummaryStat label="bullish" value={summary.bullish_count} tone="text-up" />
          <SummaryStat label="bearish" value={summary.bearish_count} tone="text-down" />
          <SummaryStat
            label="scanned"
            value={summary.tickers_scanned}
            sub={summary.excluded_etf_count > 0 ? `${summary.excluded_etf_count} ${label("etfs excl")}` : `@ ${scannedAt}`}
          />
        </div>
      )}

      {banner && (
        <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">
          {banner}
        </div>
      )}

      <div className="overflow-x-auto rounded-xl border border-line bg-panel">
        <table className="w-full tnum border-collapse text-[13px]">
          <thead>
            <tr>
              {COLS.map((c) => (
                <th
                  key={c.key}
                  onClick={() => onSort(c)}
                  className={
                    "sticky top-0 z-10 whitespace-nowrap border-b border-line bg-[#0b1015] px-2.5 py-2.5 text-[10.5px] font-semibold tracking-wide text-dim " +
                    (c.align === "l" ? "text-left " : "text-right ") +
                    (c.sortable === false ? "cursor-default" : "cursor-pointer select-none hover:text-muted ") +
                    (sortKey === c.key && c.sortable !== false ? "text-blue" : "")
                  }
                >
                  {label(c.label)}
                  {sortKey === c.key && c.sortable !== false && (
                    <span className="ml-1 text-[9px] text-blue">{sortDir < 0 ? "v" : "^"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loadingFirst ? (
              Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)
            ) : shown.length === 0 ? (
              <tr>
                <td colSpan={COLS.length} className="p-12 text-center text-muted">
                  {status === "idle" ? "Press Scan to load options flow." : "No rows."}
                </td>
              </tr>
            ) : (
              shown.map((r, i) => <Row key={`${r.ticker}-${r.strike}-${r.expiry}-${r.option_type}-${i}`} r={r} />)
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-2.5 flex items-center gap-2 text-[11px] text-dim">
        <span>yfinance chain snapshot. Sweep/block and side are approximations.</span>
        {sorted.length > 0 && (
          <span className="text-dim/80">
            Showing {Math.min(visible, sorted.length).toLocaleString()} / {sorted.length.toLocaleString()}
          </span>
        )}
      </p>
    </div>
  );
}

const SkeletonRow = memo(function SkeletonRow() {
  return (
    <tr className="border-b border-[#10161c]">
      {COLS.map((c) => (
        <td key={c.key} className="px-2.5 py-[9px]">
          <div className={"h-3 animate-pulse rounded bg-[#1c242c] " + (c.align === "r" ? "ml-auto w-10" : "w-12")} />
        </td>
      ))}
    </tr>
  );
});

const Row = memo(function Row({ r }: { r: FlowRow }) {
  const isCall = r.option_type === "call";
  const big = r.total_premium >= 5e5;
  const sizeHot = (r.volume || 0) >= 150;
  const d = r._delta ?? 0;
  const score = r.unusual_score || 0;
  const sideColor = r.exec_side === "ask" ? "text-up" : r.exec_side === "bid" ? "text-down" : "text-muted";
  const ret = r.percent_change;
  const retColor = ret == null ? "text-dim" : ret > 0 ? "text-up" : "text-dim";

  return (
    <tr className="group border-b border-[#10161c] odd:bg-rowalt hover:bg-[#13202b]">
      <td className="px-2.5 py-[7px] text-left text-[11px] text-muted">{r._time}</td>
      <td className={"relative px-2.5 py-[7px] text-right font-bold " + (big ? "text-gold" : "text-ink")}>
        {big && <span className="absolute inset-y-1 left-0 w-[2px] rounded-full bg-gold/70" />}
        {fmtValue(r.total_premium)}
      </td>
      <td className="px-2.5 py-[7px] text-left">
        <span
          className={
            "inline-block min-w-[48px] rounded-md border px-2 py-[3px] text-center text-[11.5px] font-bold tracking-wide " +
            (isCall
              ? "border-up/30 bg-up/10 text-up"
              : "border-down/30 bg-down/10 text-down")
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
          <span className="h-[7px] w-[74px] overflow-hidden rounded-full bg-[#1c242c]">
            <span
              className="block h-full rounded-full"
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
});
