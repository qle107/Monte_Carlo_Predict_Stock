"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchSignal, setTickerAndAnalyze } from "@/lib/analysisApi";
import type { AnalysisResult } from "@/lib/analysisTypes";
import { label } from "@/lib/display";
import { SAMPLE_SIGNAL } from "@/lib/sampleSignal";
import { sampleCandles } from "@/lib/sampleCandles";
import PriceChart, { type ChartLevel, type Overlays } from "./PriceChart";
import StatusBadge from "./StatusBadge";

const LEGEND: { key: keyof Overlays; label: string; color: string }[] = [
  { key: "ema9", label: "ema 9", color: "#f2c14e" },
  { key: "ema21", label: "ema 21", color: "#58a6ff" },
  { key: "ema200", label: "ema 200", color: "#e6edf3" },
  { key: "bb", label: "bb", color: "#58a6ff" },
  { key: "vwap", label: "vwap", color: "#bc6bd9" },
];

export default function ChartPanel() {
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "live" | "offline">("idle");
  const [banner, setBanner] = useState("");
  const [tickerInput, setTickerInput] = useState("");
  const [ov, setOv] = useState<Overlays>({ ema9: true, ema21: true, ema200: false, bb: false, vwap: true });

  const load = useCallback(async (fn: () => Promise<AnalysisResult>) => {
    setStatus("loading");
    setBanner("");
    try {
      const d = await fn();
      setData(d);
      setStatus("live");
    } catch (err) {
      setData({ ...SAMPLE_SIGNAL, candles: sampleCandles() });
      setStatus("offline");
      setBanner(`Could not reach API (${(err as Error).message}). Showing sample data.`);
    }
  }, []);

  useEffect(() => {
    load(fetchSignal);
  }, [load]);

  const d = data;
  const candles = d?.candles ?? [];

  const levels: ChartLevel[] = [];
  if (d?.trade_setup?.valid) {
    if (d.trade_setup.entry != null) levels.push({ price: Number(d.trade_setup.entry), label: "entry", color: "#58a6ff" });
    if (d.trade_setup.stop != null) levels.push({ price: Number(d.trade_setup.stop), label: "stop", color: "#f0556d" });
    if (d.trade_setup.target != null) levels.push({ price: Number(d.trade_setup.target), label: "target", color: "#3fb950" });
  }

  const statusTone =
    status === "live"
      ? "bg-up shadow-[0_0_7px_#3fb950]"
      : status === "loading"
      ? "bg-gold"
      : status === "offline"
      ? "bg-gold"
      : "bg-dim";

  return (
    <div className="pt-3">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h1 className="m-0 text-[17px] font-semibold tracking-tight">
          Chart
          {d && <span className="ml-2 text-[13px] font-normal text-muted">{d.ticker} / {d.interval}</span>}
        </h1>
        {d && (
          <span className="text-[15px] font-semibold tnum">
            ${d.current_price.toFixed(2)}
          </span>
        )}
        <div className="flex-1" />
        <StatusBadge status={status} tone={statusTone} />
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (tickerInput.trim()) load(() => setTickerAndAnalyze(tickerInput));
          }}
          className="flex items-center gap-2"
        >
          <input
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            placeholder="Ticker"
            className="field w-24 uppercase"
          />
          <button type="submit" className="btn-primary">Load</button>
        </form>
        <button onClick={() => load(fetchSignal)} className="btn-ghost">Refresh</button>
      </div>

      <div className="mb-2 flex flex-wrap gap-1.5">
        {LEGEND.map((l) => {
          const on = ov[l.key];
          return (
            <button
              key={l.key}
              onClick={() => setOv((o) => ({ ...o, [l.key]: !o[l.key] }))}
              className={"flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11.5px] transition-colors " + (on ? "border-[#2c3a47] bg-[#161c23] text-ink" : "border-line text-dim")}
            >
              <span className="inline-block h-[3px] w-3 rounded" style={{ background: on ? l.color : "#3a424b" }} />
              {label(l.label)}
            </button>
          );
        })}
      </div>

      {banner && (
        <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">{banner}</div>
      )}

      <div className="rounded-xl border border-line bg-panel p-3 shadow-panel">
        {candles.length > 1 ? (
          <PriceChart candles={candles} overlays={ov} levels={levels} />
        ) : (
          <div className="p-10 text-center text-muted">Loading...</div>
        )}
      </div>

      <p className="mt-2.5 text-[11px] text-dim">
        Candles from /api/signal; overlays computed client-side.
      </p>
    </div>
  );
}
