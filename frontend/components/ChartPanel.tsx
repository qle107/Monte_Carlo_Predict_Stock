"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchSignal, setTickerAndAnalyze } from "@/lib/analysisApi";
import type { AnalysisResult } from "@/lib/analysisTypes";
import { SAMPLE_SIGNAL } from "@/lib/sampleSignal";
import { sampleCandles } from "@/lib/sampleCandles";
import PriceChart, { type ChartLevel, type Overlays } from "./PriceChart";

const LEGEND: { key: keyof Overlays; label: string; color: string }[] = [
  { key: "ema9", label: "EMA 9", color: "#f2c14e" },
  { key: "ema21", label: "EMA 21", color: "#58a6ff" },
  { key: "ema200", label: "EMA 200", color: "#e6edf3" },
  { key: "bb", label: "Bollinger", color: "#58a6ff" },
  { key: "vwap", label: "VWAP", color: "#bc6bd9" },
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
      setBanner(`Couldn't reach the API (${(err as Error).message}). Showing sample chart — start FastAPI and refresh.`);
    }
  }, []);

  useEffect(() => {
    load(fetchSignal);
  }, [load]);

  const d = data;
  const candles = d?.candles ?? [];

  const levels: ChartLevel[] = [];
  if (d?.trade_setup?.valid) {
    if (d.trade_setup.entry != null) levels.push({ price: Number(d.trade_setup.entry), label: "Entry", color: "#58a6ff" });
    if (d.trade_setup.stop != null) levels.push({ price: Number(d.trade_setup.stop), label: "Stop", color: "#f0556d" });
    if (d.trade_setup.target != null) levels.push({ price: Number(d.trade_setup.target), label: "Target", color: "#3fb950" });
  }

  return (
    <div className="pt-3">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h1 className="m-0 text-[17px] font-semibold tracking-tight">
          Price Chart
          {d && <span className="ml-2 text-[13px] font-normal text-muted">{d.ticker} · {d.interval}</span>}
        </h1>
        {d && (
          <span className="text-[15px] font-semibold tnum">
            ${d.current_price.toFixed(2)}
          </span>
        )}
        <div className="flex-1" />
        <span className="flex items-center gap-1.5 text-[12px] text-muted">
          <span className={"inline-block h-[7px] w-[7px] rounded-full " + (status === "live" ? "bg-up shadow-[0_0_7px_#3fb950]" : status === "loading" ? "bg-gold" : "bg-dim")} />
          {status}
        </span>
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
            placeholder="ticker"
            className="w-24 rounded-md border border-line bg-[#161c23] px-2.5 py-1.5 text-[12px] uppercase outline-none focus:border-[#2c3a47]"
          />
          <button type="submit" className="rounded-md border border-[#1f6feb] bg-[#1f6feb] px-3 py-1.5 text-[12px] font-semibold text-white hover:bg-[#388bfd]">Load</button>
        </form>
        <button onClick={() => load(fetchSignal)} className="rounded-md border border-line bg-[#161c23] px-3 py-1.5 text-[12px] hover:border-[#2c3a47]">Refresh</button>
      </div>

      {/* overlay toggles */}
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
              {l.label}
            </button>
          );
        })}
      </div>

      {banner && (
        <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">{banner}</div>
      )}

      <div className="rounded-xl border border-line bg-panel p-3">
        {candles.length > 1 ? (
          <PriceChart candles={candles} overlays={ov} levels={levels} />
        ) : (
          <div className="p-10 text-center text-muted">Loading chart…</div>
        )}
      </div>

      <p className="mt-2.5 text-[11px] leading-relaxed text-dim">
        Candles &amp; volume from <code className="rounded bg-[#161c23] px-1 text-[#a9c7e8]">/api/signal</code>. Overlays
        (EMA, Bollinger, VWAP) are computed client-side from the candle closes for display; the
        authoritative indicator values live in the Signal panel. Entry/Stop/Target lines come from the trade setup.
      </p>
    </div>
  );
}
