"use client";

import { useEffect, useState } from "react";
import { fetchGex } from "@/lib/gexApi";
import type { OptionsFlow } from "@/lib/gexTypes";
import { SAMPLE_GEX } from "@/lib/sampleGex";
import GexChart from "./GexChart";

export default function GexPanel() {
  const [flow, setFlow] = useState<OptionsFlow | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "live" | "offline">("idle");
  const [banner, setBanner] = useState("");
  const [tickerInput, setTickerInput] = useState("PLTR");

  async function load(t: string) {
    setStatus("loading");
    setBanner("");
    try {
      const f = await fetchGex(t);
      setFlow(f);
      setStatus("live");
    } catch (err) {
      setFlow(SAMPLE_GEX);
      setStatus("offline");
      setBanner(`Couldn't reach the API (${(err as Error).message}). Showing sample GEX — start FastAPI and load a ticker.`);
    }
  }

  useEffect(() => {
    load("PLTR");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const f = flow;

  return (
    <div className="pt-3">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h1 className="m-0 text-[17px] font-semibold tracking-tight">
          GEX / Max Pain
          {f && <span className="ml-2 text-[13px] font-normal text-muted">{f.ticker} · exp {f.expiry} · {f.days_to_expiry.toFixed(0)}d</span>}
        </h1>
        <div className="flex-1" />
        <span className="flex items-center gap-1.5 text-[12px] text-muted">
          <span className={"inline-block h-[7px] w-[7px] rounded-full " + (status === "live" ? "bg-up shadow-[0_0_7px_#3fb950]" : status === "loading" ? "bg-gold" : "bg-dim")} />
          {status}
        </span>
        <form onSubmit={(e) => { e.preventDefault(); if (tickerInput.trim()) load(tickerInput); }} className="flex items-center gap-2">
          <input value={tickerInput} onChange={(e) => setTickerInput(e.target.value)} placeholder="ticker"
            className="w-24 rounded-md border border-line bg-[#161c23] px-2.5 py-1.5 text-[12px] uppercase outline-none focus:border-[#2c3a47]" />
          <button type="submit" className="rounded-md border border-[#1f6feb] bg-[#1f6feb] px-3 py-1.5 text-[12px] font-semibold text-white hover:bg-[#388bfd]">Load</button>
        </form>
      </div>

      {banner && <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">{banner}</div>}

      {!f ? (
        <div className="rounded-xl border border-line bg-panel p-10 text-center text-muted">Loading…</div>
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Spot" value={`$${f.spot.toFixed(2)}`} />
            <Stat label="Max pain" value={`$${f.max_pain.toFixed(2)}`} tone="magenta" />
            <Stat label="γ-flip" value={`$${f.gamma_flip.toFixed(2)}`} tone="gold" />
            <Stat label="Call wall" value={`$${f.call_wall.toFixed(2)}`} tone="up" />
            <Stat label="Put wall" value={`$${f.put_wall.toFixed(2)}`} tone="down" />
            <Stat
              label="Net GEX"
              value={`${f.net_gex >= 0 ? "+" : ""}${(f.net_gex / 1e6).toFixed(0)}M`}
              tone={f.gex_positive ? "up" : "down"}
              sub={f.gex_positive ? "vol-damping" : "vol-amplifying"}
            />
          </div>

          <div className="rounded-xl border border-line bg-panel p-3">
            <div className="mb-2 flex items-center justify-between px-1">
              <h2 className="text-[11px] font-semibold uppercase tracking-wider text-dim">Net GEX by strike</h2>
              <span className="flex items-center gap-3 text-[10px] text-dim">
                <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-up" /> positive</span>
                <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-down" /> negative</span>
              </span>
            </div>
            <GexChart flow={f} />
          </div>

          <p className="mt-2.5 text-[11px] leading-relaxed text-dim">
            Dealer gamma exposure per strike from <code className="rounded bg-[#161c23] px-1 text-[#a9c7e8]">/api/options/gex</code> (Black–Scholes
            gamma × OI). <b>Positive net GEX</b> ⇒ dealers buy dips / sell rips (vol-damping); <b>negative</b> ⇒ they chase moves (vol-amplifying).
            <b> Max pain</b> is the strike minimizing total option payout; <b>γ-flip</b> is where net GEX crosses zero.
          </p>
        </>
      )}
    </div>
  );
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "up" | "down" | "gold" | "magenta" }) {
  const c = tone === "up" ? "text-up" : tone === "down" ? "text-down" : tone === "gold" ? "text-gold" : tone === "magenta" ? "text-magenta" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-panel p-3">
      <div className="text-[11px] uppercase tracking-wider text-dim">{label}</div>
      <div className={"mt-1 text-[17px] font-semibold tnum " + c}>{value}</div>
      {sub && <div className="text-[11px] text-muted">{sub}</div>}
    </div>
  );
}
