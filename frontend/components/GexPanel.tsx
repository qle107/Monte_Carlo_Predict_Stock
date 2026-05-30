"use client";

import { useEffect, useState } from "react";
import { label } from "@/lib/display";
import { fetchGex } from "@/lib/gexApi";
import type { OptionsFlow } from "@/lib/gexTypes";
import { SAMPLE_GEX } from "@/lib/sampleGex";
import GexChart from "./GexChart";
import StatusBadge from "./StatusBadge";

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
      setBanner(`Could not reach API (${(err as Error).message}). Showing sample data.`);
    }
  }

  useEffect(() => {
    load("PLTR");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const f = flow;
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
          GEX / Max Pain
          {f && (
            <span className="ml-2 text-[13px] font-normal text-muted">
              {f.ticker} / {label("exp")} {f.expiry} / {f.days_to_expiry.toFixed(0)}d
            </span>
          )}
        </h1>
        <div className="flex-1" />
        <StatusBadge status={status} tone={statusTone} />
        <form onSubmit={(e) => { e.preventDefault(); if (tickerInput.trim()) load(tickerInput); }} className="flex items-center gap-2">
          <input value={tickerInput} onChange={(e) => setTickerInput(e.target.value)} placeholder="Ticker"
            className="field w-24 uppercase" />
          <button type="submit" className="btn-primary">Load</button>
        </form>
      </div>

      {banner && (
        <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">
          {banner}
        </div>
      )}

      {!f ? (
        <div className="rounded-xl border border-line bg-panel p-10 text-center text-muted">
          Loading...
        </div>
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <Stat name="spot" value={`$${f.spot.toFixed(2)}`} />
            <Stat name="max pain" value={`$${f.max_pain.toFixed(2)}`} tone="magenta" />
            <Stat name="gamma flip" value={`$${f.gamma_flip.toFixed(2)}`} tone="gold" />
            <Stat name="call wall" value={`$${f.call_wall.toFixed(2)}`} tone="up" />
            <Stat name="put wall" value={`$${f.put_wall.toFixed(2)}`} tone="down" />
            <Stat
              name="net GEX"
              value={`${f.net_gex >= 0 ? "+" : ""}${(f.net_gex / 1e6).toFixed(0)}M`}
              tone={f.gex_positive ? "up" : "down"}
              sub={f.gex_positive ? "vol-damping" : "vol-amplifying"}
            />
          </div>

          <div className="panel">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-[11px] font-semibold tracking-wide text-dim">{label("net GEX by strike")}</h2>
              <span className="flex items-center gap-3 text-[10px] text-dim">
                <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-up" /> {label("positive")}</span>
                <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-down" /> {label("negative")}</span>
              </span>
            </div>
            <GexChart flow={f} />
          </div>

          <p className="mt-2.5 text-[11px] text-dim">
            Dealer gamma from /api/options/gex. Max pain and gamma flip marked on chart.
          </p>
        </>
      )}
    </div>
  );
}

function Stat({ name, value, sub, tone }: { name: string; value: string; sub?: string; tone?: "up" | "down" | "gold" | "magenta" }) {
  const c = tone === "up" ? "text-up" : tone === "down" ? "text-down" : tone === "gold" ? "text-gold" : tone === "magenta" ? "text-magenta" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-panel p-3">
      <div className="text-[11px] tracking-wide text-dim">{label(name)}</div>
      <div className={"mt-1 text-[18px] font-semibold tnum " + c}>{value}</div>
      {sub && <div className="mt-1 text-[11px] text-muted">{label(sub)}</div>}
    </div>
  );
}
