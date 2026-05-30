"use client";

import { useCallback, useEffect, useState } from "react";
import { displayText, label } from "@/lib/display";
import { fetchSignal, INTERVALS, MC_MODELS, setConfigAndAnalyze, setTickerAndAnalyze } from "@/lib/analysisApi";
import type { AnalysisResult } from "@/lib/analysisTypes";
import { SAMPLE_SIGNAL } from "@/lib/sampleSignal";
import McFanChart from "./McFanChart";
import Select from "./Select";
import StatusBadge from "./StatusBadge";

export default function SignalPanel() {
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "live" | "offline">("idle");
  const [banner, setBanner] = useState("");
  const [tickerInput, setTickerInput] = useState("");

  const load = useCallback(async (fn: () => Promise<AnalysisResult>) => {
    setStatus("loading");
    setBanner("");
    try {
      const d = await fn();
      setData(d);
      setStatus("live");
    } catch (err) {
      setData(SAMPLE_SIGNAL);
      setStatus("offline");
      setBanner(`Could not reach API (${(err as Error).message}). Showing sample data.`);
    }
  }, []);

  useEffect(() => {
    load(fetchSignal);
  }, [load]);

  const d = data;
  const horizons = d ? projectHorizons(d) : [];
  const dotClass =
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
          Signal &amp; Monte Carlo
          {d && (
            <span className="ml-2 text-[13px] font-normal text-muted">
              {d.ticker} / {d.interval} / {label(d.mc_model)}
            </span>
          )}
        </h1>
        <div className="flex-1" />
        <StatusBadge status={status} tone={dotClass} />
        <Select
          value={d?.interval ?? "15m"}
          onChange={(v) => load(() => setConfigAndAnalyze({ interval: v }))}
          options={INTERVALS.map((i) => ({ value: i, label: i }))}
          disabled={status === "loading"}
          title="timeframe"
          width={84}
        />
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

      {banner && (
        <div className="mb-3 rounded-lg border border-[#5a3d12] bg-[#21170a] px-3 py-2 text-[12px] text-gold">
          {banner}
        </div>
      )}

      {!d ? (
        <div className="rounded-xl border border-line bg-panel p-10 text-center text-muted">
          Loading...
        </div>
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat name="price" value={`$${d.current_price.toFixed(2)}`} />
            <Stat
              name="signal"
              value={displayText(d.signal.label)}
              tone={d.signal.composite > 0.05 ? "up" : d.signal.composite < -0.05 ? "down" : "muted"}
              sub={`${label("conf")} ${(d.signal.confidence * 100).toFixed(0)}%`}
            />
            <Stat
              name="mc prob up"
              value={`${d.mc.prob_up.toFixed(1)}%`}
              tone={d.mc.prob_up >= 50 ? "up" : "down"}
              sub={`${label("exp")} ${d.mc.expected_return >= 0 ? "+" : ""}${d.mc.expected_return.toFixed(1)}%`}
            />
            <Stat name="regime" value={label(d.regime.regime)} sub={`${label("hurst")} ${d.regime.hurst.toFixed(2)}`} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="signal">
              <Gauge value={d.signal.composite} />
              <Labeled name="confidence">
                <Bar pct={d.signal.confidence * 100} color="#58a6ff" />
                <span className="ml-2 text-[12px] text-muted">{(d.signal.confidence * 100).toFixed(0)}%</span>
              </Labeled>
              <p className="mt-2 text-[12px] leading-relaxed text-muted">{displayText(d.signal.reasoning)}</p>
              {Object.keys(d.signal.sub_scores || {}).length > 0 && (
                <div className="mt-3 space-y-1.5">
                  {Object.entries(d.signal.sub_scores).map(([k, v]) => (
                    <SubScore key={k} name={k} value={v} />
                  ))}
                </div>
              )}
            </Card>

            <section className="panel">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-[11px] font-semibold tracking-wide text-dim">Monte Carlo</h2>
                <Select
                  value={d.mc.model}
                  onChange={(v) => load(() => setConfigAndAnalyze({ mc_model: v }))}
                  options={MC_MODELS.map((m) => ({ value: m.value, label: label(m.label) }))}
                  disabled={status === "loading"}
                  title="mc model"
                  align="right"
                  width={148}
                />
              </div>

              <McFanChart mc={d.mc} spot={d.current_price} interval={d.interval} />

              <div className="mt-3">
                <ProbBar up={d.mc.prob_up} flat={d.mc.prob_flat} down={d.mc.prob_down} />
              </div>

              <div className="mt-3 grid grid-cols-3 gap-3">
                <Big name="expected" value={`$${d.mc.expected_price.toFixed(2)}`} sub={`${d.mc.expected_return >= 0 ? "+" : ""}${d.mc.expected_return.toFixed(1)}%`} tone={d.mc.expected_return >= 0 ? "up" : "down"} />
                <Big name="median" value={`$${d.mc.median_price.toFixed(2)}`} />
                <Big name="cvar 5%" value={`${d.mc.cvar_5.toFixed(1)}%`} tone="down" />
              </div>

              <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-[12px]">
                <KV k="P25-P75" v={`$${d.mc.p25_price.toFixed(2)} - $${d.mc.p75_price.toFixed(2)}`} />
                <KV k="P10-P90" v={`$${d.mc.p10_price.toFixed(2)} - $${d.mc.p90_price.toFixed(2)}`} />
              </div>

              <div className="mt-4">
                <div className="mb-1.5 text-[10.5px] font-semibold tracking-wide text-dim">
                  {label("projected price")}
                </div>
                <div className="grid grid-cols-3 gap-3">
                  {horizons.map((h) => (
                    <div key={h.label} className="rounded-lg border border-line bg-[#0d1217] p-2.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] text-muted">{h.label}</span>
                        <span className={"text-[11px] font-semibold tnum " + (h.ret >= 0 ? "text-up" : "text-down")}>
                          {h.ret >= 0 ? "+" : ""}{h.ret.toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-0.5 text-[15px] font-semibold tnum">${h.median.toFixed(2)}</div>
                      <div className="text-[10.5px] tnum text-dim">${h.lo.toFixed(2)} - ${h.hi.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              </div>

              <p className="mt-2 text-[10.5px] text-dim">
                Horizon prices extrapolate {label(d.mc.model)} drift and vol. SE: up +/-{d.mc.prob_up_se.toFixed(1)}, down +/-{d.mc.prob_down_se.toFixed(1)}.
              </p>
            </section>

            <Card title="regime">
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-[14px] font-semibold">{label(d.regime.regime)}</span>
                <span className="text-[12px] text-muted">{displayText(d.regime.verdict)}</span>
              </div>
              <PotentialBar up={d.regime.potential_up} flat={d.regime.potential_flat} down={d.regime.potential_down} />
              <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1.5 text-[12px] sm:grid-cols-3">
                <KV k="trend score" v={d.regime.trend_score.toFixed(2)} tone={d.regime.trend_score >= 0 ? "up" : "down"} />
                <KV k="range score" v={d.regime.range_score.toFixed(2)} />
                <KV k="hurst" v={d.regime.hurst.toFixed(2)} />
                <KV k="donchian pos" v={d.regime.donchian_pos.toFixed(2)} />
                <KV k="compression" v={d.regime.range_compression.toFixed(2)} />
                <KV k="HH/HL/LH/LL" v={`${d.regime.hh_count}/${d.regime.hl_count}/${d.regime.lh_count}/${d.regime.ll_count}`} />
              </div>
              {(d.regime.breakout_up || d.regime.breakout_down) && (
                <div className="mt-2">
                  <span className={"rounded-md px-2 py-0.5 text-[11px] font-semibold " + (d.regime.breakout_up ? "bg-up/15 text-up" : "bg-down/15 text-down")}>
                    {d.regime.breakout_up ? "Breakout up" : "Breakout down"}
                  </span>
                </div>
              )}
            </Card>

            <Card title="trade setup">
              {d.trade_setup?.valid ? (
                <>
                  <div className="mb-2">
                    <span className={"rounded-md px-2.5 py-1 text-[12px] font-bold " + (d.trade_setup.side === "long" ? "bg-up/15 text-up" : d.trade_setup.side === "short" ? "bg-down/15 text-down" : "bg-[#161c23] text-muted")}>
                      {label(d.trade_setup.side)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-[12px] sm:grid-cols-4">
                    {d.trade_setup.entry != null && <KV k="entry" v={`$${Number(d.trade_setup.entry).toFixed(2)}`} />}
                    {d.trade_setup.stop != null && <KV k="stop" v={`$${Number(d.trade_setup.stop).toFixed(2)}`} tone="down" />}
                    {d.trade_setup.target != null && <KV k="target" v={`$${Number(d.trade_setup.target).toFixed(2)}`} tone="up" />}
                    {d.trade_setup.rr != null && <KV k="R:R" v={`${Number(d.trade_setup.rr).toFixed(2)}`} />}
                  </div>
                  {d.trade_setup.reason && <p className="mt-2 text-[12px] text-muted">{displayText(d.trade_setup.reason)}</p>}
                </>
              ) : (
                <p className="text-[12px] text-muted">No valid setup. {d.trade_setup?.reason ? displayText(d.trade_setup.reason) : ""}</p>
              )}
              {d.warnings?.length > 0 && (
                <p className="mt-2 text-[12px] text-gold">Warnings: {d.warnings.map(displayText).join("; ")}</p>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <h2 className="mb-3 text-[11px] font-semibold tracking-wide text-dim">{label(title)}</h2>
      {children}
    </section>
  );
}

function Stat({ name, value, sub, tone }: { name: string; value: string; sub?: string; tone?: "up" | "down" | "muted" }) {
  const c = tone === "up" ? "text-up" : tone === "down" ? "text-down" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-panel p-3">
      <div className="text-[11px] tracking-wide text-dim">{label(name)}</div>
      <div className={"mt-1 text-[18px] font-semibold tnum " + c}>{value}</div>
      {sub && <div className="text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

function KV({ k, v, tone }: { k: string; v: string; tone?: "up" | "down" }) {
  const c = tone === "up" ? "text-up" : tone === "down" ? "text-down" : "text-ink";
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-muted">{label(k)}</span>
      <span className={"font-medium tnum " + c}>{v}</span>
    </div>
  );
}

function Labeled({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div className="mt-3 flex items-center">
      <span className="w-24 text-[12px] text-muted">{label(name)}</span>
      <span className="flex flex-1 items-center">{children}</span>
    </div>
  );
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <span className="h-2 flex-1 overflow-hidden rounded bg-[#1c242c]">
      <span className="block h-full rounded" style={{ width: `${Math.max(0, Math.min(100, pct))}%`, background: color }} />
    </span>
  );
}

function Gauge({ value }: { value: number }) {
  const pct = ((Math.max(-1, Math.min(1, value)) + 1) / 2) * 100;
  return (
    <div>
      <div className="relative h-2.5 w-full overflow-hidden rounded bg-gradient-to-r from-down via-[#2a323b] to-up">
        <span
          className="absolute top-1/2 h-4 w-[2px] -translate-x-1/2 -translate-y-1/2 rounded bg-ink"
          style={{ left: `${pct}%` }}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[10px] text-dim">
        <span>Bearish</span>
        <span className="text-ink">{label("composite")} <b className="tnum">{value.toFixed(2)}</b></span>
        <span>Bullish</span>
      </div>
    </div>
  );
}

function SubScore({ name, value }: { name: string; value: number }) {
  const pct = ((Math.max(-1, Math.min(1, value)) + 1) / 2) * 100;
  const pos = value >= 0;
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-20 text-muted">{label(name)}</span>
      <span className="relative h-1.5 flex-1 rounded bg-[#1c242c]">
        <span className="absolute top-0 h-full w-px bg-dim" style={{ left: "50%" }} />
        <span
          className="absolute top-0 h-full rounded"
          style={{
            background: pos ? "#3fb950" : "#f0556d",
            left: pos ? "50%" : `${pct}%`,
            width: `${Math.abs(pct - 50)}%`,
          }}
        />
      </span>
      <span className="w-10 text-right tnum text-muted">{value.toFixed(2)}</span>
    </div>
  );
}

function ProbBar({ up, flat, down }: { up: number; flat: number; down: number }) {
  return (
    <div>
      <div className="flex h-5 overflow-hidden rounded">
        <span className="flex items-center justify-center bg-up/80 text-[10px] font-bold text-black" style={{ width: `${up}%` }}>{up >= 8 ? `${up.toFixed(0)}%` : ""}</span>
        <span className="flex items-center justify-center bg-[#1c242c] text-[10px] text-muted" style={{ width: `${flat}%` }}>{flat >= 8 ? `${flat.toFixed(0)}%` : ""}</span>
        <span className="flex items-center justify-center bg-down/80 text-[10px] font-bold text-black" style={{ width: `${down}%` }}>{down >= 8 ? `${down.toFixed(0)}%` : ""}</span>
      </div>
      <div className="mt-1.5 flex justify-between text-[10px] text-dim">
        <span className="text-up">Up {up.toFixed(1)}%</span>
        <span>Flat {flat.toFixed(1)}%</span>
        <span className="text-down">Down {down.toFixed(1)}%</span>
      </div>
    </div>
  );
}

function PotentialBar({ up, flat, down }: { up: number; flat: number; down: number }) {
  return (
    <div>
      <div className="flex h-4 overflow-hidden rounded">
        <span className="bg-up/70" style={{ width: `${up}%` }} />
        <span className="bg-[#1c242c]" style={{ width: `${flat}%` }} />
        <span className="bg-down/70" style={{ width: `${down}%` }} />
      </div>
      <div className="mt-1.5 flex justify-between text-[10px] text-dim">
        <span className="text-up">{up.toFixed(0)} Up</span>
        <span>Flat {flat.toFixed(0)}</span>
        <span className="text-down">{down.toFixed(0)} Down</span>
      </div>
    </div>
  );
}

function Big({ name, value, sub, tone }: { name: string; value: string; sub?: string; tone?: "up" | "down" }) {
  const c = tone === "up" ? "text-up" : tone === "down" ? "text-down" : "text-ink";
  return (
    <div className="rounded-lg border border-line bg-[#0d1217] p-2.5">
      <div className="text-[10.5px] tracking-wide text-dim">{label(name)}</div>
      <div className={"mt-1 text-[16px] font-bold tnum " + c}>{value}</div>
      {sub && <div className={"text-[11px] tnum " + c}>{sub}</div>}
    </div>
  );
}

function barsPerDay(interval: string): number {
  switch (interval) {
    case "1d": return 1;
    case "4h": return 2;
    case "1h": return 7;
    case "30m": return 13;
    case "15m": return 26;
    case "5m": return 78;
    case "2m": return 195;
    case "1m": return 390;
    default: return 1;
  }
}

interface Horizon { label: string; days: number; median: number; lo: number; hi: number; ret: number }

function projectHorizons(d: AnalysisResult): Horizon[] {
  const spot = d.current_price || 0;
  const nf = Math.max(1, (d.mc.median_path?.length || 2) - 1);
  if (spot <= 0) return [];
  const muStep = Math.log((d.mc.expected_price || spot) / spot) / nf;
  const p75 = d.mc.p75_price || spot;
  const p25 = d.mc.p25_price || spot;
  const sigmaTotal = Math.max(1e-6, Math.log(Math.max(p75, 1e-6) / Math.max(p25, 1e-6)) / (2 * 0.6745));
  const sigmaStep = sigmaTotal / Math.sqrt(nf);
  const bpd = barsPerDay(d.interval);

  const project = (days: number) => {
    const steps = days * bpd;
    const median = spot * Math.exp(muStep * steps);
    const band = 0.6745 * sigmaStep * Math.sqrt(steps);
    return { median, lo: median * Math.exp(-band), hi: median * Math.exp(band), ret: (median / spot - 1) * 100 };
  };

  return [
    { label: "1d", days: 1 },
    { label: "3d", days: 3 },
    { label: "1w", days: 5 },
  ].map((h) => ({ ...h, ...project(h.days) }));
}
