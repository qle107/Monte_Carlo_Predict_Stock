"use client";

import { useMemo, useRef, useState } from "react";
import type { Candle } from "@/lib/analysisTypes";
import { label } from "@/lib/display";
import { bollinger, ema, vwap } from "@/lib/indicators";

export interface ChartLevel {
  price: number;
  label: string;
  color: string;
}

export interface Overlays {
  ema9: boolean;
  ema21: boolean;
  ema200: boolean;
  bb: boolean;
  vwap: boolean;
}

const W = 1000;
const PL = 8;
const PR = 58;
const PT = 12;
const PRICE_H = 330;
const VOL_T = PT + PRICE_H + 12;
const VOL_H = 64;
const H = VOL_T + VOL_H + 4;

function poly(xs: (i: number) => number, ys: (v: number) => number, vals: number[]): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (Number.isNaN(v)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)} `;
    pen = true;
  }
  return d.trim();
}

export default function PriceChart({
  candles,
  overlays,
  levels = [],
}: {
  candles: Candle[];
  overlays: Overlays;
  levels?: ChartLevel[];
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const closes = useMemo(() => candles.map((c) => c.c), [candles]);
  const e9 = useMemo(() => ema(closes, 9), [closes]);
  const e21 = useMemo(() => ema(closes, 21), [closes]);
  const e200 = useMemo(() => ema(closes, 200), [closes]);
  const bb = useMemo(() => bollinger(closes, 20, 2), [closes]);
  const vw = useMemo(() => vwap(candles), [candles]);

  const n = candles.length;
  const cw = (W - PL - PR) / Math.max(n, 1);

  const { pMin, pMax, vMax } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    let vm = 0;
    for (const c of candles) {
      if (c.l < lo) lo = c.l;
      if (c.h > hi) hi = c.h;
      if (c.v > vm) vm = c.v;
    }
    if (overlays.bb) {
      for (let i = 0; i < n; i++) {
        if (!Number.isNaN(bb.upper[i]) && bb.upper[i] > hi) hi = bb.upper[i];
        if (!Number.isNaN(bb.lower[i]) && bb.lower[i] < lo) lo = bb.lower[i];
      }
    }
    for (const lv of levels) {
      if (lv.price < lo) lo = lv.price;
      if (lv.price > hi) hi = lv.price;
    }
    const pad = (hi - lo) * 0.05 || 1;
    return { pMin: lo - pad, pMax: hi + pad, vMax: vm || 1 };
  }, [candles, overlays.bb, bb, levels, n]);

  const x = (i: number) => PL + (i + 0.5) * cw;
  const yP = (v: number) => PT + (1 - (v - pMin) / (pMax - pMin)) * PRICE_H;
  const yV = (v: number) => VOL_T + VOL_H - (v / vMax) * VOL_H;

  const gridVals = useMemo(() => {
    const out: number[] = [];
    const steps = 5;
    for (let i = 0; i <= steps; i++) out.push(pMin + ((pMax - pMin) * i) / steps);
    return out;
  }, [pMin, pMax]);

  function onMove(e: React.MouseEvent) {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width; // 0..1 across svg
    const px = fx * W;
    const i = Math.round((px - PL) / cw - 0.5);
    setHover(i >= 0 && i < n ? i : null);
  }

  const hc = hover != null ? candles[hover] : null;

  return (
    <div className="relative">
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {gridVals.map((v, i) => (
          <g key={i}>
            <line x1={PL} x2={W - PR} y1={yP(v)} y2={yP(v)} stroke="#141a20" strokeWidth={1} />
            <text x={W - PR + 4} y={yP(v) + 3} fontSize={9} fill="#5a6673">
              {v.toFixed(2)}
            </text>
          </g>
        ))}

        {overlays.bb && (
          <>
            <path d={poly(x, yP, bb.upper)} fill="none" stroke="rgba(88,166,255,0.40)" strokeWidth={1} />
            <path d={poly(x, yP, bb.mid)} fill="none" stroke="rgba(88,166,255,0.25)" strokeWidth={1} strokeDasharray="4 3" />
            <path d={poly(x, yP, bb.lower)} fill="none" stroke="rgba(88,166,255,0.40)" strokeWidth={1} />
          </>
        )}

        {candles.map((c, i) => {
          const up = c.c >= c.o;
          const col = up ? "#3fb950" : "#f0556d";
          const bx = x(i);
          const bw = Math.max(1, cw * 0.62);
          const yo = yP(c.o);
          const ycl = yP(c.c);
          const top = Math.min(yo, ycl);
          const hgt = Math.max(1, Math.abs(yo - ycl));
          return (
            <g key={i}>
              <line x1={bx} x2={bx} y1={yP(c.h)} y2={yP(c.l)} stroke={col} strokeWidth={1} />
              <rect x={bx - bw / 2} y={top} width={bw} height={hgt} fill={col} />
              <rect
                x={bx - bw / 2}
                y={yV(c.v)}
                width={bw}
                height={VOL_T + VOL_H - yV(c.v)}
                fill={up ? "rgba(63,185,80,0.35)" : "rgba(240,85,109,0.35)"}
              />
            </g>
          );
        })}

        {overlays.vwap && <path d={poly(x, yP, vw)} fill="none" stroke="#bc6bd9" strokeWidth={1.4} />}
        {overlays.ema9 && <path d={poly(x, yP, e9)} fill="none" stroke="#f2c14e" strokeWidth={1.4} />}
        {overlays.ema21 && <path d={poly(x, yP, e21)} fill="none" stroke="#58a6ff" strokeWidth={1.4} />}
        {overlays.ema200 && <path d={poly(x, yP, e200)} fill="none" stroke="#e6edf3" strokeWidth={1.2} opacity={0.7} />}

        {levels.map((lv, i) => (
          <g key={i}>
            <line x1={PL} x2={W - PR} y1={yP(lv.price)} y2={yP(lv.price)} stroke={lv.color} strokeWidth={1} strokeDasharray="5 3" opacity={0.85} />
            <text x={PL + 3} y={yP(lv.price) - 3} fontSize={9} fill={lv.color}>
              {label(lv.label)} {lv.price.toFixed(2)}
            </text>
          </g>
        ))}

        {hc && hover != null && (
          <>
            <line x1={x(hover)} x2={x(hover)} y1={PT} y2={VOL_T + VOL_H} stroke="#7d8b99" strokeWidth={1} strokeDasharray="3 3" />
            <line x1={PL} x2={W - PR} y1={yP(hc.c)} y2={yP(hc.c)} stroke="#7d8b99" strokeWidth={1} strokeDasharray="3 3" />
          </>
        )}
      </svg>

      {hc && (
        <div className="pointer-events-none absolute left-2 top-2 rounded-md border border-line bg-[#0d1217]/95 px-2.5 py-1.5 text-[11px] tnum">
          <div className="text-dim">{new Date(hc.t).toLocaleString()}</div>
          <div className="mt-0.5 flex gap-3">
            <span>o <b className="text-ink">{hc.o.toFixed(2)}</b></span>
            <span>h <b className="text-up">{hc.h.toFixed(2)}</b></span>
            <span>l <b className="text-down">{hc.l.toFixed(2)}</b></span>
            <span>c <b className={hc.c >= hc.o ? "text-up" : "text-down"}>{hc.c.toFixed(2)}</b></span>
            <span className="text-muted">v {hc.v.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
