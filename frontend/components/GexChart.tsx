"use client";

import { useRef, useState } from "react";
import type { OptionsFlow } from "@/lib/gexTypes";

const W = 920;
const H = 340;
const L = 10;
const R = 10;
const T = 18;
const B = 30;

export default function GexChart({ flow }: { flow: OptionsFlow }) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const bars = [...flow.gex_profile].sort((a, b) => a.strike - b.strike);
  const n = bars.length;
  if (n < 2) return <div className="p-6 text-center text-[12px] text-muted">No gamma profile data.</div>;

  const minK = bars[0].strike;
  const maxK = bars[n - 1].strike;
  const plotTop = T;
  const plotBot = H - B;
  const midY = (plotTop + plotBot) / 2;
  const half = (plotBot - plotTop) / 2 - 2;
  const maxAbs = Math.max(...bars.map((b) => Math.abs(b.gex)), 1);

  const cw = (W - L - R) / n;
  const x = (i: number) => L + (i + 0.5) * cw;
  const barW = Math.max(1.5, cw * 0.68);
  const xForPrice = (p: number) =>
    L + ((p - minK) / (maxK - minK || 1)) * (W - L - R);

  const labelEvery = Math.max(1, Math.ceil(n / 14));

  function onMove(e: React.MouseEvent) {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.floor((px - L) / cw);
    setHover(i >= 0 && i < n ? i : null);
  }

  const marker = (price: number, color: string, label: string) => {
    if (!(price > 0) || price < minK || price > maxK) return null;
    const mx = xForPrice(price);
    return (
      <g>
        <line x1={mx} x2={mx} y1={plotTop - 6} y2={plotBot} stroke={color} strokeWidth={1} strokeDasharray="5 3" opacity={0.85} />
        <text x={mx} y={plotTop - 8} fontSize={9} fill={color} textAnchor="middle">{label}</text>
      </g>
    );
  };

  return (
    <div className="relative">
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="w-full" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {/* y reference labels */}
        <text x={L + 2} y={plotTop + 8} fontSize={9} fill="#5a6673">+{(maxAbs / 1e6).toFixed(0)}M</text>
        <text x={L + 2} y={plotBot - 2} fontSize={9} fill="#5a6673">-{(maxAbs / 1e6).toFixed(0)}M</text>

        {/* zero line */}
        <line x1={L} x2={W - R} y1={midY} y2={midY} stroke="#2a323b" strokeWidth={1} />

        {/* bars */}
        {bars.map((b, i) => {
          const pos = b.gex >= 0;
          const h = (Math.abs(b.gex) / maxAbs) * half;
          const bx = x(i) - barW / 2;
          const by = pos ? midY - h : midY;
          return (
            <g key={i}>
              <rect
                x={bx}
                y={Math.max(0.5, by)}
                width={barW}
                height={Math.max(0.5, h)}
                fill={pos ? "rgba(63,185,80,0.8)" : "rgba(240,85,109,0.8)"}
              />
              {i % labelEvery === 0 && (
                <text x={x(i)} y={plotBot + 12} fontSize={8.5} fill="#5a6673" textAnchor="middle">
                  {b.strike}
                </text>
              )}
            </g>
          );
        })}

        {/* hover highlight */}
        {hover != null && bars[hover] && (
          <line x1={x(hover)} x2={x(hover)} y1={plotTop} y2={plotBot} stroke="#7d8b99" strokeWidth={1} strokeDasharray="3 3" opacity={0.6} />
        )}

        {/* markers */}
        {marker(flow.spot, "#e6edf3", "Spot")}
        {marker(flow.gamma_flip, "#f2c14e", "γ-flip")}
        {marker(flow.max_pain, "#bc6bd9", "Max pain")}
      </svg>

      {hover != null && bars[hover] && (
        <div className="pointer-events-none absolute left-2 top-2 rounded-md border border-line bg-[#0d1217]/95 px-2.5 py-1.5 text-[11px] tnum">
          <div className="text-ink">Strike {bars[hover].strike}</div>
          <div className="mt-0.5 text-muted">
            GEX <b className={bars[hover].gex >= 0 ? "text-up" : "text-down"}>{(bars[hover].gex / 1e6).toFixed(1)}M</b>
            {"  "}· call OI {bars[hover].call_oi.toLocaleString()} · put OI {bars[hover].put_oi.toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
}
