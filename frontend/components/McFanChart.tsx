"use client";

import type { MC } from "@/lib/analysisTypes";
import { label } from "@/lib/display";

type Pt = { x: number; y: number };

function segs(p: Pt[]): string {
  let d = "";
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = p[i - 1] || p[i];
    const p1 = p[i];
    const p2 = p[i + 1];
    const p3 = p[i + 2] || p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += `C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)} `;
  }
  return d.trim();
}
const smooth = (p: Pt[]) => `M${p[0].x},${p[0].y} ${segs(p)}`;
const band = (top: Pt[], bot: Pt[]) => {
  const r = [...bot].reverse();
  return `${smooth(top)} L${r[0].x},${r[0].y} ${segs(r)} Z`;
};

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

const HORIZONS = [
  { label: "1d", days: 1 },
  { label: "3d", days: 3 },
  { label: "1w", days: 5 },
];

export default function McFanChart({
  mc,
  spot,
  interval = "1d",
}: {
  mc: MC;
  spot: number;
  interval?: string;
}) {
  if (!spot || spot <= 0) {
    return <div className="text-[12px] text-muted">No projection data.</div>;
  }

  const nf = Math.max(1, (mc.median_path?.length || 2) - 1);
  const muStep = Math.log((mc.expected_price || spot) / spot) / nf;
  const p75 = mc.p75_price || spot;
  const p25 = mc.p25_price || spot;
  const sigmaTotal = Math.max(1e-6, Math.log(Math.max(p75, 1e-6) / Math.max(p25, 1e-6)) / (2 * 0.6745));
  const sigmaStep = sigmaTotal / Math.sqrt(nf);
  const bpd = barsPerDay(interval);

  const maxDays = HORIZONS[HORIZONS.length - 1].days;
  const project = (days: number) => {
    const steps = days * bpd;
    const median = spot * Math.exp(muStep * steps);
    const halfBand = 0.6745 * sigmaStep * Math.sqrt(Math.max(steps, 0));
    return { median, hi: median * Math.exp(halfBand), lo: median * Math.exp(-halfBand) };
  };

  const SAMPLES = 48;
  const days: number[] = Array.from({ length: SAMPLES + 1 }, (_, i) => (i / SAMPLES) * maxDays);
  const medArr = days.map((t) => project(t).median);
  const hiArr = days.map((t) => project(t).hi);
  const loArr = days.map((t) => project(t).lo);

  const W = 560;
  const H = 230;
  const padX = 6;
  const padR = 56;
  const padY = 16;

  let lo = Infinity;
  let hi = -Infinity;
  for (const v of [...hiArr, ...loArr, spot]) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo || 1;

  const x = (t: number) => padX + (t / maxDays) * (W - padX - padR);
  const y = (v: number) => padY + (1 - (v - lo) / span) * (H - 2 * padY);
  const toPts = (arr: number[]) => arr.map((v, i) => ({ x: x(days[i]), y: y(v) }));

  const medPts = toPts(medArr);
  const hiPts = toPts(hiArr);
  const loPts = toPts(loArr);

  const horizonNodes = HORIZONS.map((h) => {
    const p = project(h.days);
    return { ...h, ...p, px: x(h.days), py: y(p.median) };
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <defs>
        <linearGradient id="mcInner" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#58a6ff" stopOpacity="0.22" />
          <stop offset="100%" stopColor="#58a6ff" stopOpacity="0.02" />
        </linearGradient>
      </defs>

      <path d={band(hiPts, loPts)} fill="url(#mcInner)" />

      <line x1={padX} x2={x(maxDays)} y1={y(spot)} y2={y(spot)} stroke="#7d8b99" strokeWidth={1} strokeDasharray="4 4" opacity={0.7} />
      <text x={padX + 2} y={y(spot) - 4} fontSize={8.5} fill="#7d8b99">{label("spot")} ${spot.toFixed(2)}</text>

      <path d={smooth(medPts)} fill="none" stroke="#58a6ff" strokeWidth={2.2} strokeLinecap="round" />

      {horizonNodes.map((h) => {
        const up = h.median >= spot;
        const dotColor = up ? "#3fb950" : "#f0556d";
        const labelAbove = h.py > padY + 26;
        return (
          <g key={h.label}>
            <line x1={h.px} x2={h.px} y1={y(spot)} y2={h.py} stroke={dotColor} strokeWidth={1} strokeDasharray="3 3" opacity={0.45} />
            <circle cx={h.px} cy={h.py} r={3.4} fill={dotColor} stroke="#0b0e11" strokeWidth={1} />
            <text
              x={h.px}
              y={labelAbove ? h.py - 9 : h.py + 16}
              fontSize={10}
              fontWeight={700}
              fill={dotColor}
              textAnchor="middle"
            >
              ${h.median.toFixed(2)}
            </text>
            <text
              x={h.px}
              y={labelAbove ? h.py - 19 : h.py + 25}
              fontSize={8}
              fill="#7d8b99"
              textAnchor="middle"
            >
              {h.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
