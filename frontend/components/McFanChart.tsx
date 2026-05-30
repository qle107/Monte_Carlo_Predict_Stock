"use client";

import type { MC } from "@/lib/analysisTypes";

type Pt = { x: number; y: number };

// Catmull-Rom → cubic Bézier smoothing (segments only, assumes pen at pts[0]).
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

export default function McFanChart({ mc, spot }: { mc: MC; spot: number }) {
  const med = mc.median_path || [];
  const n = med.length;
  if (n < 2) return <div className="text-[12px] text-muted">No simulation path data.</div>;

  const W = 560;
  const H = 230;
  const padX = 6;
  const padR = 56;
  const padY = 14;

  const all = [mc.p10_band, mc.p90_band, mc.lower_band, mc.upper_band, med, [spot]];
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of all) for (const v of s) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = hi - lo || 1;

  const x = (i: number) => padX + (i / (n - 1)) * (W - padX - padR);
  const y = (v: number) => padY + (1 - (v - lo) / span) * (H - 2 * padY);
  const pts = (arr: number[]) => arr.slice(0, n).map((v, i) => ({ x: x(i), y: y(v) }));

  const lastX = x(n - 1);
  const upPrice = mc.p90_band[n - 1] ?? mc.upper_band[n - 1] ?? med[n - 1];
  const midPrice = med[n - 1];
  const dnPrice = mc.p10_band[n - 1] ?? mc.lower_band[n - 1] ?? med[n - 1];

  const node = (price: number, color: string, label: string) => {
    const ny = y(price);
    return (
      <g>
        <line x1={lastX} x2={lastX + 8} y1={ny} y2={ny} stroke={color} strokeWidth={1} opacity={0.6} />
        <circle cx={lastX} cy={ny} r={3.2} fill={color} stroke="#0b0e11" strokeWidth={1} />
        <text x={lastX + 11} y={ny - 2} fontSize={10} fontWeight={700} fill={color}>${price.toFixed(2)}</text>
        <text x={lastX + 11} y={ny + 8} fontSize={8} fill="#7d8b99">{label}</text>
      </g>
    );
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <defs>
        <linearGradient id="mcOuter" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#58a6ff" stopOpacity="0.04" />
          <stop offset="100%" stopColor="#58a6ff" stopOpacity="0.16" />
        </linearGradient>
        <linearGradient id="mcInner" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#58a6ff" stopOpacity="0.10" />
          <stop offset="100%" stopColor="#58a6ff" stopOpacity="0.30" />
        </linearGradient>
        <linearGradient id="mcLine" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#388bfd" />
          <stop offset="100%" stopColor="#7cc7ff" />
        </linearGradient>
        <filter id="mcGlow" x="-20%" y="-40%" width="140%" height="180%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* outer P10–P90 band */}
      <path d={band(pts(mc.p90_band), pts(mc.p10_band))} fill="url(#mcOuter)" />
      {/* inner P25–P75 band */}
      <path d={band(pts(mc.upper_band), pts(mc.lower_band))} fill="url(#mcInner)" />

      {/* spot reference */}
      <line x1={padX} x2={lastX} y1={y(spot)} y2={y(spot)} stroke="#7d8b99" strokeWidth={1} strokeDasharray="4 4" opacity={0.7} />
      <text x={padX + 2} y={y(spot) - 4} fontSize={8.5} fill="#7d8b99">spot ${spot.toFixed(2)}</text>

      {/* median line: soft glow + crisp gradient stroke */}
      <path d={smooth(pts(med))} fill="none" stroke="#58a6ff" strokeWidth={3} opacity={0.35} filter="url(#mcGlow)" />
      <path d={smooth(pts(med))} fill="none" stroke="url(#mcLine)" strokeWidth={2.2} strokeLinecap="round" />

      {/* end-of-horizon price nodes (up / median / down) */}
      {node(upPrice, "#3fb950", "P90")}
      {node(midPrice, "#7cc7ff", "median")}
      {node(dnPrice, "#f0556d", "P10")}
    </svg>
  );
}
