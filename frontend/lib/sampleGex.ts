import type { OptionsFlow } from "./gexTypes";

const spot = 62.0;
const profile = [];
for (let k = 50; k <= 75; k += 1) {
  // synthetic: positive GEX above spot, negative below, peaks near walls
  const dist = k - spot;
  const base = Math.exp(-((dist) ** 2) / 30) * 9e8;
  const gex = (dist >= 0 ? 1 : -1) * base * (0.6 + Math.random() * 0.8);
  profile.push({
    strike: k,
    gex: Math.round(gex),
    call_oi: Math.round(2000 + Math.max(0, dist) * 400 + Math.random() * 1500),
    put_oi: Math.round(2000 + Math.max(0, -dist) * 420 + Math.random() * 1500),
  });
}

export const SAMPLE_GEX: OptionsFlow = {
  ticker: "PLTR",
  expiry: "2026-06-19",
  spot,
  max_pain: 60,
  call_wall: 70,
  put_wall: 55,
  gamma_flip: 61.2,
  net_gex: profile.reduce((s, b) => s + b.gex, 0),
  gex_positive: true,
  days_to_expiry: 20,
  gex_profile: profile,
  error: null,
};
