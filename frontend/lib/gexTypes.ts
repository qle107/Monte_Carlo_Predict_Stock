export interface GexBar {
  strike: number;
  gex: number; // net dealer gamma exposure at this strike
  call_oi: number;
  put_oi: number;
}

export interface OptionsFlow {
  ticker: string;
  expiry: string;
  spot: number;
  max_pain: number;
  call_wall: number; // strike of peak call OI
  put_wall: number; // strike of peak put OI
  gamma_flip: number; // price where net GEX ~ 0
  net_gex: number;
  gex_positive: boolean; // true = vol-damping regime
  days_to_expiry: number;
  gex_profile: GexBar[];
  error: string | null;
}
