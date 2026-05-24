"""
core/options_flow.py — Options flow: Max Pain + Gamma Exposure (GEX).

Fetches the options chain (yfinance, free) and computes:

  max_pain      : Strike where total option-buyer loss is maximised.
                  Price gravitates here into expiry (market-maker incentive).

  net_gex       : Total dealer net gamma exposure across all strikes.
                  Positive  → dealers are long gamma → they BUY dips, SELL rips
                              → volatility is DAMPED, price tends to pin.
                  Negative  → dealers are short gamma → they SELL dips, BUY rips
                              → volatility is AMPLIFIED, moves accelerate.

  gamma_flip    : The price level where net GEX crosses zero.
                  Above gamma flip  →  volatility suppression (positive GEX).
                  Below gamma flip  →  volatility amplification (negative GEX).

  call_wall     : Strike with highest call OI (strong resistance ceiling).
  put_wall      : Strike with highest put OI (strong support floor).

  gex_profile   : Per-strike GEX bars for charting.

Why this helps predict price reaction at demand/supply zones
------------------------------------------------------------
  • Zone at / above call_wall → bounce is harder; expect compression or failure.
  • Zone at / below put_wall  → bounce is stronger; dealers must buy there too.
  • Zone between gamma_flip and call_wall (positive GEX band) → pinning / chop.
  • Zone below gamma_flip      → directional moves; zone may not hold.
  • Max pain acts as a gravity attractor going into weekly/monthly expiry.

Black-Scholes gamma (closed-form):
  d1    = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)
  Gamma = N'(d1) / (S · σ · √T)
  GEX_k = OI_calls(k)·Γ(k)·S²·100 - OI_puts(k)·Γ(k)·S²·100
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ─── Sector mapping ───────────────────────────────────────────────────────────
# Used to tag each UnusualOption hit so the frontend can filter by sector.
_SECTOR_MAP: dict[str, str] = {
    # Mega-cap / Mixed-use tech
    "AAPL": "Tech",    "MSFT": "Tech",    "GOOGL": "Tech",   "GOOG": "Tech",
    "META": "Tech",    "AMZN": "Tech",    "NFLX": "Media",
    # Semiconductors
    "NVDA": "Semis",   "AMD":  "Semis",   "INTC": "Semis",   "QCOM": "Semis",
    "AMAT": "Semis",   "MU":   "Semis",   "MRVL": "Semis",   "ARM":  "Semis",
    "TSM":  "Semis",   "ASML": "Semis",   "AVGO": "Semis",   "TXN":  "Semis",
    "SMCI": "Semis",   "KLAC": "Semis",   "LRCX": "Semis",   "ADI":  "Semis",
    "QRVO": "Semis",   "SWKS": "Semis",   "MPWR": "Semis",   "ON":   "Semis",
    "WOLF": "Semis",   "ACLS": "Semis",   "COHU": "Semis",
    # Software / Cloud
    "CRM":  "Software","ORCL": "Software","NOW":  "Software","SNOW": "Software",
    "PLTR": "Software","CRWD": "Software","PANW": "Software","ZS":   "Software",
    "DDOG": "Software","NET":  "Software","MDB":  "Software","OKTA": "Software",
    "ZM":   "Software","DOCU": "Software","TWLO": "Software","HUBS": "Software",
    "SHOP": "Software","ADBE": "Software","INTU": "Software","SAP":  "Software",
    "ACN":  "Software","IBM":  "Software","CSCO": "Software","ANET": "Software",
    "FFIV": "Software","CIEN": "Software","PSTG": "Software",
    # AI / Quantum / Blockchain
    "MSTR": "Crypto",  "IONQ": "Quantum", "RGTI": "Quantum", "QUBT": "Quantum",
    "BBAI": "AI",      "SOUN": "AI",      "AI":   "AI",
    # FinTech / Payments
    "V":    "Finance", "MA":   "Finance", "PYPL": "FinTech", "SQ":   "FinTech",
    "COIN": "Crypto",  "HOOD": "FinTech", "SOFI": "FinTech", "AFRM": "FinTech",
    "ENVA": "FinTech", "LC":   "FinTech", "UPST": "FinTech",
    # Financials
    "JPM":  "Finance", "GS":   "Finance", "MS":   "Finance", "BAC":  "Finance",
    "C":    "Finance", "WFC":  "Finance", "BX":   "Finance", "KKR":  "Finance",
    "APO":  "Finance", "CG":   "Finance", "ARES": "Finance", "OWL":  "Finance",
    "BRK-B":"Finance", "AXP":  "Finance", "COF":  "Finance", "SYF":  "Finance",
    "USB":  "Finance", "PNC":  "Finance", "TFC":  "Finance", "STT":  "Finance",
    "SCHW": "Finance", "CME":  "Finance", "ICE":  "Finance", "CBOE": "Finance",
    "MSCI": "Finance", "MCO":  "Finance", "FIS":  "Finance", "FISV": "Finance",
    "GPN":  "Finance", "AIG":  "Finance", "MET":  "Finance", "PRU":  "Finance",
    "AFL":  "Finance", "TRV":  "Finance", "CB":   "Finance", "ALL":  "Finance",
    "HIG":  "Finance", "SPGI": "Finance", "VRT":  "Finance",
    # Biotech / Pharma
    "MRNA": "Biotech", "BNTX": "Biotech", "GILD": "Biotech", "VRTX": "Biotech",
    "BIIB": "Biotech", "REGN": "Biotech", "AMGN": "Biotech", "ALNY": "Biotech",
    "BMRN": "Biotech", "SRPT": "Biotech", "EXEL": "Biotech", "ACAD": "Biotech",
    "INCY": "Biotech", "RARE": "Biotech", "BEAM": "Biotech", "NTLA": "Biotech",
    "CRSP": "Biotech", "EDIT": "Biotech", "RXRX": "Biotech", "FATE": "Biotech",
    "KYMR": "Biotech", "IMVT": "Biotech", "ABBV": "Pharma",  "LLY":  "Pharma",
    "PFE":  "Pharma",  "MRK":  "Pharma",  "AZN":  "Pharma",  "NVO":  "Pharma",
    "SNY":  "Pharma",  "GSK":  "Pharma",  "BMY":  "Pharma",
    "JNJ":  "Health",  "UNH":  "Health",  "CVS":  "Health",  "HUM":  "Health",
    "CNC":  "Health",  "MOH":  "Health",  "ELV":  "Health",  "CI":   "Health",
    "HCA":  "Health",  "THC":  "Health",  "UHS":  "Health",  "CLOV": "Health",
    "ISRG": "MedTech", "EW":   "MedTech", "DXCM": "MedTech", "PODD": "MedTech",
    "HOLX": "MedTech", "IDXX": "MedTech", "SYK":  "MedTech", "ZTS":  "MedTech",
    "BSX":  "MedTech", "MDT":  "MedTech", "ABT":  "MedTech", "TMO":  "MedTech",
    "DHR":  "MedTech", "A":    "MedTech", "BIO":  "MedTech", "IQV":  "MedTech",
    "RVTY": "MedTech", "PRCT": "MedTech", "NTRA": "MedTech",
    # Energy
    "XOM":  "Energy",  "CVX":  "Energy",  "COP":  "Energy",  "OXY":  "Energy",
    "DVN":  "Energy",  "FANG": "Energy",  "HAL":  "Energy",  "SLB":  "Energy",
    "BKR":  "Energy",  "NOV":  "Energy",  "HP":   "Energy",  "PTEN": "Energy",
    "LNG":  "Energy",  "CQP":  "Energy",  "ET":   "Energy",  "EPD":  "Energy",
    "MPLX": "Energy",  "WMB":  "Energy",  "OKE":  "Energy",  "KMI":  "Energy",
    "VLO":  "Energy",  "MPC":  "Energy",  "PSX":  "Energy",  "DK":   "Energy",
    "DINO": "Energy",  "EOG":  "Energy",  "APA":  "Energy",  "PR":   "Energy",
    "SM":   "Energy",  "CIVI": "Energy",  "WHD":  "Energy",
    # Consumer / Retail
    "NKE":  "Consumer","LULU": "Consumer","RL":   "Consumer","PVH":  "Consumer",
    "HBI":  "Consumer","UAA":  "Consumer","UA":   "Consumer","TGT":  "Consumer",
    "LOW":  "Consumer","BBY":  "Consumer","BBWI": "Consumer","M":    "Consumer",
    "KSS":  "Consumer","CMG":  "Consumer","YUM":  "Consumer","QSR":  "Consumer",
    "DRI":  "Consumer","TXRH": "Consumer","SHAK": "Consumer","WING": "Consumer",
    "ETSY": "Consumer","CHWY": "Consumer","W":    "Consumer","WMT":  "Consumer",
    "COST": "Consumer","SBUX": "Consumer","MCD":  "Consumer","PG":   "Consumer",
    "KO":   "Consumer","PEP":  "Consumer","MO":   "Consumer","PM":   "Consumer",
    "CL":   "Consumer","GIS":  "Consumer","CPB":  "Consumer","HSY":  "Consumer",
    "MKC":  "Consumer","CHD":  "Consumer","CLX":  "Consumer","SJM":  "Consumer",
    # Auto / EV
    "TSLA": "Auto/EV", "GM":   "Auto/EV", "F":    "Auto/EV", "RIVN": "Auto/EV",
    "LCID": "Auto/EV", "NIO":  "Auto/EV", "XPEV": "Auto/EV", "LI":   "Auto/EV",
    "BLNK": "Auto/EV", "CHPT": "Auto/EV", "EVGO": "Auto/EV",
    # Platforms / Gig
    "UBER": "Platform","LYFT": "Platform","ABNB": "Platform","DASH": "Platform",
    "RBLX": "Platform","SNAP": "Social",  "PINS": "Social",  "SPOT": "Media",
    "MTCH": "Platform","YELP": "Platform","TRIP": "Platform","OPEN": "Platform",
    # Hardware / Storage
    "HPQ":  "Hardware","HPE":  "Hardware","DELL": "Hardware","WDC":  "Hardware",
    "STX":  "Hardware","NTAP": "Hardware","LITE": "Hardware","VIAV": "Hardware",
    # Defense / Aerospace
    "BA":   "Defense", "LMT":  "Defense", "NOC":  "Defense", "GD":   "Defense",
    "RTX":  "Defense", "LHX":  "Defense", "HEI":  "Defense", "TDG":  "Defense",
    "HWM":  "Defense",
    # Industrial / Utilities
    "CAT":  "Industrial","DE":  "Industrial","EMR": "Industrial","ROK": "Industrial",
    "PH":   "Industrial","ITW": "Industrial","GWW": "Industrial","CMI": "Industrial",
    "ETN":  "Industrial","DOV": "Industrial","UPS": "Industrial","FDX": "Industrial",
    "NSC":  "Industrial","MMM": "Industrial","HON": "Industrial",
    "NEE":  "Utilities", "SO":  "Utilities", "DUK": "Utilities", "AEP": "Utilities",
    "EXC":  "Utilities", "D":   "Utilities", "SRE": "Utilities", "PCG": "Utilities",
    "PEG":  "Utilities", "ECL": "Materials", "APD": "Materials", "DD":  "Materials",
    "PPG":  "Materials", "SHW": "Materials",
    # Airlines / Transport / Travel
    "AAL":  "Transport","DAL":  "Transport","UAL":  "Transport","LUV":  "Transport",
    "JBLU": "Transport","ALK":  "Transport","CCL":  "Travel",   "RCL":  "Travel",
    "NCLH": "Travel",   "BKNG": "Travel",
    # Media / Telecom
    "DIS":  "Media",   "CMCSA":"Media",    "CHTR": "Media",   "WBD":  "Media",
    "FOXA": "Media",   "FOX":  "Media",   "PARA": "Media",   "SIRI": "Media",
    "NYT":  "Media",   "NWS":  "Media",   "NWSA": "Media",   "IAC":  "Media",
    "T":    "Telecom", "VZ":   "Telecom", "TMUS": "Telecom", "LUMN": "Telecom",
    # REIT / Real Estate
    "AMT":  "REIT",    "CCI":  "REIT",    "SBAC": "REIT",    "EQIX": "REIT",
    "DLR":  "REIT",    "PLD":  "REIT",    "SPG":  "REIT",    "O":    "REIT",
    "WPC":  "REIT",    "NNN":  "REIT",    "VICI": "REIT",    "VNQ":  "ETF",
    # Gaming / Casinos
    "MGM":  "Gaming",  "LVS":  "Gaming",  "WYNN": "Gaming",  "CZR":  "Gaming",
    "DKNG": "Gaming",  "PENN": "Gaming",
    # Materials / Mining
    "GOLD": "Mining",  "NEM":  "Mining",  "AEM":  "Mining",  "WPM":  "Mining",
    "PAAS": "Mining",  "AG":   "Mining",  "BHP":  "Mining",  "RIO":  "Mining",
    "VALE": "Mining",  "MT":   "Materials","STLD": "Materials","NUE":  "Materials",
    "X":    "Materials","CLF":  "Materials","AA":   "Materials","CF":   "Materials",
    "MOS":  "Materials","FCX":  "Mining",
    # Crypto / Blockchain miners
    "MARA": "Crypto",  "RIOT": "Crypto",  "HUT":  "Crypto",  "CLSK": "Crypto",
    "BTBT": "Crypto",  "CIFR": "Crypto",
    # Meme / Speculative
    "GME":  "Meme",    "AMC":  "Meme",    "SPCE": "Meme",    "NKLA": "Meme",
    "WKHS": "Meme",
    # Cannabis
    "TLRY": "Cannabis","CGC":  "Cannabis","ACB":  "Cannabis","SNDL": "Cannabis",
    # Space / Emerging
    "ASTS": "SpaceTech","RKLB":"SpaceTech",
    # ETFs
    "SPY":  "ETF", "QQQ":  "ETF", "IWM":  "ETF", "DIA":  "ETF", "VTI":  "ETF",
    "VOO":  "ETF", "TQQQ": "ETF", "SQQQ": "ETF", "GLD":  "ETF", "GDX":  "ETF",
    "GDXJ": "ETF", "SLV":  "ETF", "USO":  "ETF", "UNG":  "ETF", "TLT":  "ETF",
    "IEF":  "ETF", "SHY":  "ETF", "HYG":  "ETF", "LQD":  "ETF", "JNK":  "ETF",
    "XLE":  "ETF", "XLF":  "ETF", "XLK":  "ETF", "XLV":  "ETF", "XLI":  "ETF",
    "XLU":  "ETF", "XLP":  "ETF", "XLB":  "ETF", "XLRE": "ETF", "EEM":  "ETF",
    "EFA":  "ETF", "FXI":  "ETF", "EWZ":  "ETF", "EWJ":  "ETF", "KWEB": "ETF",
    "ARKK": "ETF", "ARKG": "ETF", "ARKW": "ETF", "ARKF": "ETF", "REET": "ETF",
    "KBWB": "ETF", "KRE":  "ETF", "XHB":  "ETF", "XRT":  "ETF", "IBB":  "ETF",
    "XBI":  "ETF", "SMH":  "ETF", "SOXX": "ETF", "VXX":  "ETF", "UVXY": "ETF",
    "SVXY": "ETF", "TPVG": "ETF", "GLAD": "ETF", "PSEC": "ETF",
}

# ─── Cache (avoid hammering yfinance) ────────────────────────────────────────
_cache_lock = threading.RLock()
_cache: dict = {}  # key → (result, expire_time)
_CACHE_TTL = 300.0  # 5 minutes

# ─── yfinance rate-limit guard ────────────────────────────────────────────────
# Limits the number of simultaneous yfinance HTTP connections across all threads.
# Even when the executor has N workers, only _YF_SEM_SLOTS connections run at once.
_YF_SEM_SLOTS = 4
_yf_semaphore = threading.Semaphore(_YF_SEM_SLOTS)

_RATE_LIMIT_PHRASES = ("too many requests", "rate limit", "429")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RATE_LIMIT_PHRASES)


def _yf_call(fn, *args, retries: int = 3, base_delay: float = 2.0, **kwargs):
    """
    Call a yfinance function under the global semaphore with exponential-backoff
    retry on rate-limit errors.

    Parameters
    ----------
    fn          : callable — the yfinance method to call
    *args       : positional args forwarded to fn
    retries     : max number of retry attempts after the first failure
    base_delay  : seconds to wait before the first retry (doubles each time)
    **kwargs    : keyword args forwarded to fn
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.2, 1.0)
            logger.debug("[yf_call] rate-limited, retry %d/%d in %.1fs", attempt, retries, delay)
            time.sleep(delay)

        with _yf_semaphore:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit(exc) and attempt < retries:
                    continue   # retry after backoff
                raise          # non-rate-limit error or out of retries

    raise last_exc  # type: ignore[misc]


def _cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.monotonic() < entry[1]:
            return entry[0]
    return None


def _cache_put(key, value):
    with _cache_lock:
        _cache[key] = (value, time.monotonic() + _CACHE_TTL)


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class GEXBar:
    strike: float
    gex: float  # net GEX at this strike (calls positive, puts negative)
    call_oi: int
    put_oi: int


@dataclass
class OptionsFlow:
    ticker: str
    expiry: str  # expiry date used (YYYY-MM-DD)
    spot: float
    max_pain: float
    call_wall: float  # strike with peak call OI
    put_wall: float  # strike with peak put OI
    gamma_flip: float  # price where net GEX ≈ 0
    net_gex: float  # total dealer net GEX
    gex_positive: bool  # True = volatility-damping regime
    days_to_expiry: float
    gex_profile: list[GEXBar]  # sorted by strike
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "expiry": self.expiry,
            "spot": round(self.spot, 4),
            "max_pain": round(self.max_pain, 4),
            "call_wall": round(self.call_wall, 4),
            "put_wall": round(self.put_wall, 4),
            "gamma_flip": round(self.gamma_flip, 4),
            "net_gex": round(self.net_gex, 2),
            "gex_positive": self.gex_positive,
            "days_to_expiry": round(self.days_to_expiry, 1),
            "error": self.error,
            "gex_profile": [
                {
                    "strike": round(b.strike, 4),
                    "gex": round(b.gex, 2),
                    "call_oi": b.call_oi,
                    "put_oi": b.put_oi,
                }
                for b in self.gex_profile
            ],
        }


# ─── Black-Scholes helpers ────────────────────────────────────────────────────


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.05) -> float:
    """
    Black-Scholes gamma.  Returns 0 on bad inputs.
    S = spot, K = strike, T = time to expiry in years, sigma = implied vol.
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0


# ─── Max Pain ─────────────────────────────────────────────────────────────────


def _compute_max_pain(calls_df, puts_df) -> float:
    """
    Max pain = strike that minimises total payout to option holders.

    Total payout at expiry price P:
      Σ_K [OI_call(K) · max(P−K, 0) + OI_put(K) · max(K−P, 0)] × 100

    We evaluate over all strikes in the chain.
    """
    all_strikes = sorted(set(list(calls_df["strike"].values) + list(puts_df["strike"].values)))
    if not all_strikes:
        return 0.0

    call_oi = {float(r["strike"]): _safe_int(r["openInterest"]) for _, r in calls_df.iterrows()}
    put_oi  = {float(r["strike"]): _safe_int(r["openInterest"]) for _, r in puts_df.iterrows()}

    min_pain = float("inf")
    max_pain_strike = all_strikes[len(all_strikes) // 2]

    for P in all_strikes:
        total = 0.0
        for K in all_strikes:
            total += call_oi.get(K, 0) * max(P - K, 0.0)
            total += put_oi.get(K, 0) * max(K - P, 0.0)
        if total < min_pain:
            min_pain = total
            max_pain_strike = P

    return float(max_pain_strike)


# ─── GEX profile ─────────────────────────────────────────────────────────────


def _compute_gex_profile(
    calls_df,
    puts_df,
    spot: float,
    T: float,
    risk_free: float = 0.05,
    max_bars: int = 50,
) -> list[GEXBar]:
    """
    For each strike compute net GEX:
      GEX_net(K) = (OI_call(K) - OI_put(K)) · Γ(S, K, T, σ) · S² · 100

    We use each option's own implied vol (IV) from the chain.
    Falls back to average IV if individual IV is missing.

    Limits output to max_bars bars centered around spot price
    to reduce response size and keep chart legible.
    """
    import pandas as pd

    # Merge calls + puts on strike
    cols = ["strike", "openInterest", "impliedVolatility"]
    c = calls_df[cols].copy().rename(columns={"openInterest": "call_oi", "impliedVolatility": "call_iv"})
    p = puts_df[cols].copy().rename(columns={"openInterest": "put_oi", "impliedVolatility": "put_iv"})

    merged = pd.merge(c, p, on="strike", how="outer").fillna(0)
    # fillna(0) handles NaN from outer join; replace any residual NaN/Inf safely
    merged["call_oi"] = merged["call_oi"].apply(lambda x: _safe_int(x))
    merged["put_oi"]  = merged["put_oi"].apply(lambda x: _safe_int(x))
    merged["call_iv"] = merged["call_iv"].apply(lambda x: _safe_float(x))
    merged["put_iv"]  = merged["put_iv"].apply(lambda x: _safe_float(x))

    # Fallback average IV
    avg_iv = float(merged[["call_iv", "put_iv"]].replace(0, float("nan")).stack().mean()) or 0.3

    bars: list[GEXBar] = []
    for _, row in merged.iterrows():
        K = float(row["strike"])
        c_iv = float(row["call_iv"]) or avg_iv
        p_iv = float(row["put_iv"]) or avg_iv
        c_oi = _safe_int(row["call_oi"])
        p_oi = _safe_int(row["put_oi"])

        g_call = _bs_gamma(spot, K, T, c_iv, risk_free)
        g_put = _bs_gamma(spot, K, T, p_iv, risk_free)

        # Dealers who sold calls are long delta → they are short gamma (negative GEX from calls)
        # Convention: call GEX positive (dealers short vol), put GEX negative
        gex = (c_oi * g_call - p_oi * g_put) * spot * spot * 100.0

        bars.append(GEXBar(strike=K, gex=gex, call_oi=c_oi, put_oi=p_oi))

    bars.sort(key=lambda b: b.strike)

    # ── Limit to most relevant strikes (centered around spot) ──────────────────
    if len(bars) > max_bars:
        # Find spot index
        spot_idx = min(range(len(bars)), key=lambda i: abs(bars[i].strike - spot))
        # Calculate window around spot
        window_half = max_bars // 2
        start = max(0, spot_idx - window_half)
        end = min(len(bars), start + max_bars)
        # Adjust start if we're near the end
        if end - start < max_bars:
            start = max(0, end - max_bars)
        bars = bars[start:end]

    return bars


def _find_gamma_flip(bars: list[GEXBar], spot: float) -> float:
    """
    Find the strike closest to where cumulative net GEX crosses zero
    (scanning from spot downward).
    """
    # Only consider strikes ≤ spot (below spot = where gamma flip matters most)
    below = [(b.strike, b.gex) for b in bars if b.strike <= spot * 1.05]
    if not below:
        return spot

    # Cumulative GEX from spot downward
    below.sort(key=lambda x: -x[0])  # high to low
    cum = 0.0
    prev_strike = spot
    for strike, gex in below:
        prev_cum = cum
        cum += gex
        if prev_cum > 0 and cum <= 0:
            # crossed zero between prev_strike and strike — interpolate
            frac = abs(prev_cum) / (abs(prev_cum) + abs(cum) + 1e-12)
            return prev_strike - frac * (prev_strike - strike)
        prev_strike = strike
    return below[-1][0]  # no flip found — return lowest strike


# ─── Public entry point ───────────────────────────────────────────────────────


def fetch_options_flow(ticker: str, spot: float | None = None) -> OptionsFlow:
    """
    Fetch options chain and compute GEX + Max Pain.
    Returns an OptionsFlow with error set if anything fails.
    Cached for 5 minutes.
    """
    key = ticker.upper()
    cached = _cache_get(key)
    if cached is not None:
        return cached

    result = _fetch(key, spot)
    _cache_put(key, result)
    return result


def _fetch(ticker: str, spot_override: float | None) -> OptionsFlow:
    _empty = OptionsFlow(
        ticker=ticker,
        expiry="",
        spot=spot_override or 0.0,
        max_pain=0.0,
        call_wall=0.0,
        put_wall=0.0,
        gamma_flip=0.0,
        net_gex=0.0,
        gex_positive=True,
        days_to_expiry=0.0,
        gex_profile=[],
        error="not_fetched",
    )
    try:
        from datetime import datetime, timezone

        import yfinance as yf

        t = yf.Ticker(ticker)

        # Get spot price
        if spot_override and spot_override > 0:
            spot = float(spot_override)
        else:
            info = t.fast_info
            spot = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None) or 0.0
            if spot <= 0:
                hist = t.history(period="1d", interval="1m")
                spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if spot <= 0:
            _empty.error = "no_spot_price"
            return _empty

        # Pick nearest expiry ≥ 7 days out (avoid pin risk noise on 0DTE)
        exps = t.options
        if not exps:
            _empty.error = "no_options_chain"
            return _empty

        now = datetime.now(timezone.utc).date()
        chosen_exp = None
        for exp in exps:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (d - now).days
            if dte >= 5:
                chosen_exp = exp
                break
        if chosen_exp is None:
            chosen_exp = exps[0]

        exp_date = datetime.strptime(chosen_exp, "%Y-%m-%d").date()
        T = max((exp_date - now).days, 1) / 365.0

        chain = t.option_chain(chosen_exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        if calls.empty or puts.empty:
            _empty.error = "empty_chain"
            return _empty

        # Focus on strikes within ±20% of spot (filter out deep OTM noise)
        lo, hi = spot * 0.80, spot * 1.20
        calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
        puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]

        if calls.empty or puts.empty:
            _empty.error = "no_near_strikes"
            return _empty

        # ── Max Pain ──────────────────────────────────────────────────────
        max_pain = _compute_max_pain(calls, puts)

        # ── Call Wall / Put Wall ──────────────────────────────────────────
        c_oi_max_idx = calls["openInterest"].fillna(0).astype(int).idxmax()
        p_oi_max_idx = puts["openInterest"].fillna(0).astype(int).idxmax()
        call_wall = float(calls.loc[c_oi_max_idx, "strike"])
        put_wall = float(puts.loc[p_oi_max_idx, "strike"])

        # ── GEX profile ───────────────────────────────────────────────────
        gex_profile = _compute_gex_profile(calls, puts, spot, T)
        net_gex = sum(b.gex for b in gex_profile)
        gamma_flip = _find_gamma_flip(gex_profile, spot)

        result = OptionsFlow(
            ticker=ticker,
            expiry=chosen_exp,
            spot=spot,
            max_pain=max_pain,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_flip=gamma_flip,
            net_gex=net_gex,
            gex_positive=(net_gex >= 0),
            days_to_expiry=T * 365,
            gex_profile=gex_profile,
            error=None,
        )
        logger.info(
            "[OptionsFlow] %s  spot=%.2f  max_pain=%.2f  call_wall=%.2f  "
            "put_wall=%.2f  gamma_flip=%.2f  net_gex=%.0f  exp=%s",
            ticker,
            spot,
            max_pain,
            call_wall,
            put_wall,
            gamma_flip,
            net_gex,
            chosen_exp,
        )
        return result

    except ImportError:
        _empty.error = "yfinance_not_installed"
        return _empty
    except Exception as exc:
        logger.warning("[OptionsFlow] %s failed: %s", ticker, exc)
        _empty.error = str(exc)[:120]
        return _empty


# ─── Unusual Options Activity Scanner ────────────────────────────────────────
#
# Detects statistically unusual options activity across a list of tickers.
#
# Signals scored (each 0–1, combined into a composite "unusual_score"):
#   vol_oi_ratio   — contract Volume / Open Interest > threshold (new money flowing in)
#   iv_spike       — implied vol significantly above the chain average (fear / conviction)
#   otm_bias       — large volume on deep OTM strikes (speculative / event-driven bets)
#   premium_size   — notional premium value (contracts × 100 × mid-price) in top decile
#   cp_divergence  — call/put volume ratio deviates far from 1 (directional sweep)
#
# Each hit is tagged with a human-readable "flag" list and a directional
# "sentiment": "bullish" | "bearish" | "mixed".


@dataclass
class UnusualOption:
    """A single unusual options contract."""
    ticker: str
    expiry: str
    strike: float
    option_type: str          # "call" | "put"
    volume: int
    open_interest: int
    vol_oi_ratio: float
    implied_vol: float
    avg_chain_iv: float       # average IV across the chain for context
    in_the_money: bool
    premium_per_contract: float   # mid-price × 100
    total_premium: float          # volume × premium_per_contract
    unusual_score: float          # 0–1 composite
    flags: list                   # e.g. ["high_vol_oi", "iv_spike", "otm_sweep"]
    sentiment: str                # "bullish" | "bearish" | "mixed"
    spot: float
    days_to_expiry: float
    percent_change: float = 0.0   # option contract % price change today
    sector: str = "Other"         # sector label from _SECTOR_MAP

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "expiry": self.expiry,
            "strike": round(self.strike, 4),
            "option_type": self.option_type,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "vol_oi_ratio": round(self.vol_oi_ratio, 2),
            "implied_vol": round(self.implied_vol * 100, 2),   # as %
            "avg_chain_iv": round(self.avg_chain_iv * 100, 2),
            "in_the_money": self.in_the_money,
            "percent_change": round(self.percent_change, 2),
            "sector": self.sector,
            "premium_per_contract": round(self.premium_per_contract, 2),
            "total_premium": round(self.total_premium, 2),
            "unusual_score": round(self.unusual_score, 4),
            "flags": self.flags,
            "sentiment": self.sentiment,
            "spot": round(self.spot, 4),
            "days_to_expiry": round(self.days_to_expiry, 1),
        }


def _safe_int(val, default: int = 0) -> int:
    """Convert val to int, returning default for None / NaN / Inf."""
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else int(f)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    """Convert val to float, returning default for None / NaN / Inf."""
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _scan_ticker_unusual(
    ticker: str,
    min_volume: int = 10,
    min_oi: int = 10,
    vol_oi_threshold: float = 3.0,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    max_dte: int = 60,
    min_premium: float = 0.0,
    new_positions_only: bool = False,
) -> list[UnusualOption] | None:
    """
    Fetch options chain for one ticker and return unusual contracts.

    Returns
    -------
    list[UnusualOption]  — zero or more hits (empty = scanned, nothing unusual)
    None                 — ticker appears delisted / has no valid market data

    Parameters
    ----------
    ticker              : stock symbol
    min_volume          : ignore contracts with volume below this (sanity floor)
    min_oi              : ignore contracts with OI below this
    vol_oi_threshold    : flag contract when volume / OI > this multiple
    iv_spike_z          : flag IV when it exceeds chain_mean + z * chain_std
    otm_pct             : flag as OTM sweep when |strike/spot - 1| > this fraction
    max_dte             : only scan expiries within this many calendar days
    min_premium         : skip contracts where total_premium < this value (USD)
    new_positions_only  : if True, only include contracts where volume > open_interest
    """
    import statistics
    from datetime import datetime, timezone

    import yfinance as yf

    results: list[UnusualOption] = []
    try:
        t = yf.Ticker(ticker.upper())

        # Spot price — guarded by semaphore + retry
        info = _yf_call(lambda: t.fast_info)
        spot = float(getattr(info, "last_price", None) or getattr(info, "regular_market_price", None) or 0.0)
        if spot <= 0:
            # No price at all — check options to distinguish delisted vs. closed market
            try:
                exps = _yf_call(lambda: t.options)
            except Exception:
                exps = []
            if not exps:
                logger.info("[UnusualOptions] %s: no price + no options chain — likely delisted/invalid", ticker)
                return None   # signal "delisted" to caller
            return []   # has options but no live price — data gap, not delisted

        now_date = datetime.now(timezone.utc).date()

        # Collect expiries within max_dte — guarded
        all_exps = _yf_call(lambda: t.options) or []
        near_exps = []
        for exp in all_exps:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (d - now_date).days
                if 0 <= dte <= max_dte:
                    near_exps.append((exp, dte))
            except ValueError:
                pass

        if not near_exps:
            return []

        # Aggregate all contracts across near-term expiries
        all_contracts: list[dict] = []
        for exp, dte in near_exps:
            try:
                chain = _yf_call(t.option_chain, exp)
            except Exception as exc:
                if _is_rate_limit(exc):
                    logger.warning("[UnusualOptions] %s rate-limited on expiry %s (giving up)", ticker, exp)
                    break   # stop fetching more expiries for this ticker rather than silently skip all
                continue

            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    vol    = _safe_int(row.get("volume"))
                    oi     = _safe_int(row.get("openInterest"))
                    iv     = _safe_float(row.get("impliedVolatility"))
                    strike = _safe_float(row.get("strike"))
                    bid    = _safe_float(row.get("bid"))
                    ask    = _safe_float(row.get("ask"))
                    last   = _safe_float(row.get("lastPrice"))
                    # Use bid/ask mid when available; fall back to lastPrice
                    mid    = (bid + ask) / 2.0 if (bid > 0 or ask > 0) else last
                    itm    = bool(row.get("inTheMoney", False))
                    pct_chg = _safe_float(row.get("percentChange"))

                    if vol < min_volume or oi < min_oi or strike <= 0:
                        continue

                    # Premium filter — skip cheap/small contracts early
                    if min_premium > 0 and (vol * mid * 100) < min_premium:
                        continue

                    # New-positions filter — only contracts where vol > OI
                    if new_positions_only and vol <= oi:
                        continue

                    all_contracts.append({
                        "ticker":   ticker.upper(),
                        "expiry":   exp,
                        "dte":      dte,
                        "strike":   strike,
                        "type":     side,
                        "volume":   vol,
                        "oi":       oi,
                        "iv":       iv,
                        "mid":      mid,
                        "itm":      itm,
                        "pct_chg":  pct_chg,
                    })

        if not all_contracts:
            return []

        # Chain-wide IV stats for z-score
        ivs = [c["iv"] for c in all_contracts if c["iv"] > 0]
        avg_iv = statistics.mean(ivs) if ivs else 0.3
        std_iv = statistics.stdev(ivs) if len(ivs) > 1 else 0.05

        # Total call + put volume for CP divergence
        total_call_vol = sum(c["volume"] for c in all_contracts if c["type"] == "call")
        total_put_vol = sum(c["volume"] for c in all_contracts if c["type"] == "put")
        cp_ratio = (total_call_vol / max(total_put_vol, 1))

        # Premium percentile threshold (top-30% = notable)
        premiums = sorted(c["volume"] * c["mid"] * 100 for c in all_contracts)
        prem_threshold = premiums[int(len(premiums) * 0.70)] if premiums else 0

        # ── Signal weights (must sum to 1.0) ──────────────────────────────────
        # cp_divergence is chain-level (same value for every contract on the ticker)
        # so it gets a low weight — prevents all contracts from scoring 100% when
        # the chain happens to be call/put-skewed.
        _SIG_W = {
            "high_vol_oi":   0.35,  # contract-specific — new money flowing in
            "iv_spike":      0.25,  # contract-specific — implied vol conviction
            "otm_sweep":     0.20,  # contract-specific — directional speculation
            "large_premium": 0.15,  # contract-specific — notional size
            "cp_divergence": 0.05,  # chain-level tie-breaker only
        }  # sum = 1.00

        for c in all_contracts:
            vol_oi = c["volume"] / max(c["oi"], 1)
            iv = c["iv"]
            mid = c["mid"]
            total_prem = c["volume"] * mid * 100
            otm_dist = abs(c["strike"] / spot - 1.0)
            is_otm = not c["itm"]

            flags: list[str] = []
            signal_scores: dict[str, float] = {}

            # ① Vol/OI ratio — fresh position opening
            if vol_oi >= vol_oi_threshold:
                flags.append("high_vol_oi")
                signal_scores["high_vol_oi"] = min(vol_oi / (vol_oi_threshold * 3), 1.0)

            # ② IV spike above chain average
            if std_iv > 0 and (iv - avg_iv) / std_iv >= iv_spike_z:
                flags.append("iv_spike")
                signal_scores["iv_spike"] = min((iv - avg_iv) / (std_iv * 3), 1.0)

            # ③ OTM sweep (speculative directional bet)
            if is_otm and otm_dist >= otm_pct and c["volume"] >= min_volume * 2:
                flags.append("otm_sweep")
                signal_scores["otm_sweep"] = min(otm_dist / 0.20, 1.0)

            # ④ Large notional premium
            if total_prem >= prem_threshold and total_prem > 0:
                flags.append("large_premium")
                max_prem = max(premiums[-1], 1)
                signal_scores["large_premium"] = min(total_prem / max_prem, 1.0)

            # ⑤ CP divergence (chain-level, low-weight tie-breaker)
            if cp_ratio > 3.0 or cp_ratio < 0.33:
                flags.append("cp_divergence")
                div = max(cp_ratio, 1.0 / max(cp_ratio, 0.01))
                signal_scores["cp_divergence"] = min((div - 1) / 5.0, 1.0)

            if not flags:
                continue

            # Weighted composite — denominator is always the full weight sum (1.0),
            # so a contract with only cp_divergence maxes at 0.05, not 1.0.
            composite = sum(signal_scores[f] * _SIG_W[f] for f in signal_scores)

            # Sentiment
            if c["type"] == "call":
                sentiment = "bullish"
            elif c["type"] == "put":
                sentiment = "bearish"
            else:
                sentiment = "mixed"
            # Override: if OTM puts dominate (cp_ratio < 0.5), even calls are mixed
            if "cp_divergence" in flags and cp_ratio < 0.5:
                sentiment = "bearish" if c["type"] == "put" else "mixed"

            results.append(UnusualOption(
                ticker=c["ticker"],
                expiry=c["expiry"],
                strike=c["strike"],
                option_type=c["type"],
                volume=c["volume"],
                open_interest=c["oi"],
                vol_oi_ratio=round(vol_oi, 2),
                implied_vol=iv,
                avg_chain_iv=avg_iv,
                in_the_money=c["itm"],
                premium_per_contract=round(mid * 100, 2),
                total_premium=round(total_prem, 2),
                unusual_score=round(composite, 4),
                flags=flags,
                sentiment=sentiment,
                spot=spot,
                days_to_expiry=float(c["dte"]),
                percent_change=c.get("pct_chg", 0.0),
                sector=_SECTOR_MAP.get(ticker.upper(), "Other"),
            ))

    except Exception as exc:
        logger.warning("[UnusualOptions] %s scan failed: %s", ticker, exc)

    # Sort by composite score descending, cap per ticker
    results.sort(key=lambda x: x.unusual_score, reverse=True)
    return results[:20]


def scan_unusual_options(
    tickers: list[str],
    min_volume: int = 10,
    min_oi: int = 10,
    vol_oi_threshold: float = 3.0,
    iv_spike_z: float = 1.5,
    otm_pct: float = 0.05,
    max_dte: int = 60,
    max_concurrent: int = 6,
    top_n: int = 50,
    min_premium: float = 0.0,
    new_positions_only: bool = False,
) -> dict:
    """
    Scan a list of tickers for unusual options activity and return the
    top-N hits ranked by composite unusual score.

    Parameters
    ----------
    tickers          : list of stock symbols to scan
    min_volume       : minimum contract volume to consider
    min_oi           : minimum open interest to consider
    vol_oi_threshold : Vol/OI ratio to flag as unusual (default 3×)
    iv_spike_z       : IV z-score above chain mean to flag as spike
    otm_pct          : fraction OTM to flag as directional sweep
    max_dte          : max calendar days to expiry to include
    max_concurrent   : parallel worker threads
    top_n            : max results returned (across all tickers)

    Returns
    -------
    {
      "hits":    [ UnusualOption.to_dict(), ... ],   # sorted by unusual_score desc
      "summary": { "tickers_scanned", "tickers_with_hits", "total_hits",
                   "bullish_count", "bearish_count", "mixed_count" },
      "scanned_at": ISO-8601
    }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone

    tickers = [t.upper().strip() for t in tickers if t.strip()]
    if not tickers:
        return {"hits": [], "summary": {}, "scanned_at": datetime.now(timezone.utc).isoformat()}

    all_hits: list[UnusualOption] = []
    tickers_with_hits: set[str] = set()
    delisted_tickers: set[str] = set()

    logger.info("[UnusualOptions] scanning %d tickers (max_concurrent=%d)", len(tickers), max_concurrent)

    # Stagger worker start times so threads don't all hit yfinance at t=0.
    # Each worker sleeps a random jitter before its first network call.
    _stagger_lock = threading.Lock()
    _stagger_counter = [0]

    def _worker(tkr: str) -> list[UnusualOption] | None:
        # Staggered start: worker N sleeps N * 0.15 s (+ jitter) before touching yfinance
        with _stagger_lock:
            idx = _stagger_counter[0]
            _stagger_counter[0] += 1
        time.sleep(idx * 0.15 + random.uniform(0.0, 0.25))
        return _scan_ticker_unusual(
            tkr,
            min_volume=min_volume,
            min_oi=min_oi,
            vol_oi_threshold=vol_oi_threshold,
            iv_spike_z=iv_spike_z,
            otm_pct=otm_pct,
            max_dte=max_dte,
            min_premium=min_premium,
            new_positions_only=new_positions_only,
        )

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_map = {pool.submit(_worker, tkr): tkr for tkr in tickers}
        for fut in as_completed(future_map):
            tkr = future_map[fut]
            try:
                hits = fut.result()
                if hits is None:
                    # None = delisted / no valid market data
                    delisted_tickers.add(tkr)
                elif hits:
                    all_hits.extend(hits)
                    tickers_with_hits.add(tkr)
            except Exception as exc:
                if _is_rate_limit(exc):
                    logger.warning("[UnusualOptions] worker %s rate-limited (all retries exhausted)", tkr)
                else:
                    logger.warning("[UnusualOptions] worker %s raised: %s", tkr, exc)

    # Global sort and cap
    all_hits.sort(key=lambda x: x.unusual_score, reverse=True)
    top_hits = all_hits[:top_n]

    bullish = sum(1 for h in top_hits if h.sentiment == "bullish")
    bearish = sum(1 for h in top_hits if h.sentiment == "bearish")
    mixed   = sum(1 for h in top_hits if h.sentiment == "mixed")

    if delisted_tickers:
        logger.info(
            "[UnusualOptions] %d tickers appear delisted/invalid: %s",
            len(delisted_tickers), ", ".join(sorted(delisted_tickers)),
        )

    logger.info(
        "[UnusualOptions] scan complete — %d hits from %d/%d tickers  bull=%d bear=%d mixed=%d  delisted=%d",
        len(top_hits), len(tickers_with_hits), len(tickers), bullish, bearish, mixed, len(delisted_tickers),
    )

    return {
        "hits": [h.to_dict() for h in top_hits],
        "summary": {
            "tickers_scanned": len(tickers),
            "tickers_with_hits": len(tickers_with_hits),
            "total_hits": len(top_hits),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "mixed_count": mixed,
            "delisted_count": len(delisted_tickers),
            "delisted_tickers": sorted(delisted_tickers),
        },
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }
