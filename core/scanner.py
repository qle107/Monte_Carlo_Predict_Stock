"""Multi-ticker breakout and breakdown scanner."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from config import cfg

from .expected_move import compute_expected_move, expected_move_for_ticker
from .fetcher import fetch_candles
from .indicators import compute_indicators
from .montecarlo import run as run_mc
from .regime import detect_regime
from .signal import compute_signal

logger = logging.getLogger(__name__)

# Predefined watchlists

WATCHLISTS: dict[str, list[str]] = {
    # Default scan universe for /api/options/unusual
    "all_optionable": [
        # Mega-cap / S&P 500 core
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "GOOG",
        "META",
        "TSLA",
        "BRK-B",
        "UNH",
        "LLY",
        "JPM",
        "V",
        "XOM",
        "MA",
        "AVGO",
        "PG",
        "HD",
        "COST",
        "MRK",
        "ABBV",
        "CVX",
        "CRM",
        "AMD",
        "PEP",
        "KO",
        "ADBE",
        "TMO",
        "ACN",
        "MCD",
        "BAC",
        "NFLX",
        "CSCO",
        "DHR",
        "LIN",
        "ABT",
        "WMT",
        "TXN",
        "INTC",
        "NEE",
        "DIS",
        "QCOM",
        "PM",
        "AMGN",
        "UPS",
        "INTU",
        "RTX",
        "CAT",
        "AMAT",
        "BKNG",
        "GS",
        "IBM",
        "MS",
        "BLK",
        "SPGI",
        "AXP",
        "HON",
        "DE",
        "MMC",
        "MO",
        "USB",
        "PNC",
        "TJX",
        "SBUX",
        "MDT",
        "GILD",
        "ADI",
        "REGN",
        "VRTX",
        "BSX",
        "SYK",
        "ZTS",
        "CI",
        "HUM",
        "CVS",
        "MCK",
        "WM",
        "FDX",
        "NSC",
        "SO",
        "DUK",
        "AEP",
        "EXC",
        "D",
        "SRE",
        "PCG",
        "PEG",
        "CL",
        "GIS",
        "CPB",
        "HSY",
        "MKC",
        "CHD",
        "CLX",
        "SJM",
        "EMR",
        "ROK",
        "PH",
        "ITW",
        "GWW",
        "CMI",
        "ETN",
        "DOV",
        "ECL",
        "APD",
        "DD",
        "PPG",
        "SHW",
        "NEM",
        "FCX",
        "NUE",
        # X removed: US Steel acquired by Nippon Steel, delisted Jun 2025.
        "CLF",
        "AA",
        "CF",
        "MOS",
        "AIG",
        "MET",
        "PRU",
        "AFL",
        "TRV",
        "CB",
        "ALL",
        "HIG",
        "WFC",
        "C",
        "TFC",
        "STT",
        "SCHW",
        "COF",
        "SYF",
        "CME",
        "ICE",
        "CBOE",
        "MSCI",
        "MCO",
        "FIS",
        "FISV",
        "GPN",
        # Technology
        "ORCL",
        "SAP",
        "NOW",
        "SNOW",
        "PLTR",
        "CRWD",
        "PANW",
        "ZS",
        "DDOG",
        "NET",
        "MDB",
        "OKTA",
        "ZM",
        "DOCU",
        "TWLO",
        "HUBS",
        "SHOP",
        "PYPL",
        "COIN",
        "HOOD",
        "SOFI",
        "AFRM",
        "XYZ",
        "PSTG",
        "MRVL",
        "KLAC",
        "LRCX",
        "ASML",
        "TSM",
        "MU",
        "SMCI",
        "ARM",
        "QRVO",
        "SWKS",
        "MPWR",
        "ON",
        "WOLF",
        "ACLS",
        "COHU",
        "UBER",
        "LYFT",
        "ABNB",
        "DASH",
        "RBLX",
        "SNAP",
        "PINS",
        "SPOT",
        "MTCH",
        "IAC",
        "YELP",
        "TRIP",
        "HPQ",
        "HPE",
        "DELL",
        "WDC",
        "STX",
        "NTAP",
        "FFIV",
        "ANET",
        "CIEN",
        "LITE",
        "VIAV",
        # Semiconductors / Quantum
        "MSTR",
        "IONQ",
        "RGTI",
        "QUBT",
        # Biotech / Healthcare
        "MRNA",
        "BNTX",
        "BIIB",
        "ALNY",
        "BMRN",
        "SRPT",
        "EXEL",
        "ACAD",
        "INCY",
        "RARE",
        "BEAM",
        "NTLA",
        "CRSP",
        "EDIT",
        "RXRX",
        "SEER",
        "FATE",
        "KYMR",
        "IMVT",
        "CLOV",
        "PFE",
        "JNJ",
        "AZN",
        "NVO",
        "SNY",
        "GSK",
        "RVTY",
        "ISRG",
        "EW",
        "DXCM",
        "PODD",
        "HOLX",
        "IDXX",
        "IQV",
        "A",
        "BIO",
        "MTD",
        "TECH",
        "NTRA",
        "HCA",
        "THC",
        "UHS",
        "CNC",
        "MOH",
        "ELV",
        # Financials
        "BX",
        "KKR",
        "APO",
        "CG",
        "ARES",
        "OWL",
        "VRT",
        "ENVA",
        "LC",
        "UPST",
        "OPEN",
        # Energy
        "COP",
        "OXY",
        "DVN",
        "FANG",
        "HAL",
        "SLB",
        # TE removed: TECO Energy delisted 2016 (acquired by Emera) - invalid symbol.
        # PXD acquired by ExxonMobil (2023). MRO acquired by COP (2024).
        # CLR taken private (2022). HFC renamed to DINO.
        "DINO",
        "BKR",
        "NOV",
        "HP",
        "PTEN",
        "WHD",
        "LNG",
        "CQP",
        "ET",
        "EPD",
        "MPLX",
        "WMB",
        "OKE",
        "KMI",
        "VLO",
        "MPC",
        "PSX",
        "DK",
        "EOG",
        "APA",
        "PR",
        "SM",
        # Consumer / Retail
        "NKE",
        "LULU",
        "RL",
        "PVH",
        "HBI",
        "UAA",
        "UA",
        "TGT",
        "LOW",
        "BBY",
        "BBWI",
        "M",
        "KSS",
        # JWN taken private (2025). Removed.
        "CMG",
        "YUM",
        "QSR",
        "DRI",
        "TXRH",
        "SHAK",
        "WING",
        "AMZN",
        "ETSY",
        "CHWY",
        "W",
        "GM",
        "F",
        "RIVN",
        "LCID",
        "NIO",
        "XPEV",
        "LI",
        # FSR bankrupt (2024). HYLN renamed/delisted. Removed.
        # Real estate / Industrials
        "AMT",
        "CCI",
        "SBAC",
        "EQIX",
        "DLR",
        "PLD",
        "SPG",
        "O",
        "WPC",
        "NNN",
        "VICI",
        "MGM",
        "LVS",
        "WYNN",
        "CZR",
        "BA",
        "LMT",
        "NOC",
        "GD",
        "LHX",
        "HEI",
        "TDG",
        "HWM",
        # L3H is an incorrect ticker; use LHX (L3Harris).
        "AAL",
        "DAL",
        "UAL",
        "LUV",
        "JBLU",
        "ALK",
        "CCL",
        "RCL",
        "NCLH",
        # Media / Telecom
        "CMCSA",
        "CHTR",
        "T",
        "VZ",
        "TMUS",
        "LUMN",
        "FOXA",
        "FOX",
        "WBD",
        "PARA",
        "SIRI",
        # VIAC renamed to PARA (already present). Removed duplicate.
        "NYT",
        "NWS",
        "NWSA",
        # Commodities / Materials
        "GOLD",
        "NEM",
        "AEM",
        "WPM",
        "PAAS",
        "AG",
        "BHP",
        "RIO",
        "VALE",
        "MT",
        "STLD",
        # Popular ETFs with liquid options
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "TQQQ",
        "SQQQ",
        "GLD",
        "GDX",
        "GDXJ",
        "SLV",
        "USO",
        "UNG",
        "TLT",
        "IEF",
        "SHY",
        "HYG",
        "LQD",
        "JNK",
        "XLE",
        "XLF",
        "XLK",
        "XLV",
        "XLI",
        "XLU",
        "XLP",
        "XLB",
        "XLRE",
        "EEM",
        "EFA",
        "FXI",
        "EWZ",
        "EWJ",
        "KWEB",
        "ARKK",
        "ARKG",
        "ARKW",
        "ARKF",
        "VNQ",
        "REET",
        "KBWB",
        "KRE",
        "XHB",
        "XRT",
        "IBB",
        "XBI",
        "SMH",
        "SOXX",
        "VXX",
        "UVXY",
        "SVXY",
        # High-activity retail / meme / crypto
        "GME",
        "AMC",
        "CLOV",
        "WKHS",
        # BBBY bankrupt (2023). WISH delisted. RIDE bankrupt. All removed.
        "SPCE",
        # NKLA removed: Nikola bankrupt, delisted to OTC (NKLAQ) Apr 2025.
        "BLNK",
        "CHPT",
        "EVGO",
        "TLRY",
        "CGC",
        "ACB",
        "SNDL",
        # CURLF is OTC/CSE with no US-listed options. Removed.
        "DKNG",
        "PENN",
        "MARA",
        "RIOT",
        "HUT",
        "CLSK",
        "BTBT",
        "CIFR",
        "AI",
        "BBAI",
        "SOUN",
        "PRCT",
        "ASTS",
        "RKLB",
        "OPEN",
        "TPVG",
        "GLAD",
        "PSEC",
    ],
    "sp500_large": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "BRK-B",
        "UNH",
        "LLY",
        "JPM",
        "V",
        "XOM",
        "MA",
        "AVGO",
        "PG",
        "HD",
        "COST",
        "MRK",
        "ABBV",
        "CVX",
        "CRM",
        "AMD",
        "PEP",
        "KO",
        "ADBE",
        "TMO",
        "ACN",
        "MCD",
        "BAC",
        "NFLX",
        "CSCO",
        "DHR",
        "LIN",
        "ABT",
        "WMT",
        "TXN",
        "INTC",
        "NEE",
        "DIS",
        "QCOM",
        "PM",
        "AMGN",
        "UPS",
        "INTU",
        "RTX",
        "CAT",
        "AMAT",
        "BKNG",
        "GS",
        # AI infra / memory / space momentum additions (2026)
        "CRWV",
        "NBIS",
        "SNDK",
        "TE",
        "QBTS",
        "LUNR",
        "RDW",
        "PL",
    ],
    "tech": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "INTC",
        "QCOM",
        "AVGO",
        "AMAT",
        "MU",
        "KLAC",
        "LRCX",
        "ASML",
        "TSM",
        "ARM",
        "SMCI",
        "MRVL",
        "ADBE",
        "CRM",
        "ORCL",
        "SAP",
        "SNOW",
        "PLTR",
        "CRWD",
        "PANW",
        "ZS",
        "DDOG",
        "NET",
        "MDB",
        "SHOP",
        "XYZ",
        "PYPL",
        "COIN",
    ],
    "etfs": [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "GLD",
        "SLV",
        "TLT",
        "HYG",
        "LQD",
        "EEM",
        "EFA",
        "XLE",
        "XLF",
        "XLK",
        "XLV",
        "XLI",
        "XLU",
        "XLP",
        "XLB",
        "XLRE",
        "VNQ",
        "ARKK",
        "SQQQ",
        "TQQQ",
    ],
    "biotech": [
        "MRNA",
        "BNTX",
        "GILD",
        "BIIB",
        "REGN",
        "VRTX",
        "ALNY",
        # SGEN removed: Seagen acquired by Pfizer, delisted Dec 2023.
        "RARE",
        "ACAD",
        "INCY",
        "EXEL",
        "BMRN",
        "SRPT",
        "BEAM",
    ],
    "momentum": [
        "NVDA",
        "META",
        "TSLA",
        "PLTR",
        "SMCI",
        "ARM",
        "MSTR",
        "COIN",
        "CRWD",
        "NET",
        "DDOG",
        "SNOW",
        "SHOP",
        "UBER",
        "LYFT",
        "RBLX",
        "SOFI",
        "HOOD",
        "IONQ",
        "RGTI",
        "QUBT",
        "QBTS",
        "CRWV",
        "NBIS",
        "SNDK",
        "MU",
        "IREN",
        "TE",
    ],
    "space": [
        "ASTS",
        "RKLB",
        "SPCE",
        "IRDM",
        "GSAT",
        "VSAT",
        "LUNR",
        "RDW",
        "PL",
        "SIDU",
        "LMT",
        "NOC",
        "RTX",
        "GD",
        "LHX",
        "HEI",
        "TDG",
        "KTOS",
        "AVAV",
        "BWXT",
    ],
    "robotics": [
        "ISRG",
        "TER",
        "CGNX",
        "ROK",
        "SYM",
        "ZBRA",
        "AZTA",
        "PATH",
        "NVDA",
        "BOTZ",
    ],
    "energy": [
        "OKLO",
        "SMR",
        "CEG",
        "VST",
        "NRG",
        "BWXT",
        "CCJ",
        "UEC",
        "UUUU",
        "LEU",
        "XOM",
        "CVX",
        "COP",
        "OXY",
        "DVN",
        "FANG",
        "HAL",
        "SLB",
        "BKR",
        "LNG",
        "ET",
        "EPD",
        "WMB",
        "OKE",
        "VLO",
        "MPC",
        "PSX",
        "EOG",
        "APA",
        "FSLR",
        "ENPH",
        "PLUG",
        "BE",
        "XLE",
        "URA",
    ],
    "defense": [
        "LMT",
        "NOC",
        "RTX",
        "GD",
        "BA",
        "LHX",
        "HEI",
        "TDG",
        "HWM",
        "KTOS",
        "AVAV",
        "PLTR",
        "LDOS",
        "SAIC",
        "BAH",
        "CACI",
        "AXON",
        "BWXT",
        "DRS",
    ],
    "crypto": [
        "COIN",
        "HOOD",
        "XYZ",
        "MSTR",
        "MARA",
        "RIOT",
        "CLSK",
        "CORZ",
        "WULF",
        "IREN",
        "HUT",
        "CIFR",
        "BTBT",
    ],
    "ev": [
        "TSLA",
        "RIVN",
        "LCID",
        "NIO",
        "XPEV",
        "LI",
        "GM",
        "F",
        "CHPT",
        "BLNK",
        "EVGO",
    ],
    "financials": [
        "JPM",
        "GS",
        "MS",
        "BAC",
        "C",
        "WFC",
        "USB",
        "PNC",
        "TFC",
        "BX",
        "KKR",
        "APO",
        "CG",
        "ARES",
        "CME",
        "ICE",
        "CBOE",
        "V",
        "MA",
        "AXP",
        "SCHW",
        "COF",
    ],
    "cannabis": ["TLRY", "CGC", "ACB", "SNDL"],
}

DEFAULT_WATCHLIST = "sp500_large"

# Result dataclass


@dataclass
class ScanResult:
    ticker: str
    price: float
    score: float  # signed: + breakout up, - breakdown
    direction: str  # "breakout_up" | "breakdown" | "trending_up" | "trending_down" | "neutral"
    strength: str  # "strong" | "moderate" | "weak"
    regime: str
    signal_label: str
    confidence: float  # 0..1
    prob_up: float | None  # from MC if computed
    prob_down: float | None
    rsi: float
    adx: float
    macd_hist: float
    bb_position: float
    obv_slope: float
    atr_pct: float
    vol_regime: str
    donchian_pos: float
    hurst: float
    trend_score: float
    potential_up: float
    potential_down: float
    rsi_divergence: float
    ema200_dist: float
    price_vs_52w: float
    reasoning: str
    error: str | None = None
    elapsed_ms: float = 0.0
    expected_move: dict | None = None  # 1d/1w 1σ-2σ ranges from realized vol


# Scoring


def _compute_scan_score(reg, sig, ind) -> float:
    """
    Composite breakout/breakdown score in [-1, 1].
    Combines regime, signal, and key indicator reads.
    Positive -> breakout up; Negative -> breakdown.
    """
    score = 0.0

    regime_map = {
        "breakout_up": +1.0,
        "strong_uptrend": +0.75,
        "weak_uptrend": +0.40,
        "breakout_down": -1.0,
        "strong_downtrend": -0.75,
        "weak_downtrend": -0.40,
        "range_bound": 0.0,
        "choppy": 0.0,
    }
    score += regime_map.get(reg.regime, 0.0) * 0.40

    score += float(np.clip(sig.composite * sig.confidence, -1.0, 1.0)) * 0.30

    score += float(np.clip(reg.donchian_pos, -1.0, 1.0)) * 0.15

    score += float(np.clip(ind.obv_slope / 1.0, -1.0, 1.0)) * 0.10

    rsi_score = 0.0
    if ind.rsi > 70:
        rsi_score = -0.3  # overbought - mild fade
    elif ind.rsi < 30:
        rsi_score = 0.3  # oversold - mild bounce
    score += rsi_score * 0.05

    return float(np.clip(score, -1.0, 1.0))


def _classify_direction(score: float, regime: str) -> tuple[str, str]:
    """Returns (direction, strength)."""
    abs_s = abs(score)

    if regime in ("breakout_up", "breakout_down"):
        strength = "strong" if abs_s > 0.5 else "moderate"
    elif abs_s > 0.55:
        strength = "strong"
    elif abs_s > 0.30:
        strength = "moderate"
    else:
        strength = "weak"

    if score > 0.20:
        if regime == "breakout_up":
            direction = "breakout_up"
        elif "uptrend" in regime:
            direction = "trending_up"
        else:
            direction = "bullish"
    elif score < -0.20:
        if regime == "breakout_down":
            direction = "breakdown"
        elif "downtrend" in regime:
            direction = "trending_down"
        else:
            direction = "bearish"
    else:
        direction = "neutral"

    return direction, strength


# Single-ticker scan


async def _scan_one(
    ticker: str,
    interval: str,
    lookback: int,
    extended: bool,
    loop: asyncio.AbstractEventLoop,
) -> ScanResult:
    t0 = time.monotonic()
    try:
        df = await loop.run_in_executor(None, fetch_candles, ticker, interval, lookback, extended)
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(None, detect_regime, df, ind.adx, ind.obv_slope)
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)

        price = float(df["close"].iloc[-1])
        score = _compute_scan_score(reg, sig, ind)
        direction, strength = _classify_direction(score, reg.regime)

        # Expected-move ranges (realized vol). Daily scans reuse the candles
        # already fetched; intraday scans fetch ~60 daily bars (cached).
        if interval == "1d":
            em = compute_expected_move(
                df["close"].to_numpy(dtype=float),
                spot=price,
                daily_opens=df["open"].to_numpy(dtype=float),
                daily_highs=df["high"].to_numpy(dtype=float),
                daily_lows=df["low"].to_numpy(dtype=float),
            )
        else:
            em = await loop.run_in_executor(None, expected_move_for_ticker, ticker, price)

        elapsed = (time.monotonic() - t0) * 1000

        return ScanResult(
            ticker=ticker,
            price=round(price, 4),
            score=round(score, 4),
            direction=direction,
            strength=strength,
            regime=reg.regime,
            signal_label=sig.label,
            confidence=round(sig.confidence, 3),
            prob_up=None,
            prob_down=None,
            rsi=ind.rsi,
            adx=ind.adx,
            macd_hist=ind.macd_hist,
            bb_position=ind.bb_position,
            obv_slope=ind.obv_slope,
            atr_pct=ind.atr_pct,
            vol_regime=ind.vol_regime,
            donchian_pos=reg.donchian_pos,
            hurst=reg.hurst,
            trend_score=reg.trend_score,
            potential_up=reg.potential_up,
            potential_down=reg.potential_down,
            rsi_divergence=ind.rsi_divergence,
            ema200_dist=ind.ema200_dist,
            price_vs_52w=ind.price_vs_52w,
            reasoning=sig.reasoning,
            elapsed_ms=round(elapsed, 1),
            expected_move=em,
        )

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("[scanner] %s failed: %s", ticker, e)
        return ScanResult(
            ticker=ticker,
            price=0.0,
            score=0.0,
            direction="error",
            strength="weak",
            regime="unknown",
            signal_label="Error",
            confidence=0.0,
            prob_up=None,
            prob_down=None,
            rsi=50.0,
            adx=0.0,
            macd_hist=0.0,
            bb_position=0.0,
            obv_slope=0.0,
            atr_pct=0.0,
            vol_regime="unknown",
            donchian_pos=0.0,
            hurst=0.5,
            trend_score=0.0,
            potential_up=33.3,
            potential_down=33.3,
            rsi_divergence=0.0,
            ema200_dist=0.0,
            price_vs_52w=0.0,
            reasoning="",
            error=str(e),
            elapsed_ms=round(elapsed, 1),
        )


# Public: scan watchlist


async def _run_mc_on_result(
    result: ScanResult,
    interval: str,
    lookback: int,
    extended: bool,
    loop: asyncio.AbstractEventLoop,
    n_sim: int = 500,
    n_forward: int = 10,
    mc_model: str = "garch",
) -> ScanResult:
    """
    Re-fetch candles for one ScanResult and run a fast MC (n_sim=500) to
    produce accurate prob_up / prob_down estimates. Mutates in place.
    Returns the updated ScanResult.
    """
    try:
        df = await loop.run_in_executor(None, fetch_candles, result.ticker, interval, lookback, extended)
        ind = await loop.run_in_executor(None, compute_indicators, df)
        reg = await loop.run_in_executor(None, detect_regime, df, ind.adx, ind.obv_slope)
        sig = await loop.run_in_executor(None, compute_signal, ind, reg)
        price = float(df["close"].iloc[-1])

        mc = await loop.run_in_executor(
            None,
            run_mc,
            price,
            sig,
            n_sim,
            n_forward,
            mc_model,
            ind.returns,
            ind.kurtosis,
        )
        result.prob_up = round(float(mc.prob_up), 1)
        result.prob_down = round(float(mc.prob_down), 1)
    except Exception as e:
        logger.debug("[scanner MC] %s failed: %s", result.ticker, e)
    return result


async def scan_tickers(
    tickers: list[str],
    interval: str = "1d",
    lookback: int = 60,
    extended: bool = False,
    max_concurrent: int | None = None,  # defaults to cfg.scan_max_concurrent
    min_score_abs: float | None = None,  # defaults to cfg.scan_min_score
    mc_top_n: int = 5,  # run fast MC on top N results (0 to disable)
    mc_n_sim: int = 500,  # simulations for top-N MC pass
    mc_n_forward: int = 10,  # forward candles for top-N MC
    mc_model: str = "garch",
) -> dict[str, Any]:
    """
    Scan tickers concurrently. Returns a structured report with:
      - breakouts:  top bullish signals (score > 0)
      - breakdowns: top bearish signals (score < 0)
      - all:        full ranked list

    If mc_top_n > 0, a fast Monte Carlo is run on the top mc_top_n results
    (by absolute score) to produce accurate prob_up / prob_down estimates.
    """
    if max_concurrent is None:
        max_concurrent = cfg.scan_max_concurrent
    if min_score_abs is None:
        min_score_abs = cfg.scan_min_score
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)

    async def _throttled(t: str) -> ScanResult:
        async with sem:
            return await _scan_one(t, interval, lookback, extended, loop)

    t_start = time.monotonic()
    tasks = [asyncio.create_task(_throttled(t)) for t in tickers]
    results = await asyncio.gather(*tasks)

    # Filter errors
    valid = [r for r in results if r.direction != "error"]
    errors = [r for r in results if r.direction == "error"]

    # Apply minimum score filter
    if min_score_abs > 0:
        valid = [r for r in valid if abs(r.score) >= min_score_abs]

    # Sort by score descending
    ranked = sorted(valid, key=lambda r: r.score, reverse=True)

    # Fast MC on top-N results
    if mc_top_n > 0 and ranked:
        # Pick top N by |score| (covers both strong breakouts and breakdowns)
        by_abs = sorted(ranked, key=lambda r: abs(r.score), reverse=True)
        top_n = by_abs[:mc_top_n]
        mc_sem = asyncio.Semaphore(min(mc_top_n, 3))  # limit concurrency for MC

        async def _mc_throttled(r: ScanResult) -> ScanResult:
            async with mc_sem:
                return await _run_mc_on_result(
                    r,
                    interval,
                    lookback,
                    extended,
                    loop,
                    n_sim=mc_n_sim,
                    n_forward=mc_n_forward,
                    mc_model=mc_model,
                )

        mc_tasks = [asyncio.create_task(_mc_throttled(r)) for r in top_n]
        await asyncio.gather(*mc_tasks)
        # Results are mutated in place - ranked already references the same objects

    elapsed = round((time.monotonic() - t_start) * 1000)

    breakouts = [r for r in ranked if r.score > 0.20]
    breakdowns = [r for r in ranked if r.score < -0.20]
    neutral = [r for r in ranked if abs(r.score) <= 0.20]

    def _to_dict(r: ScanResult) -> dict:
        return asdict(r)

    return {
        "scanned": len(tickers),
        "succeeded": len(valid),
        "failed": len(errors),
        "elapsed_ms": elapsed,
        "interval": interval,
        "lookback": lookback,
        "breakouts": [_to_dict(r) for r in breakouts],
        "breakdowns": [_to_dict(r) for r in breakdowns],
        "neutral": [_to_dict(r) for r in neutral],
        "all": [_to_dict(r) for r in ranked],
        "errors": [{"ticker": r.ticker, "error": r.error} for r in errors],
    }


def get_watchlist(name: str) -> list[str]:
    """Return a named watchlist (case-insensitive). Falls back to sp500_large."""
    return WATCHLISTS.get(name.lower(), WATCHLISTS[DEFAULT_WATCHLIST])
