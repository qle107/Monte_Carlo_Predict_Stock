"""Static sector/industry labels and ETF universes.

Pure data module - no network or third-party imports. The canonical source is
``SECTORS`` (sector -> tickers); ``SECTOR_MAP`` (ticker -> sector) is derived.
"""

from __future__ import annotations

# fmt: off
SECTORS: dict[str, list[str]] = {
    "Tech": ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN"],
    "Media": [
        "NFLX", "SPOT", "DIS", "CMCSA", "CHTR", "WBD", "FOXA", "FOX", "PARA",
        "SIRI", "NYT", "NWS", "NWSA", "IAC",
    ],
    "Semis": [
        "NVDA", "AMD", "INTC", "QCOM", "AMAT", "MU", "MRVL", "ARM", "TSM",
        "ASML", "AVGO", "TXN", "SMCI", "KLAC", "LRCX", "ADI", "QRVO", "SWKS",
        "MPWR", "ON", "WOLF", "ACLS", "COHU", "SNDK",
    ],
    "Software": [
        "CRM", "ORCL", "NOW", "SNOW", "CRWD", "PANW", "ZS", "DDOG", "NET",
        "MDB", "OKTA", "ZM", "DOCU", "TWLO", "HUBS", "SHOP", "ADBE", "INTU",
        "SAP", "ACN", "IBM", "CSCO", "ANET", "FFIV", "CIEN", "PSTG",
    ],
    "Quantum": ["IONQ", "RGTI", "QUBT", "QBTS"],
    "AI": ["BBAI", "SOUN", "AI", "CRWV", "NBIS"],
    "Defense": [
        "PLTR", "BA", "LMT", "NOC", "GD", "RTX", "LHX", "HEI", "TDG", "HWM",
        "KTOS", "AVAV", "DRS", "LDOS", "SAIC", "BAH", "CACI", "AXON",
    ],
    "Finance": [
        "V", "MA", "JPM", "GS", "MS", "BAC", "C", "WFC", "BX", "KKR", "APO",
        "CG", "ARES", "OWL", "BRK-B", "AXP", "COF", "SYF", "USB", "PNC", "TFC",
        "STT", "SCHW", "CME", "ICE", "CBOE", "MSCI", "MCO", "FIS", "FISV",
        "GPN", "AIG", "MET", "PRU", "AFL", "TRV", "CB", "ALL", "HIG", "SPGI",
        "VRT",
    ],
    "FinTech": ["PYPL", "XYZ", "SOFI", "AFRM", "ENVA", "LC", "UPST"],
    "Crypto": [
        "COIN", "HOOD", "MSTR", "MARA", "RIOT", "HUT", "CLSK", "BTBT", "CIFR",
        "CORZ", "WULF", "IREN", "BITF",
    ],
    "Biotech": [
        "MRNA", "BNTX", "GILD", "VRTX", "BIIB", "REGN", "AMGN", "ALNY",
        "BMRN", "SRPT", "EXEL", "ACAD", "INCY", "RARE", "BEAM", "NTLA",
        "CRSP", "EDIT", "RXRX", "FATE", "KYMR", "IMVT",
    ],
    "Pharma": ["ABBV", "LLY", "PFE", "MRK", "AZN", "NVO", "SNY", "GSK", "BMY"],
    "Health": [
        "JNJ", "UNH", "CVS", "HUM", "CNC", "MOH", "ELV", "CI", "HCA", "THC",
        "UHS", "CLOV",
    ],
    "Robotics": [
        "ISRG", "TER", "CGNX", "ROK", "PATH", "SYM", "ZBRA",
        "AZTA", "BOTZ",
    ],
    "MedTech": [
        "EW", "DXCM", "PODD", "HOLX", "IDXX", "SYK", "ZTS", "BSX", "MDT",
        "ABT", "TMO", "DHR", "A", "BIO", "IQV", "RVTY", "PRCT", "NTRA",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "OXY", "DVN", "FANG", "HAL", "SLB", "BKR", "NOV",
        "HP", "PTEN", "LNG", "CQP", "ET", "EPD", "MPLX", "WMB", "OKE", "KMI",
        "VLO", "MPC", "PSX", "DK", "DINO", "EOG", "APA", "PR", "SM",
        "WHD",
    ],
    "Nuclear": ["OKLO", "SMR", "CEG", "VST", "NRG", "BWXT", "TLN"],
    "Uranium": ["CCJ", "UEC", "UUUU", "LEU"],
    "Solar": ["FSLR", "ENPH", "TE"],
    "CleanEnergy": ["PLUG", "BE"],
    "Consumer": [
        "NKE", "LULU", "RL", "PVH", "HBI", "UAA", "UA", "TGT", "LOW", "BBY",
        "BBWI", "M", "KSS", "CMG", "YUM", "QSR", "DRI", "TXRH", "SHAK",
        "WING", "ETSY", "CHWY", "W", "WMT", "COST", "SBUX", "MCD", "PG", "KO",
        "PEP", "MO", "PM", "CL", "GIS", "CPB", "HSY", "MKC", "CHD", "CLX",
        "SJM",
    ],
    "Auto/EV": [
        "TSLA", "GM", "F", "RIVN", "LCID", "NIO", "XPEV", "LI", "BLNK",
        "CHPT", "EVGO",
    ],
    "Platform": [
        "UBER", "LYFT", "ABNB", "DASH", "RBLX", "MTCH", "YELP", "TRIP", "OPEN",
    ],
    "Social": ["SNAP", "PINS"],
    "Hardware": ["HPQ", "HPE", "DELL", "WDC", "STX", "NTAP", "LITE", "VIAV"],
    "SpaceTech": ["ASTS", "RKLB", "SPCE", "IRDM", "GSAT", "VSAT", "LUNR", "RDW", "PL", "SIDU"],
    "Industrial": [
        "CAT", "DE", "EMR", "PH", "ITW", "GWW", "CMI", "ETN", "DOV", "UPS",
        "FDX", "NSC", "MMM", "HON",
    ],
    "Utilities": ["NEE", "SO", "DUK", "AEP", "EXC", "D", "SRE", "PCG", "PEG"],
    "Materials": [
        "ECL", "APD", "DD", "PPG", "SHW", "MT", "STLD", "NUE", "CLF", "AA",
        "CF", "MOS",
    ],
    "Transport": ["AAL", "DAL", "UAL", "LUV", "JBLU", "ALK"],
    "Travel": ["CCL", "RCL", "NCLH", "BKNG"],
    "Telecom": ["T", "VZ", "TMUS", "LUMN"],
    "REIT": [
        "AMT", "CCI", "SBAC", "EQIX", "DLR", "PLD", "SPG", "O", "WPC", "NNN",
        "VICI",
    ],
    "Gaming": ["MGM", "LVS", "WYNN", "CZR", "DKNG", "PENN"],
    "Mining": [
        "GOLD", "NEM", "AEM", "WPM", "PAAS", "AG", "BHP", "RIO", "VALE", "FCX",
    ],
    "Meme": ["GME", "AMC", "WKHS"],
    "Cannabis": ["TLRY", "CGC", "ACB", "SNDL"],
    "ETF": [
        "VNQ", "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "TQQQ", "SQQQ",
        "GLD", "GDX", "GDXJ", "SLV", "USO", "UNG", "TLT", "IEF", "SHY", "HYG",
        "LQD", "JNK", "XLE", "URA", "XLF", "XLK", "XLV", "XLI", "XLU", "XLP",
        "XLB", "XLRE", "EEM", "EFA", "FXI", "EWZ", "EWJ", "KWEB", "ARKK",
        "ARKG", "ARKW", "ARKF", "REET", "KBWB", "KRE", "XHB", "XRT", "IBB",
        "XBI", "SMH", "SOXX", "VXX", "UVXY", "SVXY", "TPVG", "GLAD", "PSEC",
    ],
}
# fmt: on

# Derived ticker -> sector lookup.
SECTOR_MAP: dict[str, str] = {t: sector for sector, tickers in SECTORS.items() for t in tickers}


def sector_for(ticker: str) -> str:
    """Sector label for a ticker, or "Other" when unknown."""
    return SECTOR_MAP.get(ticker.upper(), "Other")


# Liquid ETFs excluded from the unusual-options feed by default - their tape is
# dominated by hedging rather than conviction flow.
HIGH_VOLUME_ETFS: frozenset[str] = frozenset(
    {
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
        "TQQQ", "SQQQ", "QID", "SDS", "UPRO", "SPXU", "SPXL",
        "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "URA",
        "TLT", "IEF", "SHY", "HYG", "LQD", "JNK",
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "XLP", "XLB", "XLRE", "XLC", "XLY",
        "EEM", "EFA", "FXI", "EWZ", "EWJ", "KWEB",
        "SMH", "SOXX", "SOXL", "SOXS",
        "VXX", "UVXY", "SVXY", "VIXY",
        "ARKK", "ARKG", "ARKW", "ARKF",
        "KRE", "KBWB", "XHB", "XRT", "IBB", "XBI",
        "VNQ", "REET",
    }
)
