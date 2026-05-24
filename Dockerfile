# Dockerfile — mc-trader self-hosted Monte Carlo dashboard
#
# Build:   docker build -t mc-trader .
# Run:     docker run --rm -p 8000:8000 --env-file .env mc-trader
#
# Security notes (see SECURITY.md):
#   • Default HOST=0.0.0.0 inside the container so the port is reachable from
#     the host.  Bind to 127.0.0.1 on the host with -p 127.0.0.1:8000:8000
#     to avoid exposing the port to the LAN.
#   • Set API_KEY in .env before running in any shared environment.
#   • Use paper-trading Alpaca keys only — never live-trading keys.

# ── Stage 1: dependency layer (cache-friendly) ────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System deps required by some Python packages (scipy, lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: application image ────────────────────────────────────────────────
FROM python:3.11-slim AS app

WORKDIR /app

# Copy installed packages from the deps stage (keeps the image lean)
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy runtime libs needed at import time (lxml / scipy shared libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY . .

# Create a non-root user for runtime
RUN useradd --create-home --shell /bin/bash mctrader \
    && chown -R mctrader:mctrader /app
USER mctrader

# ── Runtime configuration ─────────────────────────────────────────────────────
# These defaults are overridden by --env-file .env at runtime.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    TICKER=AAPL \
    ALPACA_MODE=paper \
    MC_MODEL=garch \
    MC_SIMULATIONS=2000 \
    MC_FORWARD_CANDLES=10 \
    CANDLE_INTERVAL=15m

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

CMD ["python", "main.py"]
