"""Adaptive conformal calibration of Monte Carlo forecast bands (Gibbs & Candes 2021)."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from core.data.db import sqlite_connect

logger = logging.getLogger(__name__)

ALPHA_TARGET = 0.20  # nominal miscoverage of the outer band (80% coverage)
ALPHA_MIN = 0.02  # never wider than P1-P99
ALPHA_MAX = 0.45  # never narrower than P22.5-P77.5
GAMMA = 0.02  # ACI learning rate
MIN_SETTLED = 10  # keep nominal alpha until this many bands have settled

_INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conformal_forecasts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    ticker   TEXT NOT NULL,
    interval TEXT NOT NULL,
    horizon  INTEGER NOT NULL,
    spot     REAL NOT NULL,
    lo       REAL NOT NULL,
    hi       REAL NOT NULL,
    realized REAL,
    err      INTEGER,
    settled  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conf_open
    ON conformal_forecasts(ticker, interval, horizon, settled, ts);

CREATE TABLE IF NOT EXISTS conformal_state (
    ticker    TEXT NOT NULL,
    interval  TEXT NOT NULL,
    horizon   INTEGER NOT NULL,
    alpha     REAL NOT NULL,
    n_settled INTEGER NOT NULL DEFAULT 0,
    n_miss    INTEGER NOT NULL DEFAULT 0,
    updated   TEXT,
    PRIMARY KEY (ticker, interval, horizon)
);
"""


def _parse_ts(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


class BandCalibrator:
    """SQLite-backed ACI state, one alpha per (ticker, interval, horizon)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite_connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    # -- public API ---------------------------------------------------------

    def target_alpha(self, ticker: str, interval: str, horizon: int) -> float:
        """Calibrated miscoverage level for the next forecast (default 0.20)."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT alpha, n_settled FROM conformal_state "
                    "WHERE ticker=? AND interval=? AND horizon=?",
                    (ticker.upper(), interval, int(horizon)),
                ).fetchone()
            if row is None or int(row["n_settled"]) < MIN_SETTLED:
                return ALPHA_TARGET
            return float(min(max(row["alpha"], ALPHA_MIN), ALPHA_MAX))
        except sqlite3.Error as e:
            logger.warning("BandCalibrator.target_alpha failed: %s", e)
            return ALPHA_TARGET

    def observe(
        self,
        ticker: str,
        interval: str,
        horizon: int,
        ts: str,
        spot: float,
        lo: float,
        hi: float,
    ) -> None:
        """Store one issued band (called right after each MC run)."""
        if not (lo < hi) or spot <= 0:
            return
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT INTO conformal_forecasts "
                    "(ts, ticker, interval, horizon, spot, lo, hi) VALUES (?,?,?,?,?,?,?)",
                    (ts, ticker.upper(), interval, int(horizon), float(spot), float(lo), float(hi)),
                )
        except sqlite3.Error as e:
            logger.warning("BandCalibrator.observe failed: %s", e)

    def settle(self, ticker: str, interval: str, price_now: float, now_ts: str | None = None) -> int:
        """
        Score every open forecast whose maturity has passed, run the ACI
        update sequentially (chronological order), persist the new alpha.
        Returns the number of forecasts settled.
        """
        sec = _INTERVAL_SECONDS.get(interval)
        if sec is None or price_now <= 0:
            return 0
        now = _parse_ts(now_ts) if now_ts else datetime.now(timezone.utc)
        if now is None:
            return 0

        try:
            with self._lock, self._conn() as conn:
                rows = conn.execute(
                    "SELECT id, ts, horizon, lo, hi FROM conformal_forecasts "
                    "WHERE ticker=? AND interval=? AND settled=0 ORDER BY ts",
                    (ticker.upper(), interval),
                ).fetchall()

                n_done = 0
                # alpha state per horizon (several horizons can coexist)
                states: dict[int, tuple[float, int, int]] = {}

                for r in rows:
                    issued = _parse_ts(r["ts"])
                    if issued is None:
                        conn.execute("DELETE FROM conformal_forecasts WHERE id=?", (r["id"],))
                        continue
                    h = int(r["horizon"])
                    if issued + timedelta(seconds=sec * h) > now:
                        continue  # not mature yet

                    err = int(not (float(r["lo"]) <= price_now <= float(r["hi"])))
                    conn.execute(
                        "UPDATE conformal_forecasts SET realized=?, err=?, settled=1 WHERE id=?",
                        (float(price_now), err, r["id"]),
                    )

                    if h not in states:
                        srow = conn.execute(
                            "SELECT alpha, n_settled, n_miss FROM conformal_state "
                            "WHERE ticker=? AND interval=? AND horizon=?",
                            (ticker.upper(), interval, h),
                        ).fetchone()
                        states[h] = (
                            (float(srow["alpha"]), int(srow["n_settled"]), int(srow["n_miss"]))
                            if srow
                            else (ALPHA_TARGET, 0, 0)
                        )
                    alpha, n_settled, n_miss = states[h]
                    # ACI update (Gibbs & Candes 2021)
                    alpha = alpha + GAMMA * (ALPHA_TARGET - err)
                    alpha = float(min(max(alpha, ALPHA_MIN), ALPHA_MAX))
                    states[h] = (alpha, n_settled + 1, n_miss + err)
                    n_done += 1

                for h, (alpha, n_settled, n_miss) in states.items():
                    conn.execute(
                        "INSERT INTO conformal_state (ticker, interval, horizon, alpha, n_settled, n_miss, updated) "
                        "VALUES (?,?,?,?,?,?,?) "
                        "ON CONFLICT(ticker, interval, horizon) DO UPDATE SET "
                        "alpha=excluded.alpha, n_settled=excluded.n_settled, "
                        "n_miss=excluded.n_miss, updated=excluded.updated",
                        (ticker.upper(), interval, h, alpha, n_settled, n_miss, now.isoformat()),
                    )

                # housekeeping: drop settled rows older than 30 days
                cutoff = (now - timedelta(days=30)).isoformat()
                conn.execute("DELETE FROM conformal_forecasts WHERE settled=1 AND ts < ?", (cutoff,))

                return n_done
        except sqlite3.Error as e:
            logger.warning("BandCalibrator.settle failed: %s", e)
            return 0

    def seed_state(self, ticker: str, interval: str, horizon: int, errs: list[int]) -> None:
        """
        Initialise the ACI state from a backfilled hit/miss sequence
        (warm start).  No-op when live state already exists, so real settled
        forecasts are never overwritten by a replay.
        """
        if not errs:
            return
        alpha = ALPHA_TARGET
        for e in errs:
            alpha = min(max(alpha + GAMMA * (ALPHA_TARGET - e), ALPHA_MIN), ALPHA_MAX)
        try:
            with self._lock, self._conn() as conn:
                row = conn.execute(
                    "SELECT n_settled FROM conformal_state "
                    "WHERE ticker=? AND interval=? AND horizon=?",
                    (ticker.upper(), interval, int(horizon)),
                ).fetchone()
                if row is not None and int(row["n_settled"]) > 0:
                    return
                conn.execute(
                    "INSERT INTO conformal_state (ticker, interval, horizon, alpha, n_settled, n_miss, updated) "
                    "VALUES (?,?,?,?,?,?,?) "
                    "ON CONFLICT(ticker, interval, horizon) DO UPDATE SET "
                    "alpha=excluded.alpha, n_settled=excluded.n_settled, "
                    "n_miss=excluded.n_miss, updated=excluded.updated",
                    (
                        ticker.upper(),
                        interval,
                        int(horizon),
                        float(alpha),
                        len(errs),
                        int(sum(errs)),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except sqlite3.Error as e:
            logger.warning("BandCalibrator.seed_state failed: %s", e)

    def coverage(self, ticker: str, interval: str, horizon: int) -> dict:
        """Diagnostics: empirical coverage and current alpha."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT alpha, n_settled, n_miss, updated FROM conformal_state "
                    "WHERE ticker=? AND interval=? AND horizon=?",
                    (ticker.upper(), interval, int(horizon)),
                ).fetchone()
            if row is None:
                return {"alpha": ALPHA_TARGET, "n_settled": 0, "empirical_coverage": None}
            n, miss = int(row["n_settled"]), int(row["n_miss"])
            return {
                "alpha": round(float(row["alpha"]), 4),
                "n_settled": n,
                "empirical_coverage": round(1.0 - miss / n, 4) if n else None,
                "target_coverage": 1.0 - ALPHA_TARGET,
                "updated": row["updated"],
            }
        except sqlite3.Error as e:
            logger.warning("BandCalibrator.coverage failed: %s", e)
            return {"alpha": ALPHA_TARGET, "n_settled": 0, "empirical_coverage": None}


def warm_start_from_history(
    cal: BandCalibrator,
    df,  # pandas OHLCV DataFrame (full display history)
    ticker: str,
    interval: str,
    horizon: int,
    mc_model: str,
    n_sim: int = 300,
    min_history: int = 60,
    max_origins: int = 30,
) -> int:
    """
    Cold-start fix: instead of waiting hours for live forecasts to mature,
    REPLAY the forecast over history the caller already holds.  At each
    origin bar t: build indicators/signal from bars <= t, run a reduced-size
    MC forecast, and score the nominal P10-P90 band against the actual close
    at t+horizon.  The resulting hit/miss sequence seeds the ACI state, so a
    freshly loaded ticker shows calibration numbers immediately.

    Pseudo-out-of-sample: no future data leaks into any individual forecast.
    Origins are spaced >= horizon/2 apart to limit overlap correlation.
    Skipped (returns 0) when live state already exists.  Returns #scored.
    """
    try:
        if cal.coverage(ticker, interval, horizon).get("n_settled", 0) > 0:
            return 0

        # Local imports avoid a circular dependency (montecarlo never imports us)
        from core.analysis.indicators import compute_indicators
        from core.analysis.montecarlo import run as run_mc
        from core.analysis.regime import detect_regime
        from core.analysis.signal import compute_signal

        closes = df["close"].to_numpy(dtype=float)
        n = len(df)
        last = n - int(horizon) - 1
        if last <= min_history:
            return 0
        step = max(1, int(horizon) // 2, (last - min_history) // max_origins)

        errs: list[int] = []
        for i in range(min_history, last + 1, step):
            sub = df.iloc[: i + 1]
            try:
                ind = compute_indicators(sub)
                reg = detect_regime(sub, adx=ind.adx, obv_slope=ind.obv_slope)
                sig = compute_signal(ind, regime=reg)
                entry = float(sub["close"].iloc[-1])
                if entry <= 0:
                    continue
                mc = run_mc(
                    entry,
                    sig,
                    n_simulations=n_sim,
                    n_candles=int(horizon),
                    model=mc_model,
                    recent_returns=ind.returns,
                    kurtosis_excess=ind.kurtosis,
                )
            except Exception:
                continue
            realized = float(closes[i + int(horizon)])
            if not (realized > 0):
                continue
            errs.append(int(not (mc.p10_price <= realized <= mc.p90_price)))

        if len(errs) >= 5:
            cal.seed_state(ticker, interval, horizon, errs)
            return len(errs)
        return 0
    except Exception as e:
        logger.warning("conformal warm_start failed: %s", e)
        return 0
