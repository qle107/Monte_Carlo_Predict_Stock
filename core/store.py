"""
core/store.py — SQLite-backed signal history.

Records every analysis the server produces. Lets the dashboard show
recent calls and aggregate accuracy. Thread-safe (one connection per
operation; SQLite handles the locking).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    interval        TEXT    NOT NULL,
    price           REAL    NOT NULL,
    label           TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    drift_bias      REAL    NOT NULL,
    prob_up         REAL    NOT NULL,
    prob_flat       REAL    NOT NULL,
    prob_down       REAL    NOT NULL,
    median_price    REAL    NOT NULL,
    expected_ret    REAL,
    cvar_5          REAL,
    mc_model        TEXT    NOT NULL,
    regime          TEXT,
    potential_up    REAL,
    potential_down  REAL,
    potential_flat  REAL,
    raw_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_ts ON signals(ticker, ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_ts        ON signals(ts DESC);
"""

# Schema migration: add regime columns if upgrading from v1 schema.
_MIGRATIONS = [
    "ALTER TABLE signals ADD COLUMN regime TEXT",
    "ALTER TABLE signals ADD COLUMN potential_up REAL",
    "ALTER TABLE signals ADD COLUMN potential_down REAL",
    "ALTER TABLE signals ADD COLUMN potential_flat REAL",
]


class SignalStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Apply additive migrations for older databases (no-op on fresh DBs).
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists

    @contextmanager
    def _cur(self):
        with self._lock:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()

    # ─── Writes ──────────────────────────────────────────────────────────

    def record(self, result: dict) -> None:
        """Write one analysis row. Tolerates missing fields."""
        if not isinstance(result, dict) or "signal" not in result or "mc" not in result:
            return
        sig = result.get("signal", {})
        mc  = result.get("mc",     {})
        reg = result.get("regime", {}) or {}
        row = (
            result.get("updated_at", ""),
            result.get("ticker",   ""),
            result.get("interval", ""),
            float(result.get("current_price", 0.0)),
            sig.get("label", ""),
            float(sig.get("confidence",  0.0)),
            float(sig.get("drift_bias",  0.0)),
            float(mc.get("prob_up",   0.0)),
            float(mc.get("prob_flat", 0.0)),
            float(mc.get("prob_down", 0.0)),
            float(mc.get("median_price", 0.0)),
            float(mc.get("expected_return", 0.0)),
            float(mc.get("cvar_5", 0.0)),
            result.get("mc_model", "gaussian"),
            reg.get("regime", ""),
            float(reg.get("potential_up",   0.0)),
            float(reg.get("potential_down", 0.0)),
            float(reg.get("potential_flat", 0.0)),
            json.dumps({"sub_scores": sig.get("sub_scores", {}),
                        "regime":     reg,
                        "indicators": result.get("indicators", {})},
                       default=str),
        )
        try:
            with self._cur() as conn:
                conn.execute(
                    """INSERT INTO signals
                    (ts, ticker, interval, price, label, confidence, drift_bias,
                     prob_up, prob_flat, prob_down, median_price, expected_ret,
                     cvar_5, mc_model, regime, potential_up, potential_down,
                     potential_flat, raw_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    row,
                )
        except sqlite3.Error as e:
            logger.warning("SignalStore.record failed: %s", e)

    # ─── Reads ───────────────────────────────────────────────────────────

    def recent(self, ticker: Optional[str] = None, limit: int = 100) -> List[dict]:
        sql = """SELECT ts, ticker, interval, price, label, confidence,
                        drift_bias, prob_up, prob_flat, prob_down,
                        median_price, expected_ret, cvar_5, mc_model,
                        regime, potential_up, potential_down, potential_flat
                 FROM signals"""
        params: tuple = ()
        if ticker:
            sql += " WHERE ticker = ?"
            params = (ticker.upper().strip(),)
        sql += " ORDER BY ts DESC LIMIT ?"
        params = (*params, int(limit))
        with self._cur() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def metrics(self, ticker: Optional[str] = None) -> dict:
        """Lightweight aggregate metrics."""
        where = ""
        params: tuple = ()
        if ticker:
            where = " WHERE ticker = ?"
            params = (ticker.upper().strip(),)

        with self._cur() as conn:
            row = conn.execute(
                f"""SELECT COUNT(*) AS n,
                          AVG(prob_up)    AS avg_prob_up,
                          AVG(confidence) AS avg_conf,
                          AVG(drift_bias) AS avg_drift,
                          MIN(ts)         AS first_ts,
                          MAX(ts)         AS last_ts
                       FROM signals{where}""",
                params,
            ).fetchone()

            label_rows = conn.execute(
                f"SELECT label, COUNT(*) AS c FROM signals{where} GROUP BY label",
                params,
            ).fetchall()

        labels = {r["label"]: r["c"] for r in label_rows}
        n = int(row["n"] or 0)
        return {
            "ticker":      ticker,
            "signals":     n,
            "avg_prob_up": round(float(row["avg_prob_up"] or 0), 2),
            "avg_conf":    round(float(row["avg_conf"]    or 0), 3),
            "avg_drift":   round(float(row["avg_drift"]   or 0), 6),
            "first_ts":    row["first_ts"],
            "last_ts":     row["last_ts"],
            "label_counts": labels,
        }

    def prune(self, days: int = 30) -> int:
        """
        Delete signal rows older than `days` days.
        Returns the number of rows deleted.

        Safe to call on a schedule (e.g. daily) to keep the DB small.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        try:
            with self._cur() as conn:
                cur = conn.execute(
                    "DELETE FROM signals WHERE ts < ?", (cutoff,)
                )
                deleted = cur.rowcount
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            if deleted:
                logger.info("SignalStore.prune: deleted %d rows older than %d days", deleted, days)
            return deleted
        except sqlite3.Error as e:
            logger.warning("SignalStore.prune failed: %s", e)
            return 0

    def accuracy_window(
        self,
        ticker: Optional[str] = None,
        limit: int = 200,
    ) -> dict:
        """
        Compute directional accuracy for the most recent `limit` rows that
        have a non-neutral label (Buy / Sell).

        For each consecutive pair of signals we use the *next* recorded price
        as the realised outcome — a simple but honest measure of whether
        the model's directional call was correct at the time of the next update.

        Returns
        -------
        {
          "n_calls": int,           # Buy/Sell signals evaluated
          "hit_rate": float|None,   # % correct (None if no data)
          "avg_prob_up_on_buys": float|None,
          "avg_prob_up_on_sells": float|None,
        }
        """
        rows = self.recent(ticker=ticker, limit=limit + 1)  # +1 to get next price
        if len(rows) < 2:
            return {"n_calls": 0, "hit_rate": None,
                    "avg_prob_up_on_buys": None, "avg_prob_up_on_sells": None}

        # rows are DESC by ts — reverse to get chronological order
        rows = list(reversed(rows))

        correct = 0
        total   = 0
        buy_probs:  List[float] = []
        sell_probs: List[float] = []

        for i in range(len(rows) - 1):
            row      = rows[i]
            next_row = rows[i + 1]
            label    = row.get("label", "")
            price    = float(row.get("price", 0.0) or 0.0)
            next_p   = float(next_row.get("price", 0.0) or 0.0)

            if not ("Buy" in label or "Sell" in label):
                continue
            if price <= 0 or next_p <= 0:
                continue

            up_call = "Buy" in label
            realised_up = next_p > price

            if up_call:
                buy_probs.append(float(row.get("prob_up", 50.0) or 50.0))
                if realised_up:
                    correct += 1
            else:
                sell_probs.append(float(row.get("prob_up", 50.0) or 50.0))
                if not realised_up:
                    correct += 1
            total += 1

        return {
            "n_calls":              total,
            "hit_rate":             round(correct / total * 100, 2) if total else None,
            "avg_prob_up_on_buys":  round(sum(buy_probs)  / len(buy_probs),  2) if buy_probs  else None,
            "avg_prob_up_on_sells": round(sum(sell_probs) / len(sell_probs), 2) if sell_probs else None,
        }

    def close(self) -> None:
        # Per-call connections; nothing to close globally.
        pass
