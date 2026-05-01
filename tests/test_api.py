"""API smoke tests using FastAPI's TestClient."""

from __future__ import annotations

import os
import sys

import pytest

# Make sure the in-process `cfg.db_path` points to a temp file
import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config.cfg, "db_path", str(tmp_path / "api.db"))
    # Patch fetch_candles so the API never touches the network
    import pandas as pd
    import numpy as np
    from core import fetcher
    rng = np.random.default_rng(7)

    def _fake_fetch(ticker, interval, lookback, extended):
        n = max(lookback, 60)
        rets = 0.0001 + 0.005 * rng.standard_normal(n)
        closes = 100 * np.cumprod(1 + rets)
        opens  = np.concatenate([[100.0], closes[:-1]])
        idx = pd.date_range("2024-01-02", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "open": opens, "high": closes * 1.002, "low": closes * 0.998,
            "close": closes, "volume": [1000]*n,
        }, index=idx)
        df.attrs["session"] = "regular"
        df.attrs["session_now"] = "regular"
        df.attrs["extended"] = False
        return df

    monkeypatch.setattr(fetcher, "fetch_candles", _fake_fetch)
    # Also patch through the module path used by the API
    import core
    monkeypatch.setattr(core, "fetch_candles", _fake_fetch, raising=False)
    from api import server as api_server
    monkeypatch.setattr(api_server, "fetch_candles", _fake_fetch)

    from fastapi.testclient import TestClient
    return TestClient(api_server.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_signal(client):
    r = client.get("/api/signal")
    assert r.status_code == 200
    j = r.json()
    assert "signal" in j and "mc" in j and "indicators" in j


def test_config_get(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    j = r.json()
    assert "valid_intervals" in j and "valid_mc_models" in j


def test_config_post_validation(client):
    r = client.post("/api/config", json={"interval": "abc"})
    assert r.status_code == 422


def test_config_post_ok(client):
    r = client.post("/api/config", json={"mc_model": "gaussian", "n_sim": 100})
    assert r.status_code == 200, r.text
    j = r.json()
    assert any("mc_model" in c for c in j["changed"])


def test_history_endpoint(client):
    # First trigger one analysis so the store has a row
    client.get("/api/signal")
    r = client.get("/api/history")
    assert r.status_code == 200
    items = r.json().get("items", [])
    assert isinstance(items, list)


def test_metrics_endpoint(client):
    client.get("/api/signal")
    r = client.get("/api/metrics")
    assert r.status_code == 200
    j = r.json()
    assert "signals" in j


def test_backtest_endpoint(client):
    r = client.post("/api/backtest",
                    json={"history_bars": 150, "n_forward": 5, "n_sim": 80,
                          "mc_model": "gaussian"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert "n_evaluated" in j


def test_export_csv(client):
    client.get("/api/signal")
    r = client.get("/api/export.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    body = r.text
    assert "ticker" in body.splitlines()[0]
