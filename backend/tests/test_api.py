from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.db import SessionStoreError
from app.main import app
from app.simulation.channels import qiskit_available


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        try:
            with psycopg.connect(
                os.environ["QS_DATABASE_URL"], autocommit=True
            ) as conn:
                conn.execute("TRUNCATE TABLE run_sessions")
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
        yield test_client


def _run(client: TestClient, **overrides):
    payload = {
        "eve": "off",
        "eve_strength": 50,
        "noise": 4,
        "qubits": 2000,
        "channel": "depolarizing",
        "engine": "js",
        "seed": 42,
    }
    payload.update(overrides)
    return client.post("/run", json=payload)


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database_connected"] is True
    assert body["database_name"]
    assert body["model"]["name"] == "BB84 QuantumSentinel Detector"
    assert response.headers["cache-control"] == "no-store"


def test_dashboard_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "QuantumSentinel" in response.text
    assert 'id="qberChart"' in response.text


def test_burst_attack_beats_the_threshold_rule(client: TestClient) -> None:
    response = _run(
        client, eve="burst", eve_strength=50, noise=4, seed=1
    )
    assert response.status_code == 200
    body = response.json()
    assert body["attack_present"] is True
    assert body["attack_mode"] == "burst"
    assert body["standard_detected"] is False
    assert body["ml_detected"] is True
    assert body["attack_probability"] > 0.9
    assert body["ml_confidence"] > 90.0
    assert body["threshold"] == 11.0
    assert body["engine"] == "classical"
    assert body["engine_requested"] == "js"
    assert len(body["block_qber"]) == 20
    assert body["sifted_bits"] > 0
    assert body["seed"] == 1
    assert body["model"]["name"] == "BB84 QuantumSentinel Detector"


def test_high_noise_false_alarm_is_cleared_by_ml(client: TestClient) -> None:
    response = _run(client, eve="off", noise=20, seed=1)
    assert response.status_code == 200
    body = response.json()
    assert body["qber"] > 11.0
    assert body["standard_detected"] is True
    assert body["ml_detected"] is False
    assert body["attack_present"] is False
    assert body["attack_mode"] == "none"


def test_off_mode_with_low_noise_is_secure(client: TestClient) -> None:
    response = _run(client, eve="off", noise=4, seed=42)
    assert response.status_code == 200
    body = response.json()
    assert body["attack_present"] is False
    assert body["standard_detected"] is False
    assert body["ml_detected"] is False


def test_amplitude_damping_channel_is_supported(client: TestClient) -> None:
    response = _run(
        client,
        channel="amplitude_damping",
        eve="burst",
        eve_strength=50,
        seed=3,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["x_qber"] >= 0.0
    assert body["z_qber"] >= 0.0
    assert len(body["block_qber"]) == 20


@pytest.mark.skipif(not qiskit_available(), reason="qiskit is not installed")
def test_qiskit_engine(client: TestClient) -> None:
    response = _run(
        client,
        eve="burst",
        eve_strength=50,
        seed=1,
        engine="qiskit",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "qiskit"
    assert body["engine_requested"] == "qiskit"
    assert body["ml_detected"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"noise": 21},
        {"noise": -1},
        {"qubits": 99},
        {"qubits": 5001},
        {"eve": "sometimes"},
        {"engine": "classical"},
        {"channel": "bit_flip"},
        {"unexpected": True},
    ],
)
def test_invalid_payloads_are_rejected(client: TestClient, payload) -> None:
    response = _run(client, **payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_seed_is_reproducible(client: TestClient) -> None:
    first = _run(client, eve="burst", seed=99).json()
    second = _run(client, eve="burst", seed=99).json()
    assert first["qber"] == second["qber"]
    assert first["block_qber"] == second["block_qber"]
    assert first["ml_confidence"] == second["ml_confidence"]


def test_model_endpoint_exposes_card(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_names"]
    assert body["metrics"]["ml"]["roc_auc"] > 0.9
    assert body["notes"]


def test_run_is_archived_in_postgres(client: TestClient) -> None:
    response = _run(client, eve="burst", eve_strength=50, seed=321)
    assert response.status_code == 200
    body = response.json()
    session_id = body["session_id"]

    listing = client.get("/sessions", params={"limit": 20})
    assert listing.status_code == 200
    assert listing.headers["cache-control"] == "no-store"
    payload = listing.json()
    assert payload["total"] >= 1
    assert payload["limit"] == 20
    assert payload["offset"] == 0

    summary = next(
        item for item in payload["items"] if item["session_id"] == session_id
    )
    assert summary["engine"] == "classical"
    assert summary["eve_mode"] == "burst"
    assert summary["eve_strength"] == 50
    assert summary["qubits"] == 2000
    assert summary["channel"] == "depolarizing"
    assert summary["qber"] == pytest.approx(body["qber"], abs=0.01)
    assert summary["ml_detected"] is True
    assert summary["attack_present"] is True
    assert summary["attack_mode"] == "burst"
    assert summary["created_at"]

    detail = client.get(f"/sessions/{session_id}")
    assert detail.status_code == 200
    record = detail.json()
    assert record["session_id"] == session_id
    assert record["engine_requested"] == "js"
    assert record["seed"] == 321
    assert len(record["block_qber"]) == 20
    assert len(record["features"]) == 33
    assert record["features"]["qber"] == pytest.approx(body["qber"], abs=0.01)
    assert record["model_name"] == "BB84 QuantumSentinel Detector"
    assert record["model_threshold"] == pytest.approx(0.66)

    flagged = client.get("/sessions", params={"ml_detected": True}).json()
    assert any(item["session_id"] == session_id for item in flagged["items"])


def test_archive_filters_and_pagination(client: TestClient) -> None:
    _run(client, eve="off", noise=4, seed=654)
    _run(client, eve="burst", seed=655)

    clean = client.get(
        "/sessions", params={"ml_detected": False, "limit": 50}
    ).json()
    assert clean["items"]
    assert all(item["ml_detected"] is False for item in clean["items"])

    first_page = client.get("/sessions", params={"limit": 1, "offset": 0}).json()
    second_page = client.get("/sessions", params={"limit": 1, "offset": 1}).json()
    assert len(first_page["items"]) == 1
    assert len(second_page["items"]) == 1
    assert first_page["items"][0]["session_id"] != second_page["items"][0]["session_id"]
    assert first_page["total"] == second_page["total"]


def test_archive_validates_and_handles_missing_sessions(
    client: TestClient,
) -> None:
    assert client.get("/sessions", params={"limit": 0}).status_code == 422
    assert client.get("/sessions", params={"limit": 101}).status_code == 422
    assert client.get("/sessions", params={"offset": -1}).status_code == 422
    assert client.get("/sessions/not-a-uuid").status_code == 422

    missing = client.get(f"/sessions/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Session not found."


def test_run_fails_when_archive_is_unavailable(client: TestClient) -> None:
    class FailingRepository:
        def save(self, record) -> None:  # noqa: ANN001
            raise SessionStoreError("database is down")

    previous = client.app.state.session_repository
    client.app.state.session_repository = FailingRepository()
    try:
        response = _run(client, seed=999)
        assert response.status_code == 503
        assert "could not be archived" in response.json()["detail"].lower()
    finally:
        client.app.state.session_repository = previous


def test_cors_allows_local_frontend(client: TestClient) -> None:
    response = client.get(
        "/health", headers={"Origin": "http://localhost:5500"}
    )
    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:5500"
    )


def test_missing_model_returns_503(client: TestClient) -> None:
    previous = client.app.state.detector
    client.app.state.detector = None
    try:
        response = _run(client)
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()
    finally:
        client.app.state.detector = previous
