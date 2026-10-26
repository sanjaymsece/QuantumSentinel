from __future__ import annotations

import pytest

from app.config import MODEL_FILE
from app.detection.detector import load_detector
from app.detection.features import FEATURE_NAMES
from app.simulation.bb84 import (
    Engine,
    EveMode,
    SimulationRequest,
    simulate_session,
)
from app.simulation.channels import ChannelType


@pytest.fixture(scope="module")
def detector():
    return load_detector(MODEL_FILE)


def _predict(detector, eve: EveMode, noise: int, seed: int):
    result = simulate_session(
        SimulationRequest(
            eve=eve,
            eve_strength=50,
            noise=noise,
            qubits=2000,
            channel=ChannelType.DEPOLARIZING,
            engine=Engine.CLASSICAL,
            seed=seed,
        )
    )
    return result, detector.predict(result.observation)


def test_model_schema_matches_backend(detector) -> None:
    assert detector.feature_names == FEATURE_NAMES
    assert 0.0 < detector.threshold < 1.0
    assert detector.metadata["name"] == "BB84 QuantumSentinel Detector"
    assert len(detector.metadata["feature_names"]) == len(FEATURE_NAMES)


def test_burst_attack_is_detected_with_high_confidence(detector) -> None:
    result, detection = _predict(detector, EveMode.BURST, 4, seed=1)
    assert result.observation.qber < 11.0
    assert detection.detected is True
    assert detection.attack_probability > 0.9
    assert detection.confidence > 0.9


def test_benign_low_noise_session_is_secure(detector) -> None:
    _, detection = _predict(detector, EveMode.OFF, 4, seed=42)
    assert detection.detected is False
    assert detection.attack_probability < 0.2


def test_benign_high_noise_is_not_an_ml_false_alarm(detector) -> None:
    """The 11% rule flags this session; the ML detector correctly clears it."""
    result, detection = _predict(detector, EveMode.OFF, 20, seed=1)
    assert result.observation.qber > 11.0
    assert detection.detected is False


def test_model_card_reports_meaningful_metrics(detector) -> None:
    metrics = detector.metadata["metrics"]
    assert metrics["ml"]["recall"] > 0.85
    assert metrics["ml"]["false_positive_rate"] < 0.03
    assert metrics["ml"]["roc_auc"] > 0.95
    assert metrics["low_qber_attacks"]["recall"] > 0.8
