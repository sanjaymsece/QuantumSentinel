"""Load and run the trained BB84 intrusion-detection classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.detection.features import FEATURE_NAMES, extract_features
from app.simulation.bb84 import Observation

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the detector artifact is missing or malformed."""


@dataclass(frozen=True, slots=True)
class DetectionResult:
    attack_probability: float
    detected: bool
    confidence: float
    threshold: float


class Detector:
    """Thin, validated wrapper around the persisted scikit-learn model."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        feature_names = tuple(metadata.get("feature_names", ()))
        if feature_names != FEATURE_NAMES:
            raise ModelLoadError(
                "Model feature schema does not match the current backend. "
                "Retrain with `python -m app.detection.train`."
            )
        if not 0.0 < float(metadata.get("decision_threshold", 0.5)) < 1.0:
            raise ModelLoadError("Model decision_threshold must be in (0, 1)")
        if not hasattr(model, "predict_proba"):
            raise ModelLoadError("Loaded model does not expose predict_proba")
        self.model = model
        self.metadata = metadata
        self.feature_names = feature_names
        self.threshold = float(metadata["decision_threshold"])

    def predict(self, observation: Observation) -> DetectionResult:
        return self.predict_features(extract_features(observation))

    def predict_features(self, features: np.ndarray) -> DetectionResult:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (len(self.feature_names),):
            raise ValueError(
                "detector received an unexpected feature vector shape "
                f"{vector.shape}; expected ({len(self.feature_names)},)"
            )
        probability = float(
            self.model.predict_proba(vector.reshape(1, -1))[0, 1]
        )
        probability = min(max(probability, 0.0), 1.0)
        detected = probability >= self.threshold
        confidence = probability if detected else 1.0 - probability
        return DetectionResult(
            attack_probability=probability,
            detected=detected,
            confidence=confidence,
            threshold=self.threshold,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.metadata.get("name", "BB84 QuantumSentinel Detector"),
            "version": self.metadata.get("version", "1.0.0"),
            "trained_at": self.metadata.get("trained_at", ""),
            "threshold": self.threshold,
        }


def load_detector(path: str | Path) -> Detector:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ModelLoadError(
            f"Detector artifact not found at {artifact_path}. "
            "Train it with `python -m app.detection.train` from the backend directory."
        )
    try:
        payload = joblib.load(artifact_path)
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ModelLoadError(f"Could not load detector artifact: {exc}") from exc

    if not isinstance(payload, dict):
        raise ModelLoadError("Detector artifact has an unexpected format")
    model = payload.get("model")
    metadata = payload.get("metadata")
    if model is None or not isinstance(metadata, dict):
        raise ModelLoadError("Detector artifact is missing model or metadata")

    detector = Detector(model=model, metadata=metadata)
    logger.info(
        "Loaded detector %s v%s (threshold %.3f)",
        detector.metadata.get("name"),
        detector.metadata.get("version"),
        detector.threshold,
    )
    return detector


__all__ = ["DetectionResult", "Detector", "ModelLoadError", "load_detector"]
