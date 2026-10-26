"""Runtime configuration for the QuantumSentinel API."""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
ARTIFACT_DIR = BACKEND_ROOT / "artifacts"

MODEL_FILE = Path(
    os.getenv("QS_MODEL_PATH", str(ARTIFACT_DIR / "bb84_detector.joblib"))
).resolve()
MODEL_CARD_FILE = Path(
    os.getenv("QS_MODEL_CARD_PATH", str(ARTIFACT_DIR / "model_card.json"))
).resolve()
FRONTEND_FILE = Path(
    os.getenv("QS_FRONTEND_PATH", str(PROJECT_ROOT / "index.html"))
).resolve()

# PostgreSQL archive for completed sessions.  Only QS_DATABASE_URL is read so
# the service can never accidentally attach to another project's DATABASE_URL.
DATABASE_URL = os.getenv(
    "QS_DATABASE_URL",
    "postgresql://postgres@127.0.0.1:5432/quantum_sentinel",
)
DATABASE_POOL_MIN_SIZE = int(os.getenv("QS_DATABASE_POOL_MIN", "1"))
DATABASE_POOL_MAX_SIZE = int(os.getenv("QS_DATABASE_POOL_MAX", "8"))
DATABASE_CONNECT_TIMEOUT = float(os.getenv("QS_DATABASE_CONNECT_TIMEOUT", "3"))

API_VERSION = "1.1.0"
MODEL_NAME = "BB84 QuantumSentinel Detector"
MODEL_VERSION = "1.0.0"

# Traditional BB84 error-rate threshold used by the dashboard.
STANDARD_QBER_THRESHOLD = 11.0
BLOCK_COUNT = 20

# Allowed browser origins for local development (file:// sends "null").
CORS_ORIGINS = ["null"]
CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
