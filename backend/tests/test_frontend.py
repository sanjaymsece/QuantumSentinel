from __future__ import annotations

import re

from app.config import FRONTEND_FILE

REQUIRED_IDS = (
    "btn-off",
    "btn-steady",
    "btn-burst",
    "eve-strength",
    "val-eve-strength",
    "channel",
    "noise",
    "val-noise",
    "qubits",
    "val-qubits",
    "btn-js",
    "btn-qiskit",
    "run-btn",
    "loading-icon",
    "stat-qber",
    "stat-qber-status",
    "card-standard",
    "stat-standard",
    "stat-standard-desc",
    "card-ml",
    "stat-ml",
    "stat-ml-desc",
    "qberChart",
    "terminal",
    "health-dot",
    "health-text",
    "session-status",
    "session-id",
    "chart-note",
    "archive-title",
    "archive-meta",
    "archive-refresh",
    "archive-body",
)


def test_dashboard_has_required_controls() -> None:
    html = FRONTEND_FILE.read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    missing = sorted(set(REQUIRED_IDS) - ids)
    assert not missing, f"dashboard is missing required IDs: {missing}"


def test_dashboard_has_no_placeholder_logic() -> None:
    html = FRONTEND_FILE.read_text(encoding="utf-8").lower()
    for token in ("tailwind", "isevereal", "math.random", "mock simulation"):
        assert token not in html


def test_dashboard_calls_the_real_api() -> None:
    html = FRONTEND_FILE.read_text(encoding="utf-8")
    assert "fetch(API_BASE + '/run'" in html
    assert "API_BASE + '/health'" in html
    assert "API_BASE + '/sessions?limit=8'" in html
    assert "PostgreSQL" in html
    assert "channel:" in html
    assert "engine:" in html
    assert "?api=" in html
    assert "chart.js@4.4.7" in html


def test_dashboard_renders_backend_block_series() -> None:
    html = FRONTEND_FILE.read_text(encoding="utf-8")
    assert "block_qber" in html
    assert "ml_detected" in html
    assert "attack_present" in html
    assert "standard_detected" in html
    assert "ml_confidence" in html
