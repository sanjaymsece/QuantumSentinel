"""Pydantic request and response contracts for the QuantumSentinel API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RunRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "eve": "burst",
                "eve_strength": 50,
                "noise": 4,
                "qubits": 2000,
                "channel": "depolarizing",
                "engine": "js",
            }
        },
    )

    eve: Literal["off", "steady", "burst"] = Field(
        "off", description="Eavesdropper strategy."
    )
    eve_strength: int = Field(
        50, ge=1, le=100, description="Interception strength in percent."
    )
    noise: int = Field(
        4, ge=0, le=20, description="Nominal channel noise level in percent QBER."
    )
    qubits: int = Field(
        2000, ge=100, le=5000, description="Number of transmitted qubits."
    )
    channel: Literal[
        "depolarizing", "amplitude_damping", "phase_damping"
    ] = Field("depolarizing", description="Quantum noise channel.")
    engine: Literal["js", "qiskit"] = Field(
        "js", description="Simulation engine: classical NumPy or Qiskit."
    )
    seed: int | None = Field(
        None,
        ge=0,
        le=2**32 - 1,
        description="Optional RNG seed for reproducible sessions.",
    )


class ModelSummary(BaseModel):
    name: str
    version: str
    trained_at: str
    threshold: float = Field(
        description="Probability threshold used for the ML verdict."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    qiskit_available: bool
    database_connected: bool = False
    database_backend: Literal["postgresql"] = "postgresql"
    database_name: str | None = None
    model: ModelSummary | None = None


class RunResponse(BaseModel):
    session_id: str
    engine: Literal["classical", "qiskit"]
    engine_requested: Literal["js", "qiskit"]
    warning: str | None = None

    qber: float = Field(description="Post-sift quantum bit error rate in percent.")
    threshold: float = Field(description="Traditional BB84 threshold in percent.")
    standard_detected: bool
    ml_detected: bool
    ml_confidence: float = Field(
        description="Confidence in the displayed verdict, in percent."
    )
    attack_probability: float = Field(ge=0.0, le=1.0)

    # Simulator ground truth.  Exposed so the demo can explain when the
    # threshold rule missed an attack; never used to compute ML results.
    attack_present: bool
    attack_mode: Literal["none", "steady", "burst"]

    block_qber: list[float] = Field(
        min_length=20, max_length=20, description="Per-block QBER series."
    )
    sifted_bits: int
    sifted_rate: float
    x_qber: float
    z_qber: float
    expected_qber: float
    elapsed_ms: float
    seed: int
    model: ModelSummary


class SessionSummary(BaseModel):
    """One row in the archived-session listing."""

    session_id: UUID
    created_at: datetime
    engine: Literal["classical", "qiskit"]
    eve_mode: Literal["off", "steady", "burst"]
    eve_strength: int
    noise_percent: int
    qubits: int
    channel: Literal["depolarizing", "amplitude_damping", "phase_damping"]
    qber: float
    threshold: float
    standard_detected: bool
    ml_detected: bool
    ml_confidence: float
    attack_probability: float
    attack_present: bool
    attack_mode: Literal["none", "steady", "burst"]
    elapsed_ms: float


class SessionListResponse(BaseModel):
    items: list[SessionSummary]
    total: int
    limit: int
    offset: int


class SessionDetail(SessionSummary):
    """Full archived record, including the exact ML inputs and block series."""

    engine_requested: Literal["js", "qiskit"]
    seed: int
    block_qber: list[float]
    sifted_bits: int
    sifted_rate: float
    x_qber: float
    z_qber: float
    expected_qber: float
    model_name: str
    model_version: str
    model_threshold: float
    features: dict[str, float]
