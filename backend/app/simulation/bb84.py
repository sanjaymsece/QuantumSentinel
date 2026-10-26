"""Vectorised BB84 simulation with intercept-resend eavesdropping.

The model is intentionally transparent: Alice prepares random BB84 states,
an optional Eve performs intercept-resend with a per-qubit probability, the
state passes through the selected noise channel, and Bob measures in a random
basis.  Only basis-sifted bits contribute to the QBER.

Ground truth (whether Eve was active and in which mode) is kept in
:class:`SessionResult` for the simulator's diagnostics.  It is deliberately
not part of :class:`Observation` and is never visible to the ML detector.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from app.config import BLOCK_COUNT
from app.simulation.channels import (
    ChannelModel,
    ChannelType,
    QiskitUnavailableError,
    build_channel,
    qiskit_channel_model,
)


class EveMode(StrEnum):
    OFF = "off"
    STEADY = "steady"
    BURST = "burst"


class Engine(StrEnum):
    CLASSICAL = "js"
    QISKIT = "qiskit"


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    eve: EveMode
    eve_strength: float
    noise: float
    qubits: int
    channel: ChannelType
    engine: Engine = Engine.CLASSICAL
    seed: int | None = None

    def validate(self) -> None:
        if not 100 <= int(self.qubits) <= 5000:
            raise ValueError("qubits must be between 100 and 5000")
        if not 0.0 <= float(self.noise) <= 20.0:
            raise ValueError("noise must be between 0 and 20 percent")
        if self.eve is not EveMode.OFF and not 1.0 <= float(self.eve_strength) <= 100.0:
            raise ValueError("eve_strength must be between 1 and 100 percent")
        if self.seed is not None and not 0 <= int(self.seed) <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")


@dataclass(frozen=True, slots=True)
class Observation:
    """Detector-visible measurements.  Contains no eavesdropper ground truth."""

    qber: float
    block_qber: tuple[float, ...]
    x_qber: float
    z_qber: float
    sifted_bits: int
    sifted_rate: float
    expected_qber: float
    expected_x_qber: float
    expected_z_qber: float
    noise_percent: float
    channel: ChannelType


@dataclass(frozen=True, slots=True)
class SessionResult:
    observation: Observation
    attack_present: bool
    attack_mode: str
    errors: int
    key_fingerprint: str
    seed: int
    engine_used: str


def _burst_probabilities(
    rng: np.random.Generator, qubits: int, strength_percent: float
) -> np.ndarray:
    """Return per-qubit interception probabilities for burst mode.

    A burst is a contiguous window of full intercept-resend.  Its length
    scales with Eve strength from roughly 3% to 30% of the session, so a
    strong burst can still keep the *session-average* QBER below the 11%
    rule while producing an unmistakable block-level signature.
    """
    fraction = 0.03 + 0.27 * min(max(strength_percent / 100.0, 0.0), 1.0)
    span = max(1, min(qubits, int(round(qubits * fraction))))
    start = int(rng.integers(0, qubits - span + 1)) if span < qubits else 0
    probabilities = np.zeros(qubits, dtype=np.float64)
    probabilities[start : start + span] = 1.0
    return probabilities


def _channel_for(request: SimulationRequest) -> ChannelModel:
    if request.engine is Engine.QISKIT:
        return qiskit_channel_model(request.channel, request.noise)
    return build_channel(request.channel, request.noise)


def simulate_session(request: SimulationRequest) -> SessionResult:
    """Run one BB84 session and return observations plus simulator diagnostics."""
    if not isinstance(request.eve, EveMode):
        request = replace(request, eve=EveMode(request.eve))
    if not isinstance(request.engine, Engine):
        request = replace(request, engine=Engine(request.engine))
    if not isinstance(request.channel, ChannelType):
        request = replace(request, channel=ChannelType(request.channel))
    request.validate()
    seed = request.seed if request.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)
    qubits = int(request.qubits)

    # Alice: random bit and random basis per qubit.
    alice_bits = rng.integers(0, 2, size=qubits, dtype=np.int8)
    alice_basis = rng.integers(0, 2, size=qubits, dtype=np.int8)  # 0=Z, 1=X

    # Bob: independent random measurement basis.
    bob_basis = rng.integers(0, 2, size=qubits, dtype=np.int8)
    matched = alice_basis == bob_basis

    # Eve: per-qubit intercept-resend probability.
    if request.eve is EveMode.STEADY:
        eve_probability = np.full(
            qubits, min(max(float(request.eve_strength) / 100.0, 0.0), 1.0)
        )
    elif request.eve is EveMode.BURST:
        eve_probability = _burst_probabilities(
            rng, qubits, float(request.eve_strength)
        )
    else:
        eve_probability = np.zeros(qubits, dtype=np.float64)

    channel = _channel_for(request)

    # Probability of a bit error conditioned on Alice's state, Bob's basis,
    # the channel, and Eve.  For an intercepted qubit, averaged over Eve's
    # random basis and outcome, the state is
    #   (1 - p_e/2) rho + (p_e/2) I/2,
    # so the error probability is (1 - p_e/2) * e_channel + p_e * 0.5.
    e_noise = np.where(
        alice_basis == 0,
        np.where(alice_bits == 0, channel.z_error_0, channel.z_error_1),
        np.where(alice_bits == 0, channel.x_error_0, channel.x_error_1),
    )
    e_sifted_state = (1.0 - 0.5 * eve_probability) * e_noise + 0.25 * eve_probability

    # Bits measured in the wrong basis are uniformly random regardless of
    # the state, so they never need to be sampled for the QBER.
    sifted_idx = np.flatnonzero(matched)
    probabilities = e_sifted_state[sifted_idx]
    errors = rng.random(sifted_idx.size) < probabilities
    errors_i8 = errors.astype(np.int8)

    sifted_bits = int(sifted_idx.size)
    error_count = int(errors_i8.sum())
    qber = (
        float(error_count) / sifted_bits * 100.0 if sifted_bits else 0.0
    )

    # Per-basis error rates after sifting.
    sifted_basis = alice_basis[sifted_idx]
    x_errors = errors_i8[sifted_basis == 1]
    z_errors = errors_i8[sifted_basis == 0]
    x_qber = float(x_errors.mean() * 100.0) if x_errors.size else 0.0
    z_qber = float(z_errors.mean() * 100.0) if z_errors.size else 0.0

    # Ordered block QBER series for the dashboard and the detector.
    block_qber = tuple(
        float(block.mean() * 100.0) if block.size else 0.0
        for block in np.array_split(errors_i8, BLOCK_COUNT)
    )

    key_bits = (alice_bits[sifted_idx] ^ errors_i8).astype(np.uint8)
    key_fingerprint = hashlib.blake2b(
        key_bits.tobytes(), digest_size=8, person=b"quantumsentinel"
    ).hexdigest()

    observation = Observation(
        qber=qber,
        block_qber=block_qber,
        x_qber=x_qber,
        z_qber=z_qber,
        sifted_bits=sifted_bits,
        sifted_rate=sifted_bits / qubits,
        expected_qber=channel.expected_qber * 100.0,
        expected_x_qber=channel.expected_x * 100.0,
        expected_z_qber=channel.expected_z * 100.0,
        noise_percent=float(request.noise),
        channel=request.channel,
    )

    return SessionResult(
        observation=observation,
        attack_present=request.eve != EveMode.OFF,
        attack_mode=(
            "none" if request.eve == EveMode.OFF else request.eve.value
        ),
        errors=error_count,
        key_fingerprint=key_fingerprint,
        seed=seed,
        engine_used=(
            "qiskit" if request.engine is Engine.QISKIT else "classical"
        ),
    )


__all__ = [
    "Engine",
    "EveMode",
    "Observation",
    "QiskitUnavailableError",
    "SessionResult",
    "SimulationRequest",
    "simulate_session",
]
