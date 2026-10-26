"""Quantum channel models used by the BB84 simulator.

The simulator samples bit errors from analytic single-qubit error
probabilities.  The classical engine derives those probabilities directly
from the Kraus structure; the Qiskit engine re-derives them by evolving
density matrices with Qiskit's quantum_info module.  Both paths describe the
same channels.

Noise level is interpreted as the *average post-sift QBER* contributed by the
channel in the absence of an eavesdropper.  Channel parameters are calibrated
to that definition, which keeps the dashboard's noise control intuitive.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import numpy as np


class ChannelType(StrEnum):
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"


class QiskitUnavailableError(RuntimeError):
    """Raised when the Qiskit engine is requested but Qiskit is missing."""


@dataclass(frozen=True, slots=True)
class ChannelModel:
    """Error probabilities for a single-qubit channel (values are fractions)."""

    channel: ChannelType
    noise_percent: float
    parameter: float
    z_error_0: float
    z_error_1: float
    x_error_0: float
    x_error_1: float

    @property
    def expected_z(self) -> float:
        return (self.z_error_0 + self.z_error_1) / 2.0

    @property
    def expected_x(self) -> float:
        return (self.x_error_0 + self.x_error_1) / 2.0

    @property
    def expected_qber(self) -> float:
        """Expected post-sift QBER as a fraction (average over bases/bits)."""
        return (
            self.z_error_0
            + self.z_error_1
            + self.x_error_0
            + self.x_error_1
        ) / 4.0


def qiskit_available() -> bool:
    """Return whether the optional Qiskit dependency can be imported."""
    try:
        return importlib.util.find_spec("qiskit") is not None
    except (ImportError, ValueError):
        return False


def _noise_fraction(noise_percent: float) -> float:
    return min(max(float(noise_percent) / 100.0, 0.0), 0.5)


def _channel_parameter(channel: ChannelType, noise_percent: float) -> float:
    """Choose the physical channel parameter that yields the requested QBER.

    * Depolarizing: q = 2n  (a depolarizing channel with strength q gives q/2
      error in every basis).
    * Amplitude damping: solved from
      n = (gamma/2 + (1 - sqrt(1 - gamma))/2) / 2.
    * Phase damping: Z-basis states are unaffected, X-basis states acquire
      (1 - sqrt(1 - lambda))/2 error.  The observed average over bases is
      therefore (1 - sqrt(1 - lambda))/4, so lambda = 8n(1 - 2n).
    """
    n = _noise_fraction(noise_percent)
    if channel is ChannelType.DEPOLARIZING:
        return min(2.0 * n, 1.0)
    if channel is ChannelType.AMPLITUDE_DAMPING:
        if n >= 0.5:
            return 1.0
        root = math.sqrt(max(0.0, 9.0 - 16.0 * n))
        s = (-1.0 + root) / 2.0
        return min(max(1.0 - s * s, 0.0), 1.0)
    if channel is ChannelType.PHASE_DAMPING:
        if n >= 0.25:
            return 1.0
        return min(max(8.0 * n * (1.0 - 2.0 * n), 0.0), 1.0)
    raise ValueError(f"Unsupported channel: {channel!r}")


@lru_cache(maxsize=512)
def build_channel(channel: ChannelType, noise_percent: float) -> ChannelModel:
    """Build the analytic channel model for the classical engine."""
    if not isinstance(channel, ChannelType):
        channel = ChannelType(channel)
    noise_percent = float(noise_percent)
    n = _noise_fraction(noise_percent)
    parameter = _channel_parameter(channel, noise_percent)

    if channel is ChannelType.DEPOLARIZING:
        error = parameter / 2.0
        z0 = z1 = x0 = x1 = error
    elif channel is ChannelType.AMPLITUDE_DAMPING:
        gamma = parameter
        z0 = 0.0
        z1 = gamma
        x_error = (1.0 - math.sqrt(max(0.0, 1.0 - gamma))) / 2.0
        x0 = x1 = x_error
    elif channel is ChannelType.PHASE_DAMPING:
        lam = parameter
        z0 = z1 = 0.0
        x_error = (1.0 - math.sqrt(max(0.0, 1.0 - lam))) / 2.0
        x0 = x1 = x_error
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unsupported channel: {channel!r}")

    return ChannelModel(
        channel=channel,
        noise_percent=noise_percent,
        parameter=parameter,
        z_error_0=z0,
        z_error_1=z1,
        x_error_0=x0,
        x_error_1=x1,
    )


def _kraus_operators(model: ChannelModel) -> list[np.ndarray]:
    """Kraus operators for the channel's physical parameter."""
    p = model.parameter
    if model.channel is ChannelType.DEPOLARIZING:
        # Complete depolarizing channel: (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z).
        return [
            math.sqrt(max(0.0, 1.0 - 3.0 * p / 4.0)) * np.eye(2, dtype=complex),
            math.sqrt(p / 4.0) * np.array([[0, 1], [1, 0]], dtype=complex),
            math.sqrt(p / 4.0) * np.array([[0, -1j], [1j, 0]], dtype=complex),
            math.sqrt(p / 4.0) * np.array([[1, 0], [0, -1]], dtype=complex),
        ]
    if model.channel is ChannelType.AMPLITUDE_DAMPING:
        return [
            np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - p))]], dtype=complex),
            np.array([[0.0, math.sqrt(p)], [0.0, 0.0]], dtype=complex),
        ]
    if model.channel is ChannelType.PHASE_DAMPING:
        return [
            np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - p))]], dtype=complex),
            np.array([[0.0, 0.0], [0.0, math.sqrt(p)]], dtype=complex),
        ]
    raise ValueError(f"Unsupported channel: {model.channel!r}")


@lru_cache(maxsize=128)
def qiskit_channel_model(
    channel: ChannelType, noise_percent: float
) -> ChannelModel:
    """Derive channel error probabilities with Qiskit density matrices."""
    if not qiskit_available():
        raise QiskitUnavailableError(
            "The Qiskit engine requires the optional 'qiskit' package. "
            "Install it with `pip install qiskit` or use the classical engine."
        )

    base = build_channel(channel, noise_percent)
    if base.parameter == 0.0:
        return base

    try:
        from qiskit.circuit.library import HGate
        from qiskit.quantum_info import DensityMatrix, Kraus, Operator
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise QiskitUnavailableError(
            "Qiskit is installed but could not be imported."
        ) from exc

    kraus = Kraus(_kraus_operators(base))
    hadamard = Operator(HGate())

    def error_for(label: str, bit: int, basis: str) -> float:
        rho = DensityMatrix.from_label(label).evolve(kraus)
        if basis == "x":
            rho = rho.evolve(hadamard)
        p_one = float(np.clip(rho.probabilities([0])[1], 0.0, 1.0))
        return p_one if bit == 0 else 1.0 - p_one

    z0 = error_for("0", 0, "z")
    z1 = error_for("1", 1, "z")
    x0 = error_for("+", 0, "x")
    x1 = error_for("-", 1, "x")

    return ChannelModel(
        channel=base.channel,
        noise_percent=base.noise_percent,
        parameter=base.parameter,
        z_error_0=z0,
        z_error_1=z1,
        x_error_0=x0,
        x_error_1=x1,
    )
