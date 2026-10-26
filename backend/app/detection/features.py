"""Feature extraction for the BB84 intrusion detector.

Every feature is either (a) measured after basis sifting or (b) operational
context that a deployed QKD system would know (nominal channel noise and
channel type).  The eavesdropper mode, strength, and attack label are never
used.  This is enforced by the fact that :class:`Observation` has no such
fields.
"""

from __future__ import annotations

import math

import numpy as np

from app.config import BLOCK_COUNT, STANDARD_QBER_THRESHOLD
from app.simulation.bb84 import Observation
from app.simulation.channels import ChannelType


FEATURE_NAMES: tuple[str, ...] = (
    "noise_percent",
    "qber",
    "expected_qber",
    "qber_excess",
    "qber_ratio",
    "qber_z",
    "x_qber",
    "z_qber",
    "expected_x_qber",
    "expected_z_qber",
    "x_excess",
    "z_excess",
    "sift_rate",
    "sifted_bits",
    "block_size",
    "block_mean",
    "block_std",
    "block_min",
    "block_max",
    "block_range",
    "block_median",
    "block_p90",
    "block_p95",
    "block_max_excess",
    "block_max_ratio",
    "block_std_ratio",
    "block_max_z",
    "blocks_over_threshold",
    "block_autocorr",
    "block_kurtosis",
    "channel_depolarizing",
    "channel_amplitude_damping",
    "channel_phase_damping",
)

def _autocorrelation(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std < 1e-9 or right_std < 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _excess_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    std = float(values.std())
    if std < 1e-9:
        return 0.0
    centred = values - values.mean()
    return float(np.mean(centred**4) / std**4 - 3.0)


def _safe_ratio(numerator: float, denominator: float, floor: float = 0.5) -> float:
    return float(numerator) / max(abs(float(denominator)), floor)


def extract_features(observation: Observation) -> np.ndarray:
    """Convert one session's measurements into the detector feature vector."""
    block = np.asarray(observation.block_qber, dtype=np.float64)
    if block.size != BLOCK_COUNT:
        raise ValueError(
            f"expected {BLOCK_COUNT} block QBER values, got {block.size}"
        )

    qber = float(observation.qber)
    expected = float(observation.expected_qber)
    sifted_bits = max(int(observation.sifted_bits), 1)
    block_size = max(sifted_bits / BLOCK_COUNT, 1.0)
    excess = qber - expected
    x_excess = float(observation.x_qber) - float(observation.expected_x_qber)
    z_excess = float(observation.z_qber) - float(observation.expected_z_qber)

    # Binomial standard deviation of post-sift QBER, in percentage points.
    p = min(max(expected / 100.0, 0.0), 1.0)
    qber_sigma = math.sqrt(p * (1.0 - p) / sifted_bits) * 100.0
    block_sigma = math.sqrt(p * (1.0 - p) / block_size) * 100.0
    sigma_floor = max(qber_sigma, 0.1)
    block_sigma_floor = max(block_sigma, 0.1)

    values = (
        float(observation.noise_percent),
        qber,
        expected,
        excess,
        _safe_ratio(qber, expected),
        excess / sigma_floor,
        float(observation.x_qber),
        float(observation.z_qber),
        float(observation.expected_x_qber),
        float(observation.expected_z_qber),
        x_excess,
        z_excess,
        float(observation.sifted_rate),
        float(observation.sifted_bits),
        block_size,
        float(block.mean()),
        float(block.std()),
        float(block.min()),
        float(block.max()),
        float(block.max() - block.min()),
        float(np.median(block)),
        float(np.percentile(block, 90)),
        float(np.percentile(block, 95)),
        float(block.max() - expected),
        _safe_ratio(float(block.max()), qber),
        float(block.std()) / block_sigma_floor,
        float(block.max() - expected) / block_sigma_floor,
        float(np.mean(block > STANDARD_QBER_THRESHOLD)),
        _autocorrelation(block),
        _excess_kurtosis(block),
        1.0 if observation.channel is ChannelType.DEPOLARIZING else 0.0,
        1.0 if observation.channel is ChannelType.AMPLITUDE_DAMPING else 0.0,
        1.0 if observation.channel is ChannelType.PHASE_DAMPING else 0.0,
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),):
        raise AssertionError("feature vector shape mismatch")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
