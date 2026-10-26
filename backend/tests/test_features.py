from __future__ import annotations

import numpy as np
import pytest

from app.detection.features import FEATURE_NAMES, extract_features
from app.simulation.bb84 import (
    Engine,
    EveMode,
    SimulationRequest,
    simulate_session,
)
from app.simulation.channels import ChannelType


def _observation(eve: EveMode = EveMode.BURST, seed: int = 1):
    return simulate_session(
        SimulationRequest(
            eve=eve,
            eve_strength=50,
            noise=4,
            qubits=2000,
            channel=ChannelType.DEPOLARIZING,
            engine=Engine.CLASSICAL,
            seed=seed,
        )
    ).observation


def test_feature_vector_shape_and_finiteness() -> None:
    vector = extract_features(_observation())
    assert vector.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vector).all()


def test_feature_names_contain_no_ground_truth() -> None:
    forbidden = ("eve", "attack", "strength", "label", "burst", "steady")
    names = " ".join(FEATURE_NAMES).lower()
    for token in forbidden:
        assert token not in names
    assert FEATURE_NAMES[0] == "noise_percent"
    assert "qber" in FEATURE_NAMES


def test_observations_carry_no_ground_truth_fields() -> None:
    observation = _observation()
    assert not hasattr(observation, "attack_present")
    assert not hasattr(observation, "eve")
    assert not hasattr(observation, "eve_strength")
    assert not hasattr(observation, "attack_mode")


def test_burst_features_differ_from_benign_features() -> None:
    burst = extract_features(_observation(EveMode.BURST, seed=1))
    benign = extract_features(_observation(EveMode.OFF, seed=1))
    assert not np.allclose(burst, benign)
    names = {name: index for index, name in enumerate(FEATURE_NAMES)}
    assert burst[names["block_max"]] > benign[names["block_max"]]
    assert burst[names["block_max_z"]] > benign[names["block_max_z"]]


def test_feature_extraction_rejects_wrong_block_count() -> None:
    observation = _observation()
    broken = observation.__class__(
        qber=observation.qber,
        block_qber=observation.block_qber[:10],
        x_qber=observation.x_qber,
        z_qber=observation.z_qber,
        sifted_bits=observation.sifted_bits,
        sifted_rate=observation.sifted_rate,
        expected_qber=observation.expected_qber,
        expected_x_qber=observation.expected_x_qber,
        expected_z_qber=observation.expected_z_qber,
        noise_percent=observation.noise_percent,
        channel=observation.channel,
    )
    with pytest.raises(ValueError):
        extract_features(broken)
