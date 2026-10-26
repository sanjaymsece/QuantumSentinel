from __future__ import annotations

import math

import pytest

from app.config import STANDARD_QBER_THRESHOLD
from app.simulation.bb84 import (
    Engine,
    EveMode,
    SimulationRequest,
    simulate_session,
)
from app.simulation.channels import ChannelType


def _run(
    eve: EveMode = EveMode.OFF,
    strength: float = 50,
    noise: float = 4,
    qubits: int = 2000,
    channel: ChannelType = ChannelType.DEPOLARIZING,
    engine: Engine = Engine.CLASSICAL,
    seed: int = 7,
):
    return simulate_session(
        SimulationRequest(
            eve=eve,
            eve_strength=strength,
            noise=noise,
            qubits=qubits,
            channel=channel,
            engine=engine,
            seed=seed,
        )
    )


def test_simulation_is_deterministic_for_a_seed() -> None:
    first = _run(seed=12345)
    second = _run(seed=12345)
    assert first.observation == second.observation
    assert first.key_fingerprint == second.key_fingerprint
    assert first.seed == 12345


def test_zero_noise_no_eve_has_zero_qber() -> None:
    result = _run(noise=0, seed=3)
    assert result.observation.qber == 0.0
    assert result.observation.expected_qber == 0.0
    assert result.observation.sifted_bits == pytest.approx(1000, abs=80)
    assert 0.44 < result.observation.sifted_rate < 0.56


def test_full_intercept_resend_adds_about_25_percent_error() -> None:
    result = _run(eve=EveMode.STEADY, strength=100, noise=0, seed=11)
    assert result.observation.qber == pytest.approx(25.0, abs=1.8)


def test_steady_eve_below_threshold_is_still_an_attack() -> None:
    result = _run(eve=EveMode.STEADY, strength=20, noise=4, seed=5)
    assert result.attack_present is True
    assert result.attack_mode == "steady"
    assert result.observation.qber < STANDARD_QBER_THRESHOLD
    assert result.observation.expected_qber == pytest.approx(4.0)


def test_burst_produces_block_signature_but_low_average_qber() -> None:
    result = _run(eve=EveMode.BURST, strength=50, noise=4, seed=1)
    observation = result.observation
    assert result.attack_present is True
    assert result.attack_mode == "burst"
    assert observation.qber < STANDARD_QBER_THRESHOLD
    assert max(observation.block_qber) > STANDARD_QBER_THRESHOLD
    assert len(observation.block_qber) == 20
    assert all(0.0 <= block <= 100.0 for block in observation.block_qber)


def test_off_mode_has_no_attack_ground_truth() -> None:
    result = _run(eve=EveMode.OFF, seed=9)
    assert result.attack_present is False
    assert result.attack_mode == "none"


def test_phase_damping_basis_asymmetry_is_reflected_in_observation() -> None:
    result = _run(
        eve=EveMode.OFF,
        noise=8,
        channel=ChannelType.PHASE_DAMPING,
        seed=21,
    )
    observation = result.observation
    assert observation.z_qber == 0.0
    assert observation.x_qber > 5.0
    assert observation.x_qber > observation.z_qber


def test_qiskit_engine_matches_classical_result() -> None:
    pytest.importorskip("qiskit")
    classical = _run(eve=EveMode.BURST, seed=1, engine=Engine.CLASSICAL)
    quantum = _run(eve=EveMode.BURST, seed=1, engine=Engine.QISKIT)
    assert quantum.engine_used == "qiskit"
    assert quantum.observation.qber == pytest.approx(
        classical.observation.qber, abs=0.2
    )
    assert quantum.observation.block_qber == pytest.approx(
        classical.observation.block_qber, abs=0.2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qubits": 99},
        {"qubits": 5001},
        {"noise": -1},
        {"noise": 21},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    base = dict(
        eve=EveMode.OFF,
        eve_strength=50,
        noise=4,
        qubits=2000,
        channel=ChannelType.DEPOLARIZING,
        seed=1,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        simulate_session(SimulationRequest(**base))


def test_strength_validation_for_attacks() -> None:
    with pytest.raises(ValueError):
        _run(eve=EveMode.BURST, strength=0)
    with pytest.raises(ValueError):
        _run(eve=EveMode.STEADY, strength=101)


def test_qber_is_fraction_of_sifted_errors() -> None:
    result = _run(seed=33)
    observation = result.observation
    assert observation.sifted_bits > 0
    expected = result.errors / observation.sifted_bits * 100.0
    assert observation.qber == pytest.approx(expected, abs=1e-9)
    assert math.isfinite(observation.qber)
