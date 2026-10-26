from __future__ import annotations

import pytest

from app.simulation.channels import (
    ChannelType,
    build_channel,
    qiskit_available,
    qiskit_channel_model,
)


@pytest.mark.parametrize("channel", list(ChannelType))
@pytest.mark.parametrize("noise", [0, 4, 10, 20])
def test_channel_calibration_matches_requested_noise(
    channel: ChannelType, noise: int
) -> None:
    model = build_channel(channel, noise)
    assert model.expected_qber * 100.0 == pytest.approx(noise, abs=1e-9)
    assert 0.0 <= model.expected_qber <= 0.5
    for value in (
        model.z_error_0,
        model.z_error_1,
        model.x_error_0,
        model.x_error_1,
    ):
        assert 0.0 <= value <= 1.0


@pytest.mark.skipif(not qiskit_available(), reason="qiskit is not installed")
@pytest.mark.parametrize("channel", list(ChannelType))
@pytest.mark.parametrize("noise", [0, 4, 10, 20])
def test_qiskit_channel_matches_analytic_channel(
    channel: ChannelType, noise: int
) -> None:
    analytic = build_channel(channel, noise)
    quantum = qiskit_channel_model(channel, noise)
    assert quantum.expected_qber == pytest.approx(
        analytic.expected_qber, abs=1e-10
    )
    for left, right in (
        (quantum.z_error_0, analytic.z_error_0),
        (quantum.z_error_1, analytic.z_error_1),
        (quantum.x_error_0, analytic.x_error_0),
        (quantum.x_error_1, analytic.x_error_1),
    ):
        assert left == pytest.approx(right, abs=1e-10)


def test_phase_damping_only_errors_in_x_basis() -> None:
    model = build_channel(ChannelType.PHASE_DAMPING, 8)
    assert model.z_error_0 == 0.0
    assert model.z_error_1 == 0.0
    assert model.x_error_0 > 0.0
    assert model.x_error_1 == pytest.approx(model.x_error_0)


def test_amplitude_damping_is_bit_dependent_in_z_basis() -> None:
    model = build_channel(ChannelType.AMPLITUDE_DAMPING, 8)
    assert model.z_error_0 == 0.0
    assert model.z_error_1 > 0.0
    assert model.x_error_0 == pytest.approx(model.x_error_1)
