import math

import pytest
import torch

import agfb_generators as ag
from agfb_generators.base import normalize_contrast

TOLERANCE = 2e-5


GENERATOR_CASES = [
    (ag.gaussian_blob, {"scale_sigma": 12.0}),
    (
        ag.anisotropic_blob,
        {"length_sigma": 22.0, "width_sigma": 8.0, "angle_rad": 0.4},
    ),
    (ag.gaussian_ridge, {"width_sigma": 8.0, "angle_rad": 0.0}),
    (
        ag.asymmetric_ridge,
        {"negative_sigma": 5.0, "positive_sigma": 12.0, "angle_rad": 0.0},
    ),
    (ag.curved_ridge, {"width_sigma": 6.0, "angle_rad": 0.0, "curvature": 0.002}),
    (ag.curved_arc, {"radius": 45.0}),
    (ag.smoothed_step, {}),
    (ag.finite_ramp, {"ramp_width": 64.0, "angle_rad": 0.0}),
    (ag.smoothed_ramp, {}),
    (ag.mach_band, {}),
    (ag.roof_profile, {}),
    (ag.smoothed_bar, {}),
    (ag.sinusoid, {}),
    (
        ag.chirp,
        {"base_frequency": 0.03, "frequency_slope": 0.0005, "angle_rad": 0.0},
    ),
    (
        ag.gabor_packet,
        {
            "carrier_frequency": 0.07,
            "angle_rad": 0.0,
            "envelope_length_sigma": 20.0,
            "envelope_width_sigma": 8.0,
        },
    ),
    (ag.polynomial, {}),
    (ag.smoothed_l_junction, {"arm_width": 18.0}),
    (ag.hard_l_junction, {"arm_width": 18.0}),
    (ag.smoothed_t_junction, {}),
    (ag.hard_t_junction, {}),
    (ag.smoothed_x_junction, {}),
    (ag.hard_x_junction, {}),
    (ag.smoothed_y_junction, {}),
    (ag.hard_y_junction, {}),
    (ag.vessel_crossing, {}),
    (ag.vessel_bifurcation, {}),
]


@pytest.mark.parametrize(("generator", "kwargs"), GENERATOR_CASES)
@pytest.mark.parametrize("amplitude", [1.0, 0.5, 0.0])
def test_public_generators_realize_requested_intensity_contract(generator, kwargs, amplitude):
    frame = generator(128, 128, amplitude=amplitude, **kwargs)

    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()
    assert float(frame.I.min()) >= -TOLERANCE
    assert float(frame.I.max()) <= 1.0 + TOLERANCE
    assert float(torch.abs((frame.I.amax() - frame.I.amin()) - amplitude)) <= TOLERANCE
    if amplitude == 0.0:
        assert torch.allclose(frame.I, torch.full_like(frame.I, 0.5), atol=TOLERANCE)
        assert torch.allclose(frame.g, torch.zeros_like(frame.g), atol=TOLERANCE)


def test_batched_amplitudes_are_applied_per_frame():
    amplitudes = torch.tensor([1.0, 0.5, 0.0])
    frame = ag.sinusoid(96, 96, amplitude=amplitudes, phase_rad=torch.tensor([0.0, 0.2, 0.4]))

    spans = frame.I.amax(dim=(1, 2)) - frame.I.amin(dim=(1, 2))
    assert torch.allclose(spans, amplitudes, atol=TOLERANCE)
    assert torch.all(frame.I >= -TOLERANCE)
    assert torch.all(frame.I <= 1.0 + TOLERANCE)


def test_vessel_crossing_uses_branch_amplitudes_as_relative_weights():
    balanced = ag.vessel_crossing(96, 96, branch_a_amplitude=1.0, branch_b_amplitude=1.0)
    imbalanced = ag.vessel_crossing(96, 96, branch_a_amplitude=1.0, branch_b_amplitude=0.2)

    assert torch.isclose(balanced.I.amax() - balanced.I.amin(), torch.tensor(1.0), atol=TOLERANCE)
    assert torch.isclose(
        imbalanced.I.amax() - imbalanced.I.amin(), torch.tensor(1.0), atol=TOLERANCE
    )
    assert not torch.allclose(balanced.I, imbalanced.I)


def test_polynomial_supports_batched_amplitude_with_single_coefficient_matrix():
    coefficients = torch.tensor([[0.0, 1.0], [0.5, 0.0]])
    amplitudes = torch.tensor([1.0, 0.25])

    frame = ag.polynomial(
        64, 64, coefficients=coefficients, coordinate_scale=64.0, amplitude=amplitudes
    )

    spans = frame.I.amax(dim=(1, 2)) - frame.I.amin(dim=(1, 2))
    assert torch.allclose(spans, amplitudes, atol=TOLERANCE)


def test_normalize_contrast_applies_identical_gradient_scale_without_offset():
    raw = torch.arange(6.0).view(1, 2, 3)
    gx = torch.ones_like(raw)
    gy = torch.full_like(raw, 2.0)
    amplitude = torch.tensor([0.5]).view(1, 1, 1)

    frame = normalize_contrast(raw, gx, gy, amplitude)

    assert torch.isclose(frame.I.amin(), torch.tensor(0.25), atol=TOLERANCE)
    assert torch.isclose(frame.I.amax(), torch.tensor(0.75), atol=TOLERANCE)
    assert torch.allclose(frame.gx, torch.full_like(raw, 0.1), atol=TOLERANCE)
    assert torch.allclose(frame.gy, torch.full_like(raw, 0.2), atol=TOLERANCE)


def test_positive_amplitude_rejects_zero_span_raw_field():
    raw = torch.ones(1, 4, 4)
    amplitude = torch.tensor([1.0]).view(1, 1, 1)

    with pytest.raises(ValueError, match="span must be positive"):
        normalize_contrast(raw, raw, raw, amplitude)


@pytest.mark.parametrize("bad_amplitude", [-0.1, 1.1, math.inf, math.nan])
def test_public_amplitude_must_be_in_unit_interval(bad_amplitude):
    with pytest.raises(ValueError, match="amplitude"):
        ag.sinusoid(32, 32, amplitude=bad_amplitude)
