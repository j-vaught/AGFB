"""Tests for scale and frequency generator slices."""

from __future__ import annotations

import math

import torch
from test_analytic_gradients import _check_signal_mask

from agfb_generators.anisotropic_blob import anisotropic_blob
from agfb_generators.chirp import chirp
from agfb_generators.gabor_packet import gabor_packet


def test_anisotropic_blob_gradient_matches_fd() -> None:
    """Check the rotated anisotropic blob gradient against finite differences."""
    f = anisotropic_blob(
        256,
        256,
        length_sigma=18.0,
        width_sigma=11.0,
        angle_rad=math.radians(27.0),
        center_x=3.0,
        center_y=-5.0,
        amplitude=1.3,
    )
    _check_signal_mask(f, rel_tol=1e-3, name="anisotropic_blob")


def test_chirp_gradient_matches_fd() -> None:
    """Check the chirp gradient against finite differences."""
    f = chirp(
        256,
        256,
        base_frequency=0.025,
        frequency_slope=0.00012,
        angle_rad=math.radians(33.0),
        amplitude=0.9,
        phase_rad=0.3,
        center_offset=4.0,
    )
    _check_signal_mask(f, rel_tol=1e-2, name="chirp")


def test_chirp_batched_consistent_with_scalar() -> None:
    """Verify batched chirps match repeated scalar renders."""
    height = 80
    width = 84
    base_frequency = torch.tensor([0.012, 0.018, 0.024])
    frequency_slope = torch.tensor([0.00008, 0.00012, 0.00016])
    angle = torch.tensor([0.0, math.radians(25.0), math.radians(50.0)])
    amplitude = torch.tensor([0.7, 1.0, 1.3])
    phase = torch.tensor([0.0, 0.25, 0.5])
    center_offset = torch.tensor([-3.0, 0.0, 3.0])

    out = chirp(
        height,
        width,
        base_frequency=base_frequency,
        frequency_slope=frequency_slope,
        angle_rad=angle,
        amplitude=amplitude,
        phase_rad=phase,
        center_offset=center_offset,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = chirp(
            height,
            width,
            base_frequency=float(base_frequency[i]),
            frequency_slope=float(frequency_slope[i]),
            angle_rad=float(angle[i]),
            amplitude=float(amplitude[i]),
            phase_rad=float(phase[i]),
            center_offset=float(center_offset[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_chirp_infers_tensor_device() -> None:
    """Verify tensor inputs keep chirp output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = chirp(
        20,
        24,
        base_frequency=torch.tensor([0.012, 0.018], device=device),
        frequency_slope=0.00012,
        angle_rad=torch.tensor([0.0, math.radians(25.0)], device=device),
        amplitude=1.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_chirp_honors_requested_device() -> None:
    """Verify scalar chirp inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = chirp(
        20,
        24,
        base_frequency=0.012,
        frequency_slope=0.00012,
        angle_rad=math.radians(25.0),
        amplitude=1.0,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_gabor_packet_gradient_matches_fd() -> None:
    """Check the localized Gabor gradient against finite differences."""
    f = gabor_packet(
        256,
        256,
        carrier_frequency=0.03,
        angle_rad=math.radians(24.0),
        envelope_length_sigma=24.0,
        envelope_width_sigma=14.0,
        center_x=4.0,
        center_y=-6.0,
        amplitude=1.1,
        phase_rad=0.2,
    )
    _check_signal_mask(f, rel_tol=1e-2, name="gabor_packet")


def test_gabor_packet_batched_consistent_with_scalar() -> None:
    """Verify batched Gabor packets match repeated scalar renders."""
    height = 84
    width = 88
    carrier_frequency = torch.tensor([0.018, 0.026, 0.034])
    angle = torch.tensor([0.0, math.radians(24.0), math.radians(48.0)])
    envelope_length_sigma = torch.tensor([18.0, 24.0, 30.0])
    envelope_width_sigma = torch.tensor([8.0, 12.0, 16.0])
    center_x = torch.tensor([-4.0, 0.0, 4.0])
    center_y = torch.tensor([3.0, -3.0, 1.0])
    amplitude = torch.tensor([0.75, 1.0, 1.25])
    phase = torch.tensor([0.0, 0.2, 0.4])

    out = gabor_packet(
        height,
        width,
        carrier_frequency=carrier_frequency,
        angle_rad=angle,
        envelope_length_sigma=envelope_length_sigma,
        envelope_width_sigma=envelope_width_sigma,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        phase_rad=phase,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = gabor_packet(
            height,
            width,
            carrier_frequency=float(carrier_frequency[i]),
            angle_rad=float(angle[i]),
            envelope_length_sigma=float(envelope_length_sigma[i]),
            envelope_width_sigma=float(envelope_width_sigma[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
            phase_rad=float(phase[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_gabor_packet_honors_requested_device() -> None:
    """Verify scalar Gabor packet inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = gabor_packet(
        20,
        24,
        carrier_frequency=0.025,
        angle_rad=math.radians(24.0),
        envelope_length_sigma=18.0,
        envelope_width_sigma=9.0,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_gabor_packet_infers_tensor_device() -> None:
    """Verify tensor inputs keep Gabor packet output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = gabor_packet(
        20,
        24,
        carrier_frequency=torch.tensor([0.018, 0.026], device=device),
        angle_rad=torch.tensor([0.0, math.radians(24.0)], device=device),
        envelope_length_sigma=18.0,
        envelope_width_sigma=9.0,
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_anisotropic_blob_batched_consistent_with_scalar() -> None:
    """Verify batched anisotropic blobs match repeated scalar renders."""
    H = 96
    W = 80
    length_sigma = torch.tensor([7.0, 10.0, 13.0])
    width_sigma = torch.tensor([4.0, 6.0, 8.0])
    angle = torch.tensor([0.0, math.radians(25.0), math.radians(70.0)])
    center_x = torch.tensor([-3.0, 0.0, 4.0])
    center_y = torch.tensor([2.0, -5.0, 1.0])
    amplitude = torch.tensor([0.7, 1.0, 1.4])

    out = anisotropic_blob(
        H,
        W,
        length_sigma=length_sigma,
        width_sigma=width_sigma,
        angle_rad=angle,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
    )

    assert out.I.shape == (3, H, W)
    assert out.g.shape == (3, 2, H, W)
    for i in range(3):
        single = anisotropic_blob(
            H,
            W,
            length_sigma=float(length_sigma[i]),
            width_sigma=float(width_sigma[i]),
            angle_rad=float(angle[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_anisotropic_blob_honors_requested_device() -> None:
    """Verify anisotropic blob tensors stay on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = anisotropic_blob(
        32,
        36,
        length_sigma=torch.tensor([6.0, 9.0]),
        width_sigma=3.0,
        angle_rad=torch.tensor([0.0, math.radians(25.0)]),
        center_x=torch.tensor([-2.0, 3.0]),
        center_y=1.5,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device
    assert frame.I.shape == (2, 32, 36)
    assert frame.g.shape == (2, 2, 32, 36)


def test_anisotropic_blob_infers_tensor_device() -> None:
    """Verify tensor inputs keep anisotropic blob output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = anisotropic_blob(
        20,
        22,
        length_sigma=torch.tensor([6.0, 9.0], device=device),
        width_sigma=3.0,
        angle_rad=torch.tensor([0.0, math.radians(25.0)], device=device),
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_scale_frequency_generator_shapes_and_dtype() -> None:
    """Check shape and dtype conventions for the new generator slice."""
    dtype = torch.float64
    H = 40
    W = 44

    blob = anisotropic_blob(
        H,
        W,
        length_sigma=torch.tensor([5.0, 6.0], dtype=dtype),
        width_sigma=3.5,
        angle_rad=torch.tensor([0.1, 0.4], dtype=dtype),
        dtype=dtype,
    )
    ch = chirp(
        H,
        W,
        base_frequency=torch.tensor([0.015, 0.02], dtype=dtype),
        frequency_slope=0.0001,
        angle_rad=0.2,
        dtype=dtype,
    )
    gabor = gabor_packet(
        H,
        W,
        carrier_frequency=0.02,
        angle_rad=torch.tensor([0.0, 0.3], dtype=dtype),
        envelope_length_sigma=9.0,
        envelope_width_sigma=5.0,
        dtype=dtype,
    )

    for frame in (blob, ch, gabor):
        assert frame.I.shape == (2, H, W)
        assert frame.g.shape == (2, 2, H, W)
        assert frame.I.dtype == dtype
        assert frame.g.dtype == dtype
