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
        freq0=0.025,
        chirp_rate=0.00012,
        theta_rad=math.radians(33.0),
        contrast=0.9,
        phase=0.3,
        u0=4.0,
    )
    _check_signal_mask(f, rel_tol=1e-2, name="chirp")


def test_gabor_packet_gradient_matches_fd() -> None:
    """Check the localized Gabor gradient against finite differences."""
    f = gabor_packet(
        256,
        256,
        freq=0.03,
        theta_rad=math.radians(24.0),
        sigma_u=24.0,
        sigma_v=14.0,
        x0=4.0,
        y0=-6.0,
        contrast=1.1,
        phase=0.2,
    )
    _check_signal_mask(f, rel_tol=1e-2, name="gabor_packet")


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
        freq0=torch.tensor([0.015, 0.02], dtype=dtype),
        chirp_rate=0.0001,
        theta_rad=0.2,
        dtype=dtype,
    )
    gabor = gabor_packet(
        H,
        W,
        freq=0.02,
        theta_rad=torch.tensor([0.0, 0.3], dtype=dtype),
        sigma_u=9.0,
        sigma_v=5.0,
        dtype=dtype,
    )

    for frame in (blob, ch, gabor):
        assert frame.I.shape == (2, H, W)
        assert frame.g.shape == (2, 2, H, W)
        assert frame.I.dtype == dtype
        assert frame.g.dtype == dtype
