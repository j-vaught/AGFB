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
        sigma_u=18.0,
        sigma_v=11.0,
        theta_rad=math.radians(27.0),
        x0=3.0,
        y0=-5.0,
        contrast=1.3,
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
    sigma_u = torch.tensor([7.0, 10.0, 13.0])
    sigma_v = torch.tensor([4.0, 6.0, 8.0])
    theta = torch.tensor([0.0, math.radians(25.0), math.radians(70.0)])
    x0 = torch.tensor([-3.0, 0.0, 4.0])
    y0 = torch.tensor([2.0, -5.0, 1.0])
    contrast = torch.tensor([0.7, 1.0, 1.4])

    out = anisotropic_blob(
        H,
        W,
        sigma_u=sigma_u,
        sigma_v=sigma_v,
        theta_rad=theta,
        x0=x0,
        y0=y0,
        contrast=contrast,
    )

    assert out.I.shape == (3, H, W)
    assert out.g.shape == (3, 2, H, W)
    for i in range(3):
        single = anisotropic_blob(
            H,
            W,
            sigma_u=float(sigma_u[i]),
            sigma_v=float(sigma_v[i]),
            theta_rad=float(theta[i]),
            x0=float(x0[i]),
            y0=float(y0[i]),
            contrast=float(contrast[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_scale_frequency_generator_shapes_and_dtype() -> None:
    """Check shape and dtype conventions for the new generator slice."""
    dtype = torch.float64
    H = 40
    W = 44

    blob = anisotropic_blob(
        H,
        W,
        sigma_u=torch.tensor([5.0, 6.0], dtype=dtype),
        sigma_v=3.5,
        theta_rad=torch.tensor([0.1, 0.4], dtype=dtype),
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
