"""Behavior and gradient tests for transition-profile generators."""

from __future__ import annotations

import math

import torch

from agfb_generators.finite_ramp import finite_ramp
from agfb_generators.mach_band import mach_band
from agfb_generators.roof_profile import roof_profile
from agfb_generators.smoothed_ramp import smoothed_ramp
from tests.test_analytic_gradients import _check_signal_mask


def test_finite_ramp_horizontal_profile_and_gradient() -> None:
    """Check the exact sampled profile away from the ramp kinks."""
    f = finite_ramp(5, 17, ramp_width=8.0, angle_rad=0.0, amplitude=2.0)
    x = torch.arange(17, dtype=torch.float32) - 8.0
    expected_I = 2.0 * torch.clamp((x + 4.0) / 8.0, min=0.0, max=1.0)
    expected_gx = torch.where((x > -4.0) & (x < 4.0), torch.tensor(0.25), torch.tensor(0.0))

    assert f.I.shape == (1, 5, 17)
    assert f.g.shape == (1, 2, 5, 17)
    assert torch.equal(f.I[0, 2], expected_I)
    assert torch.equal(f.gx[0, 2], expected_gx)
    assert torch.count_nonzero(f.gy) == 0


def test_finite_ramp_batched_consistent_with_scalar() -> None:
    """Verify batched finite-ramp rendering matches scalar calls."""
    height = 48
    width = 52
    ramp_width = torch.tensor([12.0, 18.0, 24.0])
    angle = torch.tensor([0.0, math.radians(20.0), math.radians(45.0)])
    center_offset = torch.tensor([-3.0, 0.0, 5.0])
    amplitude = torch.tensor([0.8, 1.0, 1.2])

    out = finite_ramp(
        height,
        width,
        ramp_width=ramp_width,
        angle_rad=angle,
        center_offset=center_offset,
        amplitude=amplitude,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = finite_ramp(
            height,
            width,
            ramp_width=float(ramp_width[i]),
            angle_rad=float(angle[i]),
            center_offset=float(center_offset[i]),
            amplitude=float(amplitude[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_finite_ramp_honors_requested_device() -> None:
    """Verify scalar finite-ramp inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = finite_ramp(
        20,
        22,
        ramp_width=8.0,
        angle_rad=0.25,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_finite_ramp_infers_tensor_device() -> None:
    """Verify tensor inputs keep finite-ramp output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = finite_ramp(
        20,
        22,
        ramp_width=torch.tensor([8.0, 12.0], device=device),
        angle_rad=torch.tensor([0.0, 0.25], device=device),
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_roof_profile_horizontal_profile_and_gradient() -> None:
    """Check the triangular sampled profile and side-specific gradient signs."""
    f = roof_profile(5, 17, width_px=8.0, theta_rad=0.0, contrast=2.0)
    x = torch.arange(17, dtype=torch.float32) - 8.0
    expected_I = 2.0 * torch.clamp(1.0 - torch.abs(x) / 4.0, min=0.0, max=1.0)
    expected_gx = torch.zeros_like(x)
    expected_gx[(x > -4.0) & (x < 0.0)] = 0.5
    expected_gx[(x > 0.0) & (x < 4.0)] = -0.5

    assert f.I.shape == (1, 5, 17)
    assert f.g.shape == (1, 2, 5, 17)
    assert torch.equal(f.I[0, 2], expected_I)
    assert torch.equal(f.gx[0, 2], expected_gx)
    assert torch.count_nonzero(f.gy) == 0


def test_smoothed_ramp_gradient_matches_fd() -> None:
    """Check the smoothed ramp analytic gradient against finite differences."""
    f = smoothed_ramp(
        256,
        256,
        width_px=48.0,
        theta_rad=math.radians(20.0),
        sigma_e=4.0,
    )
    _check_signal_mask(f, rel_tol=1e-3, name="smoothed_ramp")


def test_mach_band_gradient_matches_fd() -> None:
    """Check the Mach-band analytic gradient against finite differences."""
    f = mach_band(
        256,
        256,
        ramp_width=48.0,
        angle_rad=math.radians(25.0),
        edge_sigma=4.0,
        shoulder_amplitude=0.08,
        shoulder_sigma=5.0,
    )
    _check_signal_mask(f, rel_tol=2e-3, name="mach_band")


def test_mach_band_has_opposite_shoulders() -> None:
    """Check that the shoulders darken the low side and brighten the high side."""
    f = mach_band(
        1,
        129,
        ramp_width=32.0,
        angle_rad=0.0,
        edge_sigma=2.0,
        shoulder_amplitude=0.1,
        shoulder_sigma=2.0,
    )
    base = smoothed_ramp(1, 129, width_px=32.0, theta_rad=0.0, sigma_e=2.0)
    x = torch.arange(129, dtype=torch.float32) - 64.0
    low_idx = int(torch.argmin(torch.abs(x + 16.0)))
    high_idx = int(torch.argmin(torch.abs(x - 16.0)))

    assert f.I[0, 0, low_idx] < base.I[0, 0, low_idx]
    assert f.I[0, 0, high_idx] > base.I[0, 0, high_idx]


def test_mach_band_default_call_renders_frame() -> None:
    """Verify Mach-band defaults render a usable analytic frame."""
    frame = mach_band(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_mach_band_infers_tensor_device() -> None:
    """Verify tensor inputs keep Mach-band output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = mach_band(
        20,
        24,
        ramp_width=torch.tensor([12.0, 16.0], device=device),
        angle_rad=torch.tensor([0.0, 0.25], device=device),
        edge_sigma=2.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_ramp_batched_consistent_with_scalar() -> None:
    """Verify batched smoothed-ramp rendering matches scalar calls."""
    H = W = 96
    width_px = torch.tensor([16.0, 24.0, 32.0])
    theta = torch.tensor([0.0, math.radians(20.0), math.radians(45.0)])
    x0 = torch.tensor([-3.0, 0.0, 5.0])
    contrast = torch.tensor([0.8, 1.0, 1.2])
    sigma_e = torch.tensor([2.0, 3.0, 4.0])
    out = smoothed_ramp(
        H,
        W,
        width_px=width_px,
        theta_rad=theta,
        x0=x0,
        contrast=contrast,
        sigma_e=sigma_e,
    )

    assert out.I.shape == (3, H, W)
    assert out.g.shape == (3, 2, H, W)
    for i in range(3):
        single = smoothed_ramp(
            H,
            W,
            width_px=float(width_px[i]),
            theta_rad=float(theta[i]),
            x0=float(x0[i]),
            contrast=float(contrast[i]),
            sigma_e=float(sigma_e[i]),
        )
        assert torch.allclose(out.I[i], single.I[0])
        assert torch.allclose(out.gx[i], single.gx[0])
        assert torch.allclose(out.gy[i], single.gy[0])


def test_transition_generators_preserve_requested_dtype() -> None:
    """Check dtype propagation for the direct formulas."""
    finite = finite_ramp(
        16,
        16,
        ramp_width=8.0,
        angle_rad=0.25,
        dtype=torch.float64,
    )
    f = smoothed_ramp(
        16,
        16,
        width_px=8.0,
        theta_rad=0.25,
        sigma_e=2.0,
        dtype=torch.float64,
    )
    assert finite.I.dtype == torch.float64
    assert finite.g.dtype == torch.float64
    assert f.I.dtype == torch.float64
    assert f.g.dtype == torch.float64
