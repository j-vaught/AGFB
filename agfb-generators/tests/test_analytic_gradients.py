"""Cross-check every generator's analytic gradient against a 4th-order centred
finite difference of the rendered intensity, on the signal mask.

Pass criterion for most generators is max relative error < 1e-3.
"""

from __future__ import annotations

import math

import torch

from agfb_generators import (
    curved_arc,
    gaussian_blob,
    gaussian_ridge,
    polynomial,
    sinusoid,
    smoothed_bar,
    smoothed_step,
)
from agfb_generators.base import Frame


def _fd4(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate gradients for one image with a fourth-order finite difference.

    The analytic-gradient tests use this helper as the numerical reference for
    a rendered AGFB intensity field before comparing only the inner signal mask.
    """
    gx = torch.zeros_like(I)
    gy = torch.zeros_like(I)
    gx[:, 2:-2] = (-I[:, 4:] + 8 * I[:, 3:-1] - 8 * I[:, 1:-3] + I[:, :-4]) / 12.0
    gy[2:-2, :] = (-I[4:, :] + 8 * I[3:-1, :] - 8 * I[1:-3, :] + I[:-4, :]) / 12.0
    return gx, gy


def _check_signal_mask(frame: Frame, *, rel_tol: float, name: str) -> None:
    """Assert finite-difference agreement on the AGFB signal mask.

    Generator tests use this helper to compute an A1-style normalized root mean
    square error between each frame's analytic gradient and the finite
    difference gradient, restricted to interior pixels where the true gradient
    is large enough to be meaningful.
    """
    I = frame.I[0]
    fd_gx, fd_gy = _fd4(I)
    inner = torch.zeros_like(I, dtype=torch.bool)
    inner[3:-3, 3:-3] = True

    mag = torch.sqrt(frame.gx[0] ** 2 + frame.gy[0] ** 2)
    signal = (mag > 1e-2 * float(mag.max())) & inner
    n = int(signal.sum())
    assert n > 50, f"{name}: signal mask too small ({n})"

    diff_x = (fd_gx - frame.gx[0])[signal]
    diff_y = (fd_gy - frame.gy[0])[signal]
    num = torch.mean(diff_x * diff_x + diff_y * diff_y)
    den = torch.mean(frame.gx[0][signal] ** 2 + frame.gy[0][signal] ** 2)
    nrmse = float(torch.sqrt(num / den))
    assert nrmse < rel_tol, f"{name}: NRMSE={nrmse:.2e} >= {rel_tol:.2e}"


def test_smoothed_step_gradient_matches_fd() -> None:
    """Check the smoothed straight edge gradient used by the AGFB benchmark."""
    f = smoothed_step(256, 256, theta_rad=math.radians(30.0), sigma_e=4.0)
    _check_signal_mask(f, rel_tol=1e-3, name="smoothed_step")


def test_curved_arc_gradient_matches_fd() -> None:
    """Check the curved-edge generator against finite-difference gradients."""
    f = curved_arc(256, 256, radius=64.0, edge_sigma=4.0)
    _check_signal_mask(f, rel_tol=1e-3, name="curved_arc")


def test_curved_arc_batched_consistent_with_scalar() -> None:
    """Verify batched curved arcs match repeated scalar renders."""
    height = 80
    width = 84
    radius = torch.tensor([40.0, 56.0, 72.0])
    center_x = torch.tensor([-12.0, 0.0, 14.0])
    center_y = torch.tensor([8.0, -6.0, 2.0])
    amplitude = torch.tensor([0.8, 1.0, 1.2])
    edge_sigma = torch.tensor([3.0, 4.0, 5.0])

    out = curved_arc(
        height,
        width,
        radius=radius,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        edge_sigma=edge_sigma,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = curved_arc(
            height,
            width,
            radius=float(radius[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
            edge_sigma=float(edge_sigma[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_curved_arc_honors_requested_device() -> None:
    """Verify scalar curved-arc inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = curved_arc(
        20,
        24,
        radius=18.0,
        center_x=-4.0,
        center_y=3.0,
        amplitude=1.0,
        edge_sigma=3.0,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_curved_arc_infers_tensor_device() -> None:
    """Verify tensor inputs keep curved-arc output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = curved_arc(
        20,
        24,
        radius=torch.tensor([18.0, 22.0], device=device),
        center_x=torch.tensor([-4.0, 4.0], device=device),
        center_y=3.0,
        amplitude=1.0,
        edge_sigma=3.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_sinusoid_gradient_matches_fd() -> None:
    """Check the sinusoidal grating gradient used for AGFB frequency response."""
    f = sinusoid(256, 256, freq=0.05, theta_rad=math.radians(30.0))
    _check_signal_mask(f, rel_tol=1e-2, name="sinusoid")


def test_gaussian_blob_gradient_matches_fd() -> None:
    """Check the Gaussian blob gradient whose direction varies over the image."""
    f = gaussian_blob(256, 256, scale_sigma=8.0)
    _check_signal_mask(f, rel_tol=1e-3, name="gaussian_blob")


def test_gaussian_blob_batched_consistent_with_scalar() -> None:
    """Verify batched Gaussian blobs match repeated scalar renders."""
    height = 72
    width = 76
    scale_sigma = torch.tensor([6.0, 9.0, 12.0])
    center_x = torch.tensor([-5.0, 0.0, 4.0])
    center_y = torch.tensor([3.0, -4.0, 2.0])
    amplitude = torch.tensor([0.75, 1.0, 1.25])

    out = gaussian_blob(
        height,
        width,
        scale_sigma=scale_sigma,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = gaussian_blob(
            height,
            width,
            scale_sigma=float(scale_sigma[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_gaussian_blob_honors_requested_device() -> None:
    """Verify scalar Gaussian blob inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = gaussian_blob(
        20,
        24,
        scale_sigma=6.0,
        center_x=-3.0,
        center_y=2.0,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_gaussian_blob_infers_tensor_device() -> None:
    """Verify tensor inputs keep Gaussian blob output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = gaussian_blob(
        20,
        24,
        scale_sigma=torch.tensor([6.0, 9.0], device=device),
        center_x=torch.tensor([-3.0, 3.0], device=device),
        center_y=2.0,
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_gaussian_ridge_gradient_matches_fd() -> None:
    """Check the oriented Gaussian ridge gradient used by AGFB ridge cases."""
    f = gaussian_ridge(256, 256, width_sigma=4.0, angle_rad=math.radians(20.0))
    _check_signal_mask(f, rel_tol=1e-3, name="gaussian_ridge")


def test_gaussian_ridge_batched_consistent_with_scalar() -> None:
    """Verify batched Gaussian ridges match repeated scalar renders."""
    height = 72
    width = 76
    width_sigma = torch.tensor([4.0, 6.0, 8.0])
    angle = torch.tensor([0.0, math.radians(25.0), math.radians(50.0)])
    center_offset = torch.tensor([-3.0, 0.0, 3.0])
    amplitude = torch.tensor([0.75, 1.0, 1.25])

    out = gaussian_ridge(
        height,
        width,
        width_sigma=width_sigma,
        angle_rad=angle,
        center_offset=center_offset,
        amplitude=amplitude,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = gaussian_ridge(
            height,
            width,
            width_sigma=float(width_sigma[i]),
            angle_rad=float(angle[i]),
            center_offset=float(center_offset[i]),
            amplitude=float(amplitude[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_gaussian_ridge_honors_requested_device() -> None:
    """Verify scalar Gaussian ridge inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = gaussian_ridge(
        20,
        24,
        width_sigma=4.0,
        angle_rad=math.radians(20.0),
        center_offset=1.0,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_gaussian_ridge_infers_tensor_device() -> None:
    """Verify tensor inputs keep Gaussian ridge output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = gaussian_ridge(
        20,
        24,
        width_sigma=torch.tensor([4.0, 6.0], device=device),
        angle_rad=torch.tensor([0.0, math.radians(20.0)], device=device),
        center_offset=1.0,
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_bar_gradient_matches_fd() -> None:
    """Check the paired-edge soft bar assembled from smoothed steps."""
    f = smoothed_bar(256, 256, width_px=32.0, theta_rad=math.radians(15.0), sigma_e=4.0)
    _check_signal_mask(f, rel_tol=1e-3, name="smoothed_bar")


def test_polynomial_gradient_matches_fd() -> None:
    """Check the polynomial field generator used for exact low-order structure."""
    coefficients = torch.zeros(1, 4, 4)
    coefficients[0, 0, 0] = 0.0
    coefficients[0, 1, 0] = 0.3
    coefficients[0, 0, 1] = -0.2
    coefficients[0, 2, 1] = 0.05
    coefficients[0, 1, 2] = -0.04
    f = polynomial(64, 64, coefficients=coefficients, coordinate_scale=64.0)
    _check_signal_mask(f, rel_tol=1e-3, name="polynomial")


def test_polynomial_default_call_renders_frame() -> None:
    """Verify the default polynomial surface is available for quick previews."""
    frame = polynomial(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_polynomial_accepts_2d_coefficients() -> None:
    """Verify a single coefficient matrix renders one frame."""
    coefficients = torch.zeros(3, 3)
    coefficients[1, 0] = 0.5
    coefficients[0, 1] = -0.25
    frame = polynomial(20, 24, coefficients=coefficients, coordinate_scale=8.0)

    assert frame.I.shape == (1, 20, 24)
    assert frame.g.shape == (1, 2, 20, 24)


def test_polynomial_infers_tensor_device() -> None:
    """Verify coefficient tensors keep polynomial output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coefficients = torch.zeros(2, 3, 3, device=device)
    coefficients[:, 1, 0] = torch.tensor([0.3, 0.6], device=device)

    frame = polynomial(20, 24, coefficients=coefficients, coordinate_scale=16.0)

    assert frame.I.device == device
    assert frame.g.device == device
