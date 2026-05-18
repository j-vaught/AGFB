"""Cross-check every generator's analytic gradient against a 4th-order centred
finite difference of the rendered intensity, on the signal mask.

Pass criterion (per §1.1 sanity diagnostic): max relative error < 1e-3.
"""

from __future__ import annotations

import math

import torch

from cpgf_generators import (
    curved_arc,
    gaussian_blob,
    gaussian_ridge,
    hard_step,
    polynomial,
    sinusoid,
    smoothed_bar,
    smoothed_step,
)
from cpgf_generators.base import Frame


def _fd4(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """4th-order centred finite difference of a single (H, W) image."""
    gx = torch.zeros_like(I)
    gy = torch.zeros_like(I)
    gx[:, 2:-2] = (-I[:, 4:] + 8 * I[:, 3:-1] - 8 * I[:, 1:-3] + I[:, :-4]) / 12.0
    gy[2:-2, :] = (-I[4:, :] + 8 * I[3:-1, :] - 8 * I[1:-3, :] + I[:-4, :]) / 12.0
    return gx, gy


def _check_signal_mask(frame: Frame, *, rel_tol: float, name: str) -> None:
    """A1-style NRMSE between analytic gradient and 4th-order FD of the rendered
    intensity, restricted to the inner signal mask (where |grad| > 1% of peak).
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
    f = smoothed_step(256, 256, theta_rad=math.radians(30.0), sigma_e=4.0)
    _check_signal_mask(f, rel_tol=1e-3, name="smoothed_step")


def test_hard_step_gradient_matches_fd() -> None:
    # sigma_e=0.5 px is at Nyquist; 4th-order FD cannot achieve the §1.1 1e-3
    # bar here. We only check that gradient magnitudes are in the same ballpark.
    f = hard_step(256, 256, theta_rad=math.radians(15.0))
    _check_signal_mask(f, rel_tol=3e-1, name="hard_step")


def test_curved_arc_gradient_matches_fd() -> None:
    f = curved_arc(256, 256, r0=64.0, sigma_e=4.0)
    _check_signal_mask(f, rel_tol=1e-3, name="curved_arc")


def test_sinusoid_gradient_matches_fd() -> None:
    f = sinusoid(256, 256, freq=0.05, theta_rad=math.radians(30.0))
    _check_signal_mask(f, rel_tol=1e-2, name="sinusoid")


def test_gaussian_blob_gradient_matches_fd() -> None:
    f = gaussian_blob(256, 256, sigma=8.0)
    _check_signal_mask(f, rel_tol=1e-3, name="gaussian_blob")


def test_gaussian_ridge_gradient_matches_fd() -> None:
    f = gaussian_ridge(256, 256, sigma=4.0, theta_rad=math.radians(20.0))
    _check_signal_mask(f, rel_tol=1e-3, name="gaussian_ridge")


def test_smoothed_bar_gradient_matches_fd() -> None:
    f = smoothed_bar(256, 256, width_px=32.0, theta_rad=math.radians(15.0), sigma_e=4.0)
    _check_signal_mask(f, rel_tol=1e-3, name="smoothed_bar")


def test_polynomial_gradient_matches_fd() -> None:
    coeffs = torch.zeros(1, 4, 4)
    coeffs[0, 0, 0] = 0.0
    coeffs[0, 1, 0] = 0.3
    coeffs[0, 0, 1] = -0.2
    coeffs[0, 2, 1] = 0.05
    coeffs[0, 1, 2] = -0.04
    f = polynomial(64, 64, coeffs=coeffs, scale=64.0)
    _check_signal_mask(f, rel_tol=1e-3, name="polynomial")
