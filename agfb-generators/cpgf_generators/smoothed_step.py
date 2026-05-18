"""Smoothed straight edge (§1.1 `smoothed_step`)."""

from __future__ import annotations

import torch

from cpgf_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    gauss_Phi,
    gauss_phi,
    infer_batch_size,
    pack,
)


def smoothed_step(
    height: int,
    width: int,
    *,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian-smoothed straight edge.

    CPGF uses this as the canonical straight-edge generator, and `hard_step`,
    `smoothed_bar`, and regression tests build on it. It evaluates
    `I = c * Phi((p . n_hat - x0) / sigma_e)` and returns the intensity plus
    the analytic gradient aligned with `n_hat = (cos t, sin t)`.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(theta_rad, x0, contrast, sigma_e)
    xx, yy = coord_grid(height, width, device, dtype)

    theta = as_batch(theta_rad, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    s = as_batch(sigma_e, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    u = (xx * cos_t + yy * sin_t - x0_b) / s

    I = c * gauss_Phi(u)
    gmag = (c / s) * gauss_phi(u)
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
