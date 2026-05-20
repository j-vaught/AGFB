"""Gaussian-smoothed finite-width ramp generator."""

from __future__ import annotations

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    gauss_Phi,
    gauss_phi,
    infer_batch_size,
    infer_device,
    pack,
    validate_positive,
)


def smoothed_ramp(
    height: int,
    width: int,
    *,
    width_px: Numeric,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian-smoothed finite-width ramp.

    This is the closed-form convolution of a finite linear ramp with a
    Gaussian edge-spread kernel. The gradient magnitude is the corresponding
    difference of two Gaussian cumulative distribution functions.
    """
    validate_positive("width_px", width_px)
    validate_positive("sigma_e", sigma_e)
    device = infer_device(device, width_px, theta_rad, x0, contrast, sigma_e)
    B = infer_batch_size(width_px, theta_rad, x0, contrast, sigma_e)
    xx, yy = coord_grid(height, width, device, dtype)

    w = as_batch(width_px, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    s = as_batch(sigma_e, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    z = xx * cos_t + yy * sin_t - x0_b
    h = w / 2.0
    a = z + h
    b = z - h

    F_a = a * gauss_Phi(a / s) + s * gauss_phi(a / s)
    F_b = b * gauss_Phi(b / s) + s * gauss_phi(b / s)
    I = (c / w) * (F_a - F_b)
    gmag = (c / w) * (gauss_Phi(a / s) - gauss_Phi(b / s))
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
