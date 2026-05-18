"""1D Gaussian ridge (§1.1 `gaussian_ridge`)."""

from __future__ import annotations

import torch

from cpgf_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def gaussian_ridge(
    height: int,
    width: int,
    *,
    sigma: Numeric,
    theta_rad: Numeric,
    u0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """`I = c * exp(-(u - u0)^2 / (2 sigma^2))` with `u = p . n_hat`."""
    device = device or torch.device("cpu")
    B = infer_batch_size(sigma, theta_rad, u0, contrast)
    xx, yy = coord_grid(height, width, device, dtype)

    s = as_batch(sigma, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    u0_b = as_batch(u0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    u = xx * cos_t + yy * sin_t - u0_b
    s2 = s * s
    I = c * torch.exp(-(u * u) / (2.0 * s2))
    gmag = -I * u / s2
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
