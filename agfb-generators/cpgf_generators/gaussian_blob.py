"""Isotropic 2D Gaussian peak (§1.1 `gaussian_blob`)."""

from __future__ import annotations

import torch

from cpgf_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def gaussian_blob(
    height: int,
    width: int,
    *,
    sigma: Numeric,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """`I = c * exp(-((x-x0)^2 + (y-y0)^2) / (2 sigma^2))`."""
    device = device or torch.device("cpu")
    B = infer_batch_size(sigma, x0, y0, contrast)
    xx, yy = coord_grid(height, width, device, dtype)

    s = as_batch(sigma, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    y0_b = as_batch(y0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)

    dx = xx - x0_b
    dy = yy - y0_b
    s2 = s * s
    I = c * torch.exp(-(dx * dx + dy * dy) / (2.0 * s2))
    gx = -I * dx / s2
    gy = -I * dy / s2
    return pack(I, gx, gy)
