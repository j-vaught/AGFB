"""Triangular roof profile generator."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def roof_profile(
    height: int,
    width: int,
    *,
    width_px: Numeric,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched triangular roof profile.

    The profile peaks at `z = 0`, falls linearly to zero at `|z| = width_px / 2`,
    and returns the piecewise-constant analytic gradient on each side.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(width_px, theta_rad, x0, contrast)
    xx, yy = coord_grid(height, width, device, dtype)

    w = as_batch(width_px, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    z = xx * cos_t + yy * sin_t - x0_b
    h = w / 2.0

    I = c * torch.clamp(1.0 - torch.abs(z) / h, min=0.0, max=1.0)
    rising = ((z > -h) & (z < 0.0)).to(dtype=dtype)
    falling = ((z > 0.0) & (z < h)).to(dtype=dtype)
    gmag = (c / h) * (rising - falling)
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
