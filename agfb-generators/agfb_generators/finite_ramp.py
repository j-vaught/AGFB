"""Finite-width linear ramp generator."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def finite_ramp(
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
    """Render a batched finite-width linear ramp.

    The ramp is evaluated along `z = p . n_hat - x0`, rises from zero to
    `contrast` over `width_px`, and returns the piecewise-constant analytic
    gradient inside the transition band.
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

    I = c * torch.clamp((z + h) / w, min=0.0, max=1.0)
    inside = ((z > -h) & (z < h)).to(dtype=dtype)
    gmag = (c / w) * inside
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
