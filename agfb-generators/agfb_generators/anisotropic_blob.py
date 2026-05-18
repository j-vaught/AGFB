"""Rotated anisotropic Gaussian blob generator."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def anisotropic_blob(
    height: int,
    width: int,
    *,
    sigma_u: Numeric,
    sigma_v: Numeric,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched rotated anisotropic Gaussian peak.

    The local coordinates are
    `u = dx * cos(theta) + dy * sin(theta)` and
    `v = -dx * sin(theta) + dy * cos(theta)`. The returned gradient is the
    closed-form spatial derivative of the rendered intensity.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(sigma_u, sigma_v, theta_rad, x0, y0, contrast)
    xx, yy = coord_grid(height, width, device, dtype)

    su = as_batch(sigma_u, B, device, dtype)
    sv = as_batch(sigma_v, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    y0_b = as_batch(y0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    dx = xx - x0_b
    dy = yy - y0_b
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t
    su2 = su * su
    sv2 = sv * sv

    I = c * torch.exp(-0.5 * ((u * u) / su2 + (v * v) / sv2))
    gx = I * (-(u / su2) * cos_t + (v / sv2) * sin_t)
    gy = I * (-(u / su2) * sin_t - (v / sv2) * cos_t)
    return pack(I, gx, gy)
