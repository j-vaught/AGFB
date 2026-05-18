"""Parabolic curved Gaussian ridge generator."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def curved_ridge(
    height: int,
    width: int,
    *,
    sigma: Numeric,
    theta_rad: Numeric,
    curvature: Numeric,
    u0: Numeric = 0.0,
    v0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian ridge bent by a parabolic centerline.

    Coordinates are rotated into normal and tangent-like axes,
    `q = x cos(theta) + y sin(theta) - u0` and
    `v = -x sin(theta) + y cos(theta) - v0`. The ridge coordinate is
    `u = q - 0.5 * curvature * v^2`, with gradients from the chain rule.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(sigma, theta_rad, curvature, u0, v0, contrast)
    xx, yy = coord_grid(height, width, device, dtype)

    s = as_batch(sigma, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    k = as_batch(curvature, B, device, dtype)
    u0_b = as_batch(u0, B, device, dtype)
    v0_b = as_batch(v0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    q = xx * cos_t + yy * sin_t - u0_b
    v = -xx * sin_t + yy * cos_t - v0_b
    u = q - 0.5 * k * v * v

    du_dx = cos_t + k * v * sin_t
    du_dy = sin_t - k * v * cos_t
    s2 = s * s

    I = c * torch.exp(-(u * u) / (2.0 * s2))
    gmag = -I * u / s2
    gx = gmag * du_dx
    gy = gmag * du_dy
    return pack(I, gx, gy)
