"""Asymmetric one-dimensional Gaussian ridge generator."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def asymmetric_ridge(
    height: int,
    width: int,
    *,
    sigma_neg: Numeric,
    sigma_pos: Numeric,
    theta_rad: Numeric,
    u0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian ridge with different side widths.

    The signed normal coordinate is `u = x cos(theta) + y sin(theta) - u0`.
    Pixels with negative `u` use `sigma_neg`; pixels with nonnegative `u` use
    `sigma_pos`. The returned gradient is the closed-form spatial derivative
    projected onto the ridge normal.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(sigma_neg, sigma_pos, theta_rad, u0, contrast)
    xx, yy = coord_grid(height, width, device, dtype)

    s_neg = as_batch(sigma_neg, B, device, dtype)
    s_pos = as_batch(sigma_pos, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    u0_b = as_batch(u0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    u = xx * cos_t + yy * sin_t - u0_b
    sigma = torch.where(u < 0.0, s_neg, s_pos)
    s2 = sigma * sigma

    I = c * torch.exp(-(u * u) / (2.0 * s2))
    gmag = -I * u / s2
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
