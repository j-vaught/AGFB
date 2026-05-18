"""Linear chirp grating generator."""

from __future__ import annotations

import math

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def chirp(
    height: int,
    width: int,
    *,
    freq0: Numeric,
    chirp_rate: Numeric,
    theta_rad: Numeric,
    contrast: Numeric = 1.0,
    phase: Numeric = 0.0,
    u0: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched oriented linear chirp.

    Frequency is measured in cycles per pixel. The instantaneous frequency is
    `freq0 + chirp_rate * u` along the oriented coordinate `u`.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(freq0, chirp_rate, theta_rad, contrast, phase, u0)
    xx, yy = coord_grid(height, width, device, dtype)

    f0 = as_batch(freq0, B, device, dtype)
    cr = as_batch(chirp_rate, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    ph = as_batch(phase, B, device, dtype)
    u0_b = as_batch(u0, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    u = xx * cos_t + yy * sin_t - u0_b
    arg = 2.0 * math.pi * (f0 * u + 0.5 * cr * u * u) + ph

    I = c * torch.sin(arg)
    gmag = c * torch.cos(arg) * 2.0 * math.pi * (f0 + cr * u)
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
