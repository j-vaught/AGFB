"""Single-frequency grating (§1.1 `sinusoid`)."""

from __future__ import annotations

import math

import torch

from cpgf_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def sinusoid(
    height: int,
    width: int,
    *,
    freq: Numeric,
    theta_rad: Numeric,
    contrast: Numeric = 1.0,
    phase: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched single-frequency sinusoidal grating.

    CPGF uses this to probe frequency and orientation response, and the
    diagnostic grating stacks call it per band. The frequency is in
    cycles/pixel, and the returned `Frame` includes the analytic derivative
    of `I = c * sin(2 pi f * p . n_hat + phase)`.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(freq, theta_rad, contrast, phase)
    xx, yy = coord_grid(height, width, device, dtype)

    f = as_batch(freq, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    ph = as_batch(phase, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    arg = 2.0 * math.pi * f * (xx * cos_t + yy * sin_t) + ph

    I = c * torch.sin(arg)
    gmag = 2.0 * math.pi * f * c * torch.cos(arg)
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
