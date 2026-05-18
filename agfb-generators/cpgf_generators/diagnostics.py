"""§1.1 sanity-only patterns: contrast ramp, multi-frequency grating, constant."""

from __future__ import annotations

import math

import torch

from cpgf_generators.base import Frame, coord_grid, pack
from cpgf_generators.sinusoid import sinusoid


def contrast_ramp(
    height: int,
    width: int,
    *,
    theta_rad: float = 0.0,
    contrast: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Linear intensity ramp from -c/2 to +c/2 along `n_hat = (cos t, sin t)`.

    Used for the contrast-linearity diagnostic: a linear ramp has constant
    gradient `(c/W) * n_hat`, so any well-behaved filter must report a flat
    response proportional to `c`.
    """
    device = device or torch.device("cpu")
    xx, yy = coord_grid(height, width, device, dtype)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    extent = max(height, width)
    g_val = contrast / extent
    I = g_val * (xx * cos_t + yy * sin_t)
    I = I.unsqueeze(0)
    gx = torch.full_like(I, g_val * cos_t)
    gy = torch.full_like(I, g_val * sin_t)
    return pack(I, gx, gy)


def multi_freq_grating(
    height: int,
    width: int,
    *,
    freqs: list[float],
    theta_rad: float = 0.0,
    contrast: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Vertical stack of bands, each band a `sinusoid` at one frequency.

    Used for phase-linearity: peak-of-|grad| position must not shift across the
    band for any linear-phase filter.
    """
    device = device or torch.device("cpu")
    if not freqs:
        raise ValueError("freqs must contain at least one frequency")
    n = len(freqs)
    band_h = height // n
    I = torch.zeros(1, height, width, device=device, dtype=dtype)
    gx = torch.zeros_like(I)
    gy = torch.zeros_like(I)
    for k, f in enumerate(freqs):
        rs = k * band_h
        re = height if k == n - 1 else (k + 1) * band_h
        frame = sinusoid(
            re - rs,
            width,
            freq=f,
            theta_rad=theta_rad,
            contrast=contrast,
            device=device,
            dtype=dtype,
        )
        I[:, rs:re, :] = frame.I
        gx[:, rs:re, :] = frame.gx
        gy[:, rs:re, :] = frame.gy
    return pack(I, gx, gy)


def constant_field(
    height: int,
    width: int,
    *,
    value: float = 0.5,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Uniform-intensity field; gradient is identically zero everywhere.

    A perfect filter must return zero on every metric here.
    """
    device = device or torch.device("cpu")
    I = torch.full((1, height, width), float(value), device=device, dtype=dtype)
    g = torch.zeros(1, 2, height, width, device=device, dtype=dtype)
    return Frame(I=I, g=g)
