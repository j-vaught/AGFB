"""Localized rotated Gabor packet generator."""

from __future__ import annotations

import math

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def gabor_packet(
    height: int,
    width: int,
    *,
    freq: Numeric,
    theta_rad: Numeric,
    sigma_u: Numeric,
    sigma_v: Numeric,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    phase: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched anisotropic Gaussian-windowed sinusoid."""
    device = device or torch.device("cpu")
    B = infer_batch_size(freq, theta_rad, sigma_u, sigma_v, x0, y0, contrast, phase)
    xx, yy = coord_grid(height, width, device, dtype)

    f = as_batch(freq, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    su = as_batch(sigma_u, B, device, dtype)
    sv = as_batch(sigma_v, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    y0_b = as_batch(y0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    ph = as_batch(phase, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    dx = xx - x0_b
    dy = yy - y0_b
    u = dx * cos_t + dy * sin_t
    v = -dx * sin_t + dy * cos_t
    su2 = su * su
    sv2 = sv * sv

    E = torch.exp(-0.5 * ((u * u) / su2 + (v * v) / sv2))
    arg = 2.0 * math.pi * f * u + ph
    sin_arg = torch.sin(arg)
    cos_arg = torch.cos(arg)

    I = c * E * sin_arg
    dI_du = c * E * ((2.0 * math.pi * f) * cos_arg - (u / su2) * sin_arg)
    dI_dv = c * E * (-(v / sv2) * sin_arg)
    gx = dI_du * cos_t - dI_dv * sin_t
    gy = dI_du * sin_t + dI_dv * cos_t
    return pack(I, gx, gy)
