"""Mach-band ramp generator."""

from __future__ import annotations

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    gauss_Phi,
    gauss_phi,
    infer_batch_size,
    pack,
)


def mach_band(
    height: int,
    width: int,
    *,
    width_px: Numeric,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 1.0,
    band_strength: Numeric = 0.08,
    band_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a smoothed ramp with Gaussian Mach-band shoulders.

    A positive shoulder is placed at `z = width_px / 2` and a negative shoulder
    at `z = -width_px / 2`, added on top of the Gaussian-smoothed finite ramp.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(
        width_px,
        theta_rad,
        x0,
        contrast,
        sigma_e,
        band_strength,
        band_sigma,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    w = as_batch(width_px, B, device, dtype)
    theta = as_batch(theta_rad, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    s = as_batch(sigma_e, B, device, dtype)
    band_amp = as_batch(band_strength, B, device, dtype)
    band_s = as_batch(band_sigma, B, device, dtype)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    z = xx * cos_t + yy * sin_t - x0_b
    h = w / 2.0
    a = z + h
    b = z - h

    F_a = a * gauss_Phi(a / s) + s * gauss_phi(a / s)
    F_b = b * gauss_Phi(b / s) + s * gauss_phi(b / s)
    base_I = (c / w) * (F_a - F_b)
    base_gmag = (c / w) * (gauss_Phi(a / s) - gauss_Phi(b / s))

    band_s2 = band_s * band_s
    G_plus = torch.exp(-0.5 * ((z - h) / band_s) ** 2)
    G_minus = torch.exp(-0.5 * ((z + h) / band_s) ** 2)
    band_I = c * band_amp * (G_plus - G_minus)
    band_gmag = c * band_amp * (((z + h) / band_s2) * G_minus - ((z - h) / band_s2) * G_plus)

    I = base_I + band_I
    gmag = base_gmag + band_gmag
    gx = gmag * cos_t
    gy = gmag * sin_t
    return pack(I, gx, gy)
