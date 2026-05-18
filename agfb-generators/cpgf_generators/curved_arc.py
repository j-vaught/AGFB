"""Radially smoothed disc (§1.1 `curved_arc`)."""

from __future__ import annotations

import torch

from cpgf_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    gauss_Phi,
    gauss_phi,
    infer_batch_size,
    pack,
)


def curved_arc(
    height: int,
    width: int,
    *,
    r0: Numeric,
    xc: Numeric = 0.0,
    yc: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched radially smoothed disc boundary.

    CPGF uses this to test curved edge recovery. It evaluates
    `I = c * Phi((r0 - rho) / sigma_e)` with `rho = ||p - center||` and
    returns the inward analytic gradient of that transition.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(r0, xc, yc, contrast, sigma_e)
    xx, yy = coord_grid(height, width, device, dtype)

    r0_b = as_batch(r0, B, device, dtype)
    xc_b = as_batch(xc, B, device, dtype)
    yc_b = as_batch(yc, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    s = as_batch(sigma_e, B, device, dtype)

    dx = xx - xc_b
    dy = yy - yc_b
    rho = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-12)
    u = (r0_b - rho) / s

    I = c * gauss_Phi(u)
    gmag = -(c / s) * gauss_phi(u)  # gradient is -phi * rho_hat (inward)
    gx = gmag * (dx / rho)
    gy = gmag * (dy / rho)
    return pack(I, gx, gy)
