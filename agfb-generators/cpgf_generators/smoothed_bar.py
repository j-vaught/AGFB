"""Two parallel opposite-sign edges (§1.1 `smoothed_bar`)."""

from __future__ import annotations

import torch

from cpgf_generators.base import Frame, Numeric, pack
from cpgf_generators.smoothed_step import smoothed_step


def smoothed_bar(
    height: int,
    width: int,
    *,
    width_px: Numeric,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched soft bar as two opposite smoothed edges.

    CPGF uses this to test paired-edge behavior over a finite-width band. The
    function calls `smoothed_step` at `x0 - width_px / 2` and
    `x0 + width_px / 2` with opposite contrasts, then sums the two returned
    frames into one intensity and analytic-gradient field.
    """
    half = width_px / 2.0 if isinstance(width_px, torch.Tensor) else float(width_px) / 2.0
    if isinstance(x0, torch.Tensor):
        x_pos = x0 + half
        x_neg = x0 - half
    elif isinstance(half, torch.Tensor):
        x_pos = half + float(x0)
        x_neg = -half + float(x0)
    else:
        x_pos = float(x0) + half
        x_neg = float(x0) - half
    neg_c = -contrast if isinstance(contrast, torch.Tensor) else -float(contrast)

    a = smoothed_step(
        height,
        width,
        theta_rad=theta_rad,
        x0=x_neg,
        contrast=contrast,
        sigma_e=sigma_e,
        device=device,
        dtype=dtype,
    )
    b = smoothed_step(
        height,
        width,
        theta_rad=theta_rad,
        x0=x_pos,
        contrast=neg_c,
        sigma_e=sigma_e,
        device=device,
        dtype=dtype,
    )
    I = a.I + b.I
    gx = a.gx + b.gx
    gy = a.gy + b.gy
    return pack(I, gx, gy)
