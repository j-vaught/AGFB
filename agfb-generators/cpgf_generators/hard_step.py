"""Heaviside step rendered at fixed `sigma_e = 0.5` px (§1.1 `hard_step`).

In production, this is `smoothed_step` at the sharpest band-limited scale the
4096² grid can resolve. The continuous-form unsmoothed step appears only in the
§1.1.0 band-limit pre-experiment, never in §1.3 production frames.
"""

from __future__ import annotations

import torch

from cpgf_generators.base import Frame, Numeric
from cpgf_generators.smoothed_step import smoothed_step


def hard_step(
    height: int,
    width: int,
    *,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    return smoothed_step(
        height,
        width,
        theta_rad=theta_rad,
        x0=x0,
        contrast=contrast,
        sigma_e=0.5,
        device=device,
        dtype=dtype,
    )
