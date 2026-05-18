"""Heaviside-like step rendered at fixed `sigma_e = 0.5` px.

This is `smoothed_step` at a sharp band-limited scale. It gives the benchmark a
near-discontinuous edge without leaving the sampled intensity and analytic
gradient relationship undefined.
"""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric
from agfb_generators.smoothed_step import smoothed_step


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
    """Render the sharpest band-limited straight edge used by AGFB.

    This is a wrapper around `smoothed_step` with `sigma_e = 0.5` px. AGFB
    uses it where the specification calls for a hard edge while still keeping
    the rendered intensity and analytic gradients numerically resolvable.
    """
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
