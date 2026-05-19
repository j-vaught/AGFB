"""Heaviside-like step rendered at fixed `sigma_e = 0.5` px.

This is `smoothed_step` at a sharp band-limited scale. It gives the benchmark a
near-discontinuous edge without leaving the sampled intensity and analytic
gradient relationship undefined.
"""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, infer_device
from agfb_generators.smoothed_step import smoothed_step


def hard_step(
    height: int,
    width: int,
    *,
    angle_rad: Numeric,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render the sharpest band-limited straight edge used by the benchmark.

    The benchmark uses this generator when a filter should handle a nearly
    discontinuous straight edge while still keeping the sampled intensity and
    analytic gradient numerically resolvable. It is implemented as
    `smoothed_step` with a fixed edge width of 0.5 px.

    `angle_rad` is the edge normal direction in radians, measured from the
    image `+x` direction. `center_offset` shifts the edge in the shared
    centered coordinate system, and `amplitude` is the high-side intensity.
    If `device` is omitted and a tensor parameter is passed, the render stays
    on that tensor's device.
    """
    device = infer_device(device, angle_rad, center_offset, amplitude)
    return smoothed_step(
        height,
        width,
        theta_rad=angle_rad,
        x0=center_offset,
        contrast=amplitude,
        sigma_e=0.5,
        device=device,
        dtype=dtype,
    )
