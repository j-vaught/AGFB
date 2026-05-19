"""Finite-width linear ramp generator."""

from __future__ import annotations

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    infer_batch_size,
    infer_device,
    pack,
)


def finite_ramp(
    height: int,
    width: int,
    *,
    ramp_width: Numeric,
    angle_rad: Numeric,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched finite-width linear intensity ramp.

    The benchmark uses this generator when a filter should respond to a
    bounded constant-slope transition rather than a step edge. It is the sharp
    analytic ramp case: intensity is flat on both sides, linear through the
    transition band, and the gradient is constant inside that band.

    `ramp_width` is the transition width in pixels. `angle_rad` is the ramp
    normal direction in radians, measured from the image `+x` direction.
    `center_offset` shifts the middle of the ramp in the shared centered
    coordinate system, and `amplitude` is the high-side intensity.

    The ramp coordinate is
    `z = x * cos(angle) + y * sin(angle) - center_offset`. The rendered
    intensity is `amplitude * clamp((z + ramp_width / 2) / ramp_width, 0, 1)`.
    The returned `Frame` contains that intensity image and a closed-form
    piecewise-constant gradient. The derivative is set to zero outside the
    open transition band, including exactly at the two kink locations. If
    `device` is omitted and a tensor parameter is passed, the render stays on
    that tensor's device.
    """
    device = infer_device(device, ramp_width, angle_rad, center_offset, amplitude)
    batch_size = infer_batch_size(ramp_width, angle_rad, center_offset, amplitude)
    xx, yy = coord_grid(height, width, device, dtype)

    ramp_width_batch = as_batch(ramp_width, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    ramp_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    half_width = ramp_width_batch / 2.0

    intensity = amplitude_batch * torch.clamp(
        (ramp_coord + half_width) / ramp_width_batch,
        min=0.0,
        max=1.0,
    )
    transition_mask = ((ramp_coord > -half_width) & (ramp_coord < half_width)).to(dtype=dtype)
    normal_gradient = (amplitude_batch / ramp_width_batch) * transition_mask
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return pack(intensity, gradient_x, gradient_y)
