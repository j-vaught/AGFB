"""Triangular roof profile generator."""

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


def roof_profile(
    height: int,
    width: int,
    *,
    roof_width: Numeric = 64.0,
    angle_rad: Numeric = 0.0,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched triangular roof intensity profile.

    The benchmark uses this generator when a filter should respond to a sharp
    ridge-like peak formed by two straight ramp faces. It is useful for testing
    whether a filter treats a peaked roof profile differently from a one-way
    ramp or a blurred Gaussian ridge. The visual notebook, cross-section
    document, and roof-profile preview use this function as the triangular
    transition example.

    The default call renders a horizontal 64 px roof centered in the image with
    unit peak intensity. `roof_width` is the full support width in pixels.
    `angle_rad` is the roof normal direction in radians, measured from the
    image `+x` direction. `center_offset` shifts the peak line in the shared
    centered coordinate system, and `amplitude` is the peak intensity.

    The projected coordinate is
    `z = x * cos(angle) + y * sin(angle) - center_offset`. The rendered
    intensity is `amplitude * clamp(1 - abs(z) / (roof_width / 2), 0, 1)`.
    The returned `Frame` contains that intensity image and the closed-form
    piecewise-constant gradients on the two roof faces. The derivative is set
    to zero outside the open support band and exactly at the roof peak and
    support edges. If `device` is omitted and a tensor parameter is passed, the
    render stays on that tensor's device.
    """
    device = infer_device(device, roof_width, angle_rad, center_offset, amplitude)
    batch_size = infer_batch_size(roof_width, angle_rad, center_offset, amplitude)
    xx, yy = coord_grid(height, width, device, dtype)

    roof_width_batch = as_batch(roof_width, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    roof_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    half_width = roof_width_batch / 2.0

    intensity = amplitude_batch * torch.clamp(
        1.0 - torch.abs(roof_coord) / half_width,
        min=0.0,
        max=1.0,
    )
    left_slope_mask = ((roof_coord > -half_width) & (roof_coord < 0.0)).to(dtype=dtype)
    right_slope_mask = ((roof_coord > 0.0) & (roof_coord < half_width)).to(dtype=dtype)
    normal_gradient = (amplitude_batch / half_width) * (left_slope_mask - right_slope_mask)
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return pack(intensity, gradient_x, gradient_y)
