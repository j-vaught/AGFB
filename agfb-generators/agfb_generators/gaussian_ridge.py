"""One-dimensional Gaussian ridge generator."""

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


def gaussian_ridge(
    height: int,
    width: int,
    *,
    width_sigma: Numeric,
    angle_rad: Numeric,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched one-dimensional Gaussian ridge.

    The benchmark uses this generator when a smooth straight ridge has a known
    orientation, width, and analytic gradient. It is useful for checking
    filters on line-like structures without curvature, junctions, or unequal
    ridge sides.

    `width_sigma` controls the Gaussian falloff perpendicular to the ridge.
    `angle_rad` is the ridge normal direction in radians, measured from the
    image `+x` direction. `center_offset` shifts the ridge across that normal
    axis, and `amplitude` is the ridge peak intensity.

    The projected coordinate is
    `u = x * cos(angle) + y * sin(angle) - center_offset`. The rendered
    intensity is `amplitude * exp(-u^2 / (2 * width_sigma^2))`. The returned
    `Frame` contains the intensity image and the closed-form gradients with
    respect to image `x` and `y`. If `device` is omitted and a tensor parameter
    is passed, the render stays on that tensor's device.
    """
    device = infer_device(device, width_sigma, angle_rad, center_offset, amplitude)
    batch_size = infer_batch_size(width_sigma, angle_rad, center_offset, amplitude)
    xx, yy = coord_grid(height, width, device, dtype)

    width_sigma_batch = as_batch(width_sigma, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    ridge_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    width_sigma_sq = width_sigma_batch * width_sigma_batch
    intensity = amplitude_batch * torch.exp(-(ridge_coord * ridge_coord) / (2.0 * width_sigma_sq))
    normal_gradient = -intensity * ridge_coord / width_sigma_sq
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return pack(intensity, gradient_x, gradient_y)
