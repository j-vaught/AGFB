"""Parabolic curved Gaussian ridge generator."""

from __future__ import annotations

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    infer_batch_size,
    infer_device,
    normalize_contrast,
    validate_amplitude,
    validate_positive,
)


def curved_ridge(
    height: int,
    width: int,
    *,
    width_sigma: Numeric,
    angle_rad: Numeric,
    curvature: Numeric,
    normal_offset: Numeric = 0.0,
    tangent_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian ridge bent by a parabolic centerline.

    The benchmark uses this generator when a ridge centerline should curve
    smoothly instead of remaining straight. It is useful for checking whether
    a filter follows elongated structures through slowly changing orientation
    while the analytic gradient remains known at every pixel.

    `width_sigma` controls the Gaussian falloff perpendicular to the local
    ridge centerline. `angle_rad` sets the unbent ridge normal direction in
    radians, measured from the image `+x` direction. `curvature` bends the
    centerline as a parabola along the tangent coordinate. `normal_offset`
    shifts the ridge across its normal axis, `tangent_offset` shifts the vertex
    along its tangent axis, and `amplitude` controls the peak intensity.

    Coordinates are rotated into a normal coordinate
    `q = x * cos(angle) + y * sin(angle) - normal_offset` and a tangent
    coordinate `v = -x * sin(angle) + y * cos(angle) - tangent_offset`. The
    ridge coordinate is `u = q - 0.5 * curvature * v^2`. The returned `Frame`
    contains the intensity image and the closed-form gradients from the chain
    rule. If `device` is omitted and a tensor parameter is passed, the render
    stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    validate_positive("width_sigma", width_sigma)
    device = infer_device(
        device,
        width_sigma,
        angle_rad,
        curvature,
        normal_offset,
        tangent_offset,
        amplitude,
    )
    batch_size = infer_batch_size(
        width_sigma,
        angle_rad,
        curvature,
        normal_offset,
        tangent_offset,
        amplitude,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    width_sigma_batch = as_batch(width_sigma, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    curvature_batch = as_batch(curvature, batch_size, device, dtype)
    normal_offset_batch = as_batch(normal_offset, batch_size, device, dtype)
    tangent_offset_batch = as_batch(tangent_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    normal_coord = xx * cos_angle + yy * sin_angle - normal_offset_batch
    tangent_coord = -xx * sin_angle + yy * cos_angle - tangent_offset_batch
    ridge_coord = normal_coord - 0.5 * curvature_batch * tangent_coord * tangent_coord

    ridge_coord_dx = cos_angle + curvature_batch * tangent_coord * sin_angle
    ridge_coord_dy = sin_angle - curvature_batch * tangent_coord * cos_angle
    width_sigma_sq = width_sigma_batch * width_sigma_batch

    intensity = amplitude_batch * torch.exp(-(ridge_coord * ridge_coord) / (2.0 * width_sigma_sq))
    normal_gradient = -intensity * ridge_coord / width_sigma_sq
    gradient_x = normal_gradient * ridge_coord_dx
    gradient_y = normal_gradient * ridge_coord_dy
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)
