"""Asymmetric one-dimensional Gaussian ridge generator."""

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


def asymmetric_ridge(
    height: int,
    width: int,
    *,
    negative_sigma: Numeric,
    positive_sigma: Numeric,
    angle_rad: Numeric,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian ridge with different widths on each side.

    The benchmark uses this generator when a ridge detector should handle a
    profile that is narrow on one side and broad on the other. This is useful
    for exposing centerline bias that a symmetric ridge does not reveal. The
    visual notebook and cross-section document use it as the asymmetric ridge
    example.

    `negative_sigma` controls the standard deviation on the negative side of
    the signed normal coordinate, and `positive_sigma` controls the
    nonnegative side. `angle_rad` is the ridge-normal angle in radians measured
    from the image `+x` direction. `center_offset` moves the ridge center along
    that normal coordinate, and `amplitude` is the centerline intensity.

    The rendered intensity is
    `amplitude * exp(-0.5 * u^2 / sigma(u)^2)`, where
    `u = x * cos(angle) + y * sin(angle) - center_offset`. The returned
    `Frame` contains the intensity image and the closed-form gradients with
    respect to image `x` and `y`. If `device` is omitted and a tensor parameter
    is passed, the render stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    validate_positive("negative_sigma", negative_sigma)
    validate_positive("positive_sigma", positive_sigma)
    device = infer_device(
        device,
        negative_sigma,
        positive_sigma,
        angle_rad,
        center_offset,
        amplitude,
    )
    batch_size = infer_batch_size(
        negative_sigma,
        positive_sigma,
        angle_rad,
        center_offset,
        amplitude,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    negative_sigma_batch = as_batch(negative_sigma, batch_size, device, dtype)
    positive_sigma_batch = as_batch(positive_sigma, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    normal_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    inv_negative_sigma_sq = torch.reciprocal(negative_sigma_batch * negative_sigma_batch)
    inv_positive_sigma_sq = torch.reciprocal(positive_sigma_batch * positive_sigma_batch)
    inv_sigma_sq = torch.where(normal_coord < 0.0, inv_negative_sigma_sq, inv_positive_sigma_sq)

    normal_scaled = normal_coord * inv_sigma_sq
    exponent = -0.5 * normal_coord * normal_scaled
    intensity = amplitude_batch * torch.exp(exponent)
    normal_gradient = -intensity * normal_scaled
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)
