"""Rotated anisotropic Gaussian blob generator."""

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


def anisotropic_blob(
    height: int,
    width: int,
    *,
    length_sigma: Numeric,
    width_sigma: Numeric,
    angle_rad: Numeric,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian peak with independent rotated axis scales.

    The benchmark uses this generator when a smooth blob has a known position,
    orientation, and two different spatial scales. It is useful for checking
    whether a filter responds consistently to elongated targets rather than
    only circular ones. The visual notebook and cross-section document also use
    this function as the oriented blob example.

    `length_sigma` controls the standard deviation along the rotated long
    axis, and `width_sigma` controls the standard deviation along the
    perpendicular short axis. `angle_rad` is the long-axis angle in radians,
    measured from the image `+x` direction. `center_x` and `center_y` move the
    blob center in the shared centered coordinate system. `amplitude` is the
    peak intensity at the blob center.

    The rendered intensity is
    `amplitude * exp(-0.5 * ((u / length_sigma)^2 + (v / width_sigma)^2))`, where
    `u = dx * cos(angle) + dy * sin(angle)` and
    `v = -dx * sin(angle) + dy * cos(angle)`. The returned `Frame` contains the
    intensity image and the closed-form gradients with respect to image `x`
    and `y`. If `device` is omitted and a tensor parameter is passed, the
    render stays on that tensor's device. All grid and parameter tensors are
    created on the resolved device, so CUDA inputs run the same vectorized path
    on the GPU.
    """
    validate_amplitude("amplitude", amplitude)
    validate_positive("length_sigma", length_sigma)
    validate_positive("width_sigma", width_sigma)
    device = infer_device(
        device,
        length_sigma,
        width_sigma,
        angle_rad,
        center_x,
        center_y,
        amplitude,
    )
    batch_size = infer_batch_size(
        length_sigma,
        width_sigma,
        angle_rad,
        center_x,
        center_y,
        amplitude,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    length_sigma_batch = as_batch(length_sigma, batch_size, device, dtype)
    width_sigma_batch = as_batch(width_sigma, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_x_batch = as_batch(center_x, batch_size, device, dtype)
    center_y_batch = as_batch(center_y, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    u_coord = x_from_center * cos_angle + y_from_center * sin_angle
    v_coord = -x_from_center * sin_angle + y_from_center * cos_angle
    inv_length_sigma_sq = torch.reciprocal(length_sigma_batch * length_sigma_batch)
    inv_width_sigma_sq = torch.reciprocal(width_sigma_batch * width_sigma_batch)

    u_scaled = u_coord * inv_length_sigma_sq
    v_scaled = v_coord * inv_width_sigma_sq
    exponent = -0.5 * (u_coord * u_scaled + v_coord * v_scaled)
    intensity = amplitude_batch * torch.exp(exponent)
    gradient_x = intensity * (-u_scaled * cos_angle + v_scaled * sin_angle)
    gradient_y = intensity * (-u_scaled * sin_angle - v_scaled * cos_angle)
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)
