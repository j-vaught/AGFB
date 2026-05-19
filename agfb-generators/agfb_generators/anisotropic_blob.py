"""Rotated anisotropic Gaussian blob generator."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def anisotropic_blob(
    height: int,
    width: int,
    *,
    length_sigma: Numeric,
    width_sigma: Numeric,
    theta_rad: Numeric,
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
    perpendicular short axis. `theta_rad` is the long-axis angle in radians,
    measured from the image `+x` direction. `center_x` and `center_y` move the
    blob center in the shared centered coordinate system. `amplitude` is the
    peak intensity at the blob center.

    The rendered intensity is
    `amplitude * exp(-0.5 * ((u / length_sigma)^2 + (v / width_sigma)^2))`, where
    `u = dx * cos(theta) + dy * sin(theta)` and
    `v = -dx * sin(theta) + dy * cos(theta)`. The returned `Frame` contains the
    intensity image and the closed-form gradients with respect to image `x`
    and `y`.
    """
    device = device or torch.device("cpu")
    batch_size = infer_batch_size(
        length_sigma,
        width_sigma,
        theta_rad,
        center_x,
        center_y,
        amplitude,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    length_sigma_batch = as_batch(length_sigma, batch_size, device, dtype)
    width_sigma_batch = as_batch(width_sigma, batch_size, device, dtype)
    theta_batch = as_batch(theta_rad, batch_size, device, dtype)
    center_x_batch = as_batch(center_x, batch_size, device, dtype)
    center_y_batch = as_batch(center_y, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    cos_theta = torch.cos(theta_batch)
    sin_theta = torch.sin(theta_batch)
    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    u_coord = x_from_center * cos_theta + y_from_center * sin_theta
    v_coord = -x_from_center * sin_theta + y_from_center * cos_theta
    length_sigma_sq = length_sigma_batch * length_sigma_batch
    width_sigma_sq = width_sigma_batch * width_sigma_batch

    exponent = -0.5 * ((u_coord * u_coord) / length_sigma_sq + (v_coord * v_coord) / width_sigma_sq)
    intensity = amplitude_batch * torch.exp(exponent)
    gradient_x = intensity * (
        -(u_coord / length_sigma_sq) * cos_theta + (v_coord / width_sigma_sq) * sin_theta
    )
    gradient_y = intensity * (
        -(u_coord / length_sigma_sq) * sin_theta - (v_coord / width_sigma_sq) * cos_theta
    )
    return pack(intensity, gradient_x, gradient_y)
