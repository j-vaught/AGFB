"""Rotated anisotropic Gaussian blob generator."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, as_batch, coord_grid, infer_batch_size, pack


def anisotropic_blob(
    height: int,
    width: int,
    *,
    sigma_u: Numeric,
    sigma_v: Numeric,
    theta_rad: Numeric,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian peak with independent rotated axis scales.

    The benchmark uses this generator when a smooth blob has a known position,
    orientation, and two different spatial scales. It is useful for checking
    whether a filter responds consistently to elongated targets rather than
    only circular ones. The visual notebook and cross-section document also use
    this function as the oriented blob example.

    `sigma_u` controls the standard deviation along the rotated `u` axis, and
    `sigma_v` controls the standard deviation along the perpendicular `v` axis.
    `theta_rad` is the angle of the `u` axis measured in radians from the
    image `+x` direction. `x0` and `y0` move the blob center in the shared
    centered coordinate system.

    The rendered intensity is
    `contrast * exp(-0.5 * ((u / sigma_u)^2 + (v / sigma_v)^2))`, where
    `u = dx * cos(theta) + dy * sin(theta)` and
    `v = -dx * sin(theta) + dy * cos(theta)`. The returned `Frame` contains the
    intensity image and the closed-form gradients with respect to image `x`
    and `y`.
    """
    device = device or torch.device("cpu")
    batch_size = infer_batch_size(sigma_u, sigma_v, theta_rad, x0, y0, contrast)
    xx, yy = coord_grid(height, width, device, dtype)

    sigma_u_batch = as_batch(sigma_u, batch_size, device, dtype)
    sigma_v_batch = as_batch(sigma_v, batch_size, device, dtype)
    theta_batch = as_batch(theta_rad, batch_size, device, dtype)
    center_x = as_batch(x0, batch_size, device, dtype)
    center_y = as_batch(y0, batch_size, device, dtype)
    contrast_batch = as_batch(contrast, batch_size, device, dtype)

    cos_theta = torch.cos(theta_batch)
    sin_theta = torch.sin(theta_batch)
    x_from_center = xx - center_x
    y_from_center = yy - center_y
    u_coord = x_from_center * cos_theta + y_from_center * sin_theta
    v_coord = -x_from_center * sin_theta + y_from_center * cos_theta
    sigma_u_sq = sigma_u_batch * sigma_u_batch
    sigma_v_sq = sigma_v_batch * sigma_v_batch

    exponent = -0.5 * ((u_coord * u_coord) / sigma_u_sq + (v_coord * v_coord) / sigma_v_sq)
    intensity = contrast_batch * torch.exp(exponent)
    gradient_x = intensity * (
        -(u_coord / sigma_u_sq) * cos_theta + (v_coord / sigma_v_sq) * sin_theta
    )
    gradient_y = intensity * (
        -(u_coord / sigma_u_sq) * sin_theta - (v_coord / sigma_v_sq) * cos_theta
    )
    return pack(intensity, gradient_x, gradient_y)
