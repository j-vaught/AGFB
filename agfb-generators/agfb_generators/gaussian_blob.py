"""Isotropic two-dimensional Gaussian peak generator."""

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


def gaussian_blob(
    height: int,
    width: int,
    *,
    scale_sigma: Numeric,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched isotropic Gaussian peak.

    The benchmark uses this generator when a smooth circular blob has a known
    center, scale, and analytic gradient. It is useful for checking filters on
    a compact target whose gradient direction changes continuously around the
    peak.

    `scale_sigma` controls the standard deviation in pixels in both image
    directions. `center_x` and `center_y` move the peak in the shared centered
    coordinate system. `amplitude` is the peak intensity at the blob center.

    The rendered intensity is
    `amplitude * exp(-((x - center_x)^2 + (y - center_y)^2) / (2 * scale_sigma^2))`.
    The returned `Frame` contains that intensity image and the closed-form
    gradients with respect to image `x` and `y`. If `device` is omitted and a
    tensor parameter is passed, the render stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    validate_positive("scale_sigma", scale_sigma)
    device = infer_device(device, scale_sigma, center_x, center_y, amplitude)
    batch_size = infer_batch_size(scale_sigma, center_x, center_y, amplitude)
    xx, yy = coord_grid(height, width, device, dtype)

    scale_sigma_batch = as_batch(scale_sigma, batch_size, device, dtype)
    center_x_batch = as_batch(center_x, batch_size, device, dtype)
    center_y_batch = as_batch(center_y, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)

    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    scale_sigma_sq = scale_sigma_batch * scale_sigma_batch
    intensity = amplitude_batch * torch.exp(
        -(x_from_center * x_from_center + y_from_center * y_from_center) / (2.0 * scale_sigma_sq)
    )
    gradient_x = -intensity * x_from_center / scale_sigma_sq
    gradient_y = -intensity * y_from_center / scale_sigma_sq
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)
