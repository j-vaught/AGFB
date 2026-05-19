"""Radially smoothed curved edge generator."""

from __future__ import annotations

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    gauss_Phi,
    gauss_phi,
    infer_batch_size,
    infer_device,
    pack,
)


def curved_arc(
    height: int,
    width: int,
    *,
    radius: Numeric,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched smoothed circular boundary as a curved edge.

    The benchmark uses this generator when a filter should handle a curved
    transition instead of only a straight step. The function renders the
    inside of a disk with a Gaussian cumulative distribution transition at the
    boundary. If the disk is centered in the crop, the image looks like a full
    circle. If the center is shifted outside the crop, the visible boundary is
    a local arc.

    `radius` sets the disk radius in pixels. `center_x` and `center_y` place
    the disk center in the shared centered coordinate system. `edge_sigma`
    controls the Gaussian transition width at the boundary, and `amplitude`
    controls the interior intensity.

    The rendered intensity is
    `amplitude * Phi((radius - rho) / edge_sigma)`, where
    `rho = sqrt((x - center_x)^2 + (y - center_y)^2)`. The returned `Frame`
    contains the intensity image and the closed-form inward gradients with
    respect to image `x` and `y`. If `device` is omitted and a tensor parameter
    is passed, the render stays on that tensor's device.
    """
    device = infer_device(device, radius, center_x, center_y, amplitude, edge_sigma)
    batch_size = infer_batch_size(radius, center_x, center_y, amplitude, edge_sigma)
    xx, yy = coord_grid(height, width, device, dtype)

    radius_batch = as_batch(radius, batch_size, device, dtype)
    center_x_batch = as_batch(center_x, batch_size, device, dtype)
    center_y_batch = as_batch(center_y, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    edge_sigma_batch = as_batch(edge_sigma, batch_size, device, dtype)

    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    radial_distance = torch.sqrt(
        x_from_center * x_from_center + y_from_center * y_from_center
    ).clamp_min(1e-12)
    normalized_distance = (radius_batch - radial_distance) / edge_sigma_batch

    intensity = amplitude_batch * gauss_Phi(normalized_distance)
    radial_derivative = -(amplitude_batch / edge_sigma_batch) * gauss_phi(normalized_distance)
    gradient_x = radial_derivative * (x_from_center / radial_distance)
    gradient_y = radial_derivative * (y_from_center / radial_distance)
    return pack(intensity, gradient_x, gradient_y)
