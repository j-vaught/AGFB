"""Smoothed straight-edge generator."""

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
    validate_positive,
)


def smoothed_step(
    height: int,
    width: int,
    *,
    angle_rad: Numeric = 0.0,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian-smoothed straight edge.

    The benchmark uses this generator when a filter should respond to a single
    straight edge with a known Gaussian edge-spread width. It is the canonical
    soft edge primitive that `smoothed_bar` builds from, and it is useful for
    checking edge localization, orientation response, and analytic-gradient
    metrics on a one-sided transition. The visual notebook, cross-section
    document, and smoothed-step preview use this function as the basic softened
    edge example.

    The default call renders a horizontal edge centered in the image with a
    2 px Gaussian edge spread. `angle_rad` is the edge normal direction in
    radians, measured from the image `+x` direction. `center_offset` shifts the
    edge in the shared centered coordinate system. `amplitude` is the high-side
    intensity, and `edge_sigma` controls the Gaussian transition width.

    The projected coordinate is
    `z = x * cos(angle) + y * sin(angle) - center_offset`. The rendered
    intensity is `amplitude * Phi(z / edge_sigma)`, where `Phi` is the standard
    normal cumulative distribution function. The returned `Frame` contains
    that intensity image and the closed-form gradient aligned with the edge
    normal. If `device` is omitted and a tensor parameter is passed, the render
    stays on that tensor's device.
    """
    validate_positive("edge_sigma", edge_sigma)
    device = infer_device(device, angle_rad, center_offset, amplitude, edge_sigma)
    batch_size = infer_batch_size(angle_rad, center_offset, amplitude, edge_sigma)
    xx, yy = coord_grid(height, width, device, dtype)

    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    edge_sigma_batch = as_batch(edge_sigma, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    edge_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    normalized_distance = edge_coord / edge_sigma_batch

    intensity = amplitude_batch * gauss_Phi(normalized_distance)
    normal_gradient = (amplitude_batch / edge_sigma_batch) * gauss_phi(normalized_distance)
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return pack(intensity, gradient_x, gradient_y)
