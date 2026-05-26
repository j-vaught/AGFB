"""Two parallel opposite-sign smoothed edges."""

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
    normalize_contrast,
    validate_amplitude,
    validate_positive,
)


def smoothed_bar(
    height: int,
    width: int,
    *,
    bar_width: Numeric = 32.0,
    angle_rad: Numeric = 0.0,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched soft bar as two opposite smoothed edges.

    The benchmark uses this generator when a filter should respond to a
    finite-width bright band with two nearby parallel edges. It is useful for
    checking paired-edge behavior, bar-width sensitivity, and whether a filter
    merges or separates opposite signed edge responses. The visual notebook,
    cross-section document, and smoothed-bar preview use this function as the
    soft spatial bar example.

    The default call renders a horizontal 32 px bar centered in the image with
    a 3 px Gaussian edge spread. `bar_width` is the distance between the two
    edge centers in pixels. `angle_rad` is the bar normal direction in radians,
    measured from the image `+x` direction. `center_offset` shifts the bar
    center in the shared centered coordinate system. `amplitude` is the
    plateau intensity between the two edges, and `edge_sigma` controls the
    smoothness of each edge.

    The implementation renders a positive smoothed step at
    `center_offset - bar_width / 2` and a matching negative smoothed step at
    `center_offset + bar_width / 2`, then sums their intensities and analytic
    gradients into one `Frame`. If `device` is omitted and a tensor parameter
    is passed, the render stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    validate_positive("bar_width", bar_width)
    validate_positive("edge_sigma", edge_sigma)
    device = infer_device(device, bar_width, angle_rad, center_offset, amplitude, edge_sigma)
    batch_size = infer_batch_size(bar_width, angle_rad, center_offset, amplitude, edge_sigma)
    xx, yy = coord_grid(height, width, device, dtype)

    bar_width_batch = as_batch(bar_width, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    edge_sigma_batch = as_batch(edge_sigma, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    bar_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    half_width = bar_width_batch / 2.0

    low_edge_scaled = (bar_coord + half_width) / edge_sigma_batch
    high_edge_scaled = (bar_coord - half_width) / edge_sigma_batch
    intensity = amplitude_batch * (gauss_Phi(low_edge_scaled) - gauss_Phi(high_edge_scaled))
    normal_gradient = (amplitude_batch / edge_sigma_batch) * (
        gauss_phi(low_edge_scaled) - gauss_phi(high_edge_scaled)
    )
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)
