"""Gaussian-smoothed finite-width ramp generator."""

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


def smoothed_ramp(
    height: int,
    width: int,
    *,
    ramp_width: Numeric = 64.0,
    angle_rad: Numeric = 0.0,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched Gaussian-smoothed finite-width ramp.

    The benchmark uses this generator when a filter should respond to a finite
    linear transition whose two kink locations have been softened by a Gaussian
    edge-spread kernel. It is useful for testing ramp localization and slope
    recovery without the discontinuities of `finite_ramp`. The visual notebook,
    cross-section document, and smoothed-ramp preview use this function as the
    soft finite-transition example.

    The default call renders a horizontal 64 px ramp centered in the image
    with a 3 px Gaussian edge spread. `ramp_width` is the distance in pixels
    between the low and high ramp edges. `angle_rad` is the ramp normal
    direction in radians, measured from the image `+x` direction.
    `center_offset` shifts the midpoint of the ramp in the shared centered
    coordinate system, `amplitude` is the high-side intensity, and
    `edge_sigma` controls Gaussian smoothing at the two ramp edges.

    The projected coordinate is
    `z = x * cos(angle) + y * sin(angle) - center_offset`. The rendered
    intensity is the closed-form convolution of a finite linear ramp over
    `-ramp_width / 2 <= z <= ramp_width / 2` with a Gaussian kernel of standard
    deviation `edge_sigma`. The returned `Frame` contains that intensity image
    and the closed-form gradient, whose normal component is the difference of
    two Gaussian cumulative distribution functions. If `device` is omitted and
    a tensor parameter is passed, the render stays on that tensor's device.
    """
    validate_positive("ramp_width", ramp_width)
    validate_positive("edge_sigma", edge_sigma)
    device = infer_device(device, ramp_width, angle_rad, center_offset, amplitude, edge_sigma)
    batch_size = infer_batch_size(ramp_width, angle_rad, center_offset, amplitude, edge_sigma)
    xx, yy = coord_grid(height, width, device, dtype)

    ramp_width_batch = as_batch(ramp_width, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    edge_sigma_batch = as_batch(edge_sigma, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    ramp_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    half_width = ramp_width_batch / 2.0
    low_edge_coord = ramp_coord + half_width
    high_edge_coord = ramp_coord - half_width

    low_edge_scaled = low_edge_coord / edge_sigma_batch
    high_edge_scaled = high_edge_coord / edge_sigma_batch
    low_edge_integral = low_edge_coord * gauss_Phi(low_edge_scaled) + edge_sigma_batch * gauss_phi(
        low_edge_scaled
    )
    high_edge_integral = high_edge_coord * gauss_Phi(
        high_edge_scaled
    ) + edge_sigma_batch * gauss_phi(high_edge_scaled)

    inv_ramp_width = torch.reciprocal(ramp_width_batch)
    intensity = amplitude_batch * inv_ramp_width * (low_edge_integral - high_edge_integral)
    normal_gradient = (
        amplitude_batch
        * inv_ramp_width
        * (gauss_Phi(low_edge_scaled) - gauss_Phi(high_edge_scaled))
    )
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return pack(intensity, gradient_x, gradient_y)
