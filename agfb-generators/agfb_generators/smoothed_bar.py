"""Two parallel opposite-sign smoothed edges."""

from __future__ import annotations

import torch

from agfb_generators.base import Frame, Numeric, infer_device, pack, validate_positive
from agfb_generators.smoothed_step import smoothed_step


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
    validate_positive("bar_width", bar_width)
    validate_positive("edge_sigma", edge_sigma)
    device = infer_device(device, bar_width, angle_rad, center_offset, amplitude, edge_sigma)
    half_width = bar_width / 2.0 if isinstance(bar_width, torch.Tensor) else float(bar_width) / 2.0

    if isinstance(center_offset, torch.Tensor):
        positive_edge_offset = center_offset + half_width
        negative_edge_offset = center_offset - half_width
    elif isinstance(half_width, torch.Tensor):
        positive_edge_offset = half_width + float(center_offset)
        negative_edge_offset = -half_width + float(center_offset)
    else:
        positive_edge_offset = float(center_offset) + half_width
        negative_edge_offset = float(center_offset) - half_width

    negative_amplitude = -amplitude if isinstance(amplitude, torch.Tensor) else -float(amplitude)

    rising_edge = smoothed_step(
        height,
        width,
        theta_rad=angle_rad,
        x0=negative_edge_offset,
        contrast=amplitude,
        sigma_e=edge_sigma,
        device=device,
        dtype=dtype,
    )
    falling_edge = smoothed_step(
        height,
        width,
        theta_rad=angle_rad,
        x0=positive_edge_offset,
        contrast=negative_amplitude,
        sigma_e=edge_sigma,
        device=device,
        dtype=dtype,
    )
    intensity = rising_edge.I + falling_edge.I
    gradient_x = rising_edge.gx + falling_edge.gx
    gradient_y = rising_edge.gy + falling_edge.gy
    return pack(intensity, gradient_x, gradient_y)
