"""T-junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators._junction import junction_frame
from agfb_generators.base import Frame, Numeric, infer_device, validate_positive


def smoothed_t_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric = 18.0,
    angle_rad: Numeric = 0.0,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched smoothed T-shaped junction.

    The benchmark uses this generator when a filter should respond to three
    finite-width structures meeting at one endpoint. It is useful for checking
    junction localization, occlusion-like T structure, arm continuity, and
    whether a filter preserves the stem-to-crossbar meeting point. The visual
    notebook, cross-section document, and T-junction preview use this function
    as the smoothed three-arm junction example.

    The default call renders an 18 px wide T junction centered in the image
    with a 2 px Gaussian edge spread. `arm_width` controls all three arm
    widths. `angle_rad` is the stem direction in radians, measured from the
    image `+x` direction, and the two crossbar arms are perpendicular to it.
    `center_x` and `center_y` place the junction point in the shared centered
    coordinate system. `amplitude` controls the high-side intensity, and
    `edge_sigma` controls boundary and endpoint softening.

    The returned `Frame` contains the smooth union intensity and closed-form
    gradients produced by the shared stroked-ray signed-distance renderer. If
    `device` is omitted and a tensor parameter is passed, the render stays on
    that tensor's device.
    """
    validate_positive("arm_width", arm_width)
    validate_positive("edge_sigma", edge_sigma)
    device = infer_device(
        device,
        arm_width,
        angle_rad,
        center_x,
        center_y,
        amplitude,
        edge_sigma,
    )
    return junction_frame(
        height,
        width,
        angles_rad=[angle_rad, angle_rad + math.pi / 2.0, angle_rad - math.pi / 2.0],
        arm_width_px=arm_width,
        x0=center_x,
        y0=center_y,
        contrast=amplitude,
        sigma_e=edge_sigma,
        device=device,
        dtype=dtype,
    )


def hard_t_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric = 18.0,
    angle_rad: Numeric = 0.0,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render the sharpest band-limited T junction used by the benchmark.

    This is the sharp T-junction counterpart to `smoothed_t_junction`. It keeps
    the same geometry and intensity parameters but fixes `edge_sigma` to
    0.5 px so the junction is close to discontinuous while retaining a sampled
    analytic gradient.
    """
    return smoothed_t_junction(
        height,
        width,
        arm_width=arm_width,
        angle_rad=angle_rad,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        edge_sigma=0.5,
        device=device,
        dtype=dtype,
    )
