"""Y-junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators._junction import junction_frame
from agfb_generators.base import Frame, Numeric

_DEFAULT_Y_JUNCTION_ANGLE_RAD = -math.pi / 2.0


def smoothed_y_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric = 18.0,
    angle_rad: Numeric = _DEFAULT_Y_JUNCTION_ANGLE_RAD,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched smoothed Y-shaped junction.

    The benchmark uses this generator when a filter should preserve three
    finite-width arms meeting at a single branching point. It is useful for
    checking junction localization, equal-angle branch continuity, and whether
    a filter blurs the branch point into a blob or breaks one arm. The visual
    notebook, cross-section document, and Y-junction preview use this function
    as the three-arm branching junction example.

    The default call renders an 18 px wide upright Y junction centered in the
    image with a 2 px Gaussian edge spread. `arm_width` controls all three arm
    widths. `angle_rad` is the first arm direction in radians, measured from
    image `+x`; the other arms are offset by plus and minus 120 degrees.
    `center_x` and `center_y` place the branch point in the shared centered
    coordinate system. `amplitude` controls the high-side intensity, and
    `edge_sigma` controls boundary and endpoint softening.

    The returned `Frame` contains the smooth union intensity and closed-form
    gradients produced by the shared stroked-ray signed-distance renderer. If
    `device` is omitted and a tensor parameter is passed, the render stays on
    that tensor's device.
    """
    return junction_frame(
        height,
        width,
        angles_rad=[
            angle_rad,
            angle_rad + 2.0 * math.pi / 3.0,
            angle_rad - 2.0 * math.pi / 3.0,
        ],
        arm_width_px=arm_width,
        x0=center_x,
        y0=center_y,
        contrast=amplitude,
        sigma_e=edge_sigma,
        device=device,
        dtype=dtype,
    )


def hard_y_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric = 18.0,
    angle_rad: Numeric = _DEFAULT_Y_JUNCTION_ANGLE_RAD,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render the sharpest band-limited Y junction used by the benchmark.

    This is the sharp Y-junction counterpart to `smoothed_y_junction`. It keeps
    the same geometry and intensity parameters but fixes `edge_sigma` to
    0.5 px so the branch point is close to discontinuous while retaining a
    sampled analytic gradient.
    """
    return smoothed_y_junction(
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
