"""X-junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators._junction import junction_frame
from agfb_generators.base import Frame, Numeric, infer_device, validate_positive

_DEFAULT_X_JUNCTION_ANGLE_RAD = math.pi / 4.0


def smoothed_x_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric = 18.0,
    angle_rad: Numeric = _DEFAULT_X_JUNCTION_ANGLE_RAD,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched smoothed X-shaped junction.

    The benchmark uses this generator when a filter should preserve two
    crossing finite-width structures that meet at one shared junction point. It
    is useful for checking crossing localization, arm continuity through the
    center, and whether the filter creates a center divot or suppresses one
    diagonal. The visual notebook, cross-section document, and X-junction
    preview use this function as the four-arm polygonal junction example.

    The default call renders an 18 px wide diagonal X junction centered in the
    image with a 2 px Gaussian edge spread. `arm_width` controls all four arm
    widths. `angle_rad` is the first arm direction in radians, measured from
    image `+x`; the other arms are spaced at 90 degree intervals. `center_x`
    and `center_y` place the crossing point in the shared centered coordinate
    system. `amplitude` controls the high-side intensity, and `edge_sigma`
    controls boundary and endpoint softening.

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
        angles_rad=[
            angle_rad,
            angle_rad + math.pi / 2.0,
            angle_rad + math.pi,
            angle_rad + 3.0 * math.pi / 2.0,
        ],
        arm_width_px=arm_width,
        x0=center_x,
        y0=center_y,
        contrast=amplitude,
        sigma_e=edge_sigma,
        device=device,
        dtype=dtype,
    )


def hard_x_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric = 18.0,
    angle_rad: Numeric = _DEFAULT_X_JUNCTION_ANGLE_RAD,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render the sharpest band-limited X junction used by the benchmark.

    This is the sharp X-junction counterpart to `smoothed_x_junction`. It keeps
    the same geometry and intensity parameters but fixes `edge_sigma` to
    0.5 px so the crossing is close to discontinuous while retaining a sampled
    analytic gradient.
    """
    return smoothed_x_junction(
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
