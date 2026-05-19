"""L-junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators._junction import junction_frame
from agfb_generators.base import Frame, Numeric, infer_device


def smoothed_l_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric,
    angle_rad: Numeric = 0.0,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a smoothed L-shaped junction with two half-bar arms.

    The benchmark uses this generator when a filter should respond to a corner
    where two finite-width structures meet at a shared endpoint. The visual
    notebook and cross-section document also use it as the simplest junction
    example before T-, Y-, and X-junction cases.

    `arm_width` controls the width of both arms in pixels. `angle_rad` is the
    direction of the first arm, measured from the image `+x` direction, and the
    second arm is rotated 90 degrees from it. `center_x` and `center_y` place
    the junction point in the shared centered coordinate system. `amplitude`
    controls the high-side intensity, and `edge_sigma` controls the Gaussian
    softening of each arm boundary and endpoint gate.

    The returned `Frame` contains the smooth union intensity and the closed-form
    gradients produced by the shared half-bar junction renderer. If `device` is
    omitted and a tensor parameter is passed, the render stays on that tensor's
    device.
    """
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
        angles_rad=[angle_rad, angle_rad + math.pi / 2.0],
        arm_width_px=arm_width,
        x0=center_x,
        y0=center_y,
        contrast=amplitude,
        sigma_e=edge_sigma,
        device=device,
        dtype=dtype,
    )


def hard_l_junction(
    height: int,
    width: int,
    *,
    arm_width: Numeric,
    angle_rad: Numeric = 0.0,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render the sharpest band-limited L junction used by the benchmark.

    This is the sharp L-junction counterpart to `smoothed_l_junction`. It keeps
    the same geometry and intensity parameters but fixes `edge_sigma` to
    0.5 px so the corner is close to discontinuous while retaining a sampled
    analytic gradient.
    """
    return smoothed_l_junction(
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
