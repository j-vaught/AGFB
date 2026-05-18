"""L-junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators._junction import junction_frame
from agfb_generators.base import Frame, Numeric


def smoothed_l_junction(
    height: int,
    width: int,
    *,
    arm_width_px: Numeric,
    theta_rad: Numeric = 0.0,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a smoothed L junction with two perpendicular arms."""
    return junction_frame(
        height,
        width,
        angles_rad=[theta_rad, theta_rad + math.pi / 2.0],
        arm_width_px=arm_width_px,
        x0=x0,
        y0=y0,
        contrast=contrast,
        sigma_e=sigma_e,
        device=device,
        dtype=dtype,
    )


def hard_l_junction(
    height: int,
    width: int,
    *,
    arm_width_px: Numeric,
    theta_rad: Numeric = 0.0,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render the sharpest band-limited L junction used by AGFB."""
    return smoothed_l_junction(
        height,
        width,
        arm_width_px=arm_width_px,
        theta_rad=theta_rad,
        x0=x0,
        y0=y0,
        contrast=contrast,
        sigma_e=0.5,
        device=device,
        dtype=dtype,
    )
