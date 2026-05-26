"""Robust nonlinear local-window gradient filters."""

from __future__ import annotations

from functools import cache

import torch

from agfb_filters.filters.definitions import (
    GradientFilterDefinition,
    define_nonlinear_window_filter,
)
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "robust_local_plane_gradient",
        "definition_factory": "robust_local_plane_gradient_definition",
        "description": "robust local plane gradient",
        "exports": (
            "RobustLocalPlaneGradient",
            "robust_local_plane_gradient",
            "robust_local_plane_gradient_definition",
        ),
        "smoke_kwargs": {"radius": 1, "weighting": "huber"},
        "smoke_path": "nonlinear_window",
    },
)


@cache
def robust_local_plane_gradient_definition(
    radius: int = 1,
    weighting: str = "huber",
    range_sigma: float = 1.0,
    robust_scale: float = 1.0,
) -> GradientFilterDefinition:
    return define_nonlinear_window_filter(
        name="robust_local_plane_gradient",
        radius=int(radius),
        weighting=str(weighting),
        range_sigma=float(range_sigma),
        robust_scale=float(robust_scale),
        default_boundary=BoundaryCondition(BoundaryMode.REFLECT),
        metadata={
            "radius": int(radius),
            "weighting": str(weighting),
            "range_sigma": float(range_sigma),
            "robust_scale": float(robust_scale),
        },
        references=("Tomasi1998Bilateral",),
    )


def robust_local_plane_gradient(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    radius: int = 1,
    weighting: str = "huber",
    range_sigma: float = 1.0,
    robust_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    definition = robust_local_plane_gradient_definition(
        radius=radius,
        weighting=weighting,
        range_sigma=range_sigma,
        robust_scale=robust_scale,
    )
    return run_filter(definition, image, path=path, boundary=boundary)


class RobustLocalPlaneGradient:
    """Holds one robust local plane gradient configuration."""

    def __init__(
        self,
        radius: int = 1,
        weighting: str = "huber",
        range_sigma: float = 1.0,
        robust_scale: float = 1.0,
    ) -> None:
        self.radius = int(radius)
        self.weighting = str(weighting)
        self.definition = robust_local_plane_gradient_definition(
            radius=radius,
            weighting=weighting,
            range_sigma=range_sigma,
            robust_scale=robust_scale,
        )

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, path=path, boundary=boundary)
