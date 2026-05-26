"""Haar-style rectangular box gradient filters."""

from __future__ import annotations

from functools import cache

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition, define_box_gradient_filter
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "haar_box_gradient",
        "definition_factory": "haar_box_gradient_definition",
        "description": "Haar rectangular box gradient",
        "exports": ("haar_box_gradient", "haar_box_gradient_definition"),
        "smoke_kwargs": {"radius": 1},
        "smoke_path": "box_integral",
    },
)


@cache
def haar_box_gradient_definition(radius: int = 1) -> GradientFilterDefinition:
    radius = int(radius)
    return define_box_gradient_filter(
        name="haar_box_gradient",
        radius=radius,
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        metadata={"radius": radius},
    )


def haar_box_gradient(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    definition = haar_box_gradient_definition(radius=radius)
    return run_filter(definition, image, path=path, boundary=boundary)
