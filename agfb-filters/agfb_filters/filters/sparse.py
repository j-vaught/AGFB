"""Direct sparse-offset finite differences."""

from __future__ import annotations

from functools import cache

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition, define_sparse_offset_filter
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "sparse_central_difference",
        "definition_factory": "sparse_central_difference_definition",
        "description": "direct sparse central finite difference",
        "exports": ("sparse_central_difference", "sparse_central_difference_definition"),
        "smoke_kwargs": {"radius": 1},
        "smoke_path": "sparse_offsets",
    },
)


@cache
def sparse_central_difference_definition(radius: int = 1) -> GradientFilterDefinition:
    radius = int(radius)
    if radius < 1:
        raise ValueError("radius must be >= 1")
    offsets = torch.tensor(
        [
            [0, -radius],
            [0, radius],
            [-radius, 0],
            [radius, 0],
        ],
        dtype=torch.int64,
    )
    scale = 1.0 / float(2 * radius)
    weights_x = torch.tensor([-scale, scale, 0.0, 0.0])
    weights_y = torch.tensor([0.0, 0.0, -scale, scale])
    return define_sparse_offset_filter(
        name="sparse_central_difference",
        offsets=offsets,
        weights_x=weights_x,
        weights_y=weights_y,
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        metadata={"radius": radius},
        parameters={"radius": radius},
    )


def sparse_central_difference(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    definition = sparse_central_difference_definition(radius=radius)
    return run_filter(definition, image, path=path, boundary=boundary)
