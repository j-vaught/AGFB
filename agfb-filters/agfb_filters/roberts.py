"""Roberts cross (2x2, half-pixel offset).

The classical Roberts cross approximates the diagonal derivatives
`diagonal_down` and `diagonal_up`, evaluated at pixel center
`(row + 0.5, column + 0.5)`. Projecting onto image axes gives the horizontal
and vertical gradients.

The half-pixel offset is intrinsic to Roberts and part of what the benchmark
measures. Output is `(batch, height, width)` via replicate padding on the right
and bottom.
"""

from __future__ import annotations

import torch

from agfb_filters.definitions import GradientFilterDefinition
from agfb_filters.execution import BoundaryCondition, BoundaryMode, ExecutionPath, ExecutionPlan
from agfb_filters.runner import run_filter

_KERNEL_X = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) / 2.0
_KERNEL_Y = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / 2.0
_DEFINITION = GradientFilterDefinition(
    name="roberts",
    default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
    kernel_x=_KERNEL_X,
    kernel_y=_KERNEL_Y,
    spatial_padding=(0, 1, 0, 1),
    support="offset_2x2",
    symmetry="odd",
)


def roberts_definition() -> GradientFilterDefinition:
    return _DEFINITION


def roberts(
    image: torch.Tensor,
    *,
    path: ExecutionPath | ExecutionPlan | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION, image, path=path, boundary=boundary)
