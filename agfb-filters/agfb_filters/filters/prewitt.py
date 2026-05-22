"""Prewitt-3 (separable, replicate padding)."""

from __future__ import annotations

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition
from agfb_filters.runtime.execution import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
)
from agfb_filters.runtime.runner import run_filter

_SMOOTH_KERNEL = torch.tensor([1.0, 1.0, 1.0]) / 3.0
_DERIVATIVE_KERNEL = torch.tensor([-1.0, 0.0, 1.0]) / 2.0
_DEFINITION = GradientFilterDefinition(
    name="prewitt_3",
    default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
    smooth_kernel_1d=_SMOOTH_KERNEL,
    derivative_kernel_1d=_DERIVATIVE_KERNEL,
    support="separable",
    symmetry="odd",
)
FILTER_SPECS = (
    {
        "name": "prewitt_3",
        "definition_factory": "prewitt_3_definition",
        "description": "Prewitt 3-tap",
        "exports": ("prewitt_3", "prewitt_3_definition"),
        "smoke_path": "separable",
    },
)


def prewitt_3_definition() -> GradientFilterDefinition:
    return _DEFINITION


def prewitt_3(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION, image, path=path, boundary=boundary)
