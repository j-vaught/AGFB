"""Farid-Simoncelli matched-pair derivatives.

From Farid & Simoncelli, "Differentiation of Discrete Multidimensional
Signals" (IEEE TIP 2004). The 5-tap interpolation and derivative filters are
jointly optimized to recover spatial derivatives with low cross-orientation
bias.
"""

from __future__ import annotations

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition
from agfb_filters.runtime.execution import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
)
from agfb_filters.runtime.runner import run_filter

_PREFILTER_5 = torch.tensor([0.030320, 0.249724, 0.439911, 0.249724, 0.030320])
_DERIVATIVE_KERNEL_5 = torch.tensor([-0.104550, -0.292315, 0.0, 0.292315, 0.104550])
_PREFILTER_7 = torch.tensor([0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711])
_DERIVATIVE_KERNEL_7 = torch.tensor(
    [-0.018708, -0.125376, -0.193091, 0.0, 0.193091, 0.125376, 0.018708]
)
_DEFINITION_5 = GradientFilterDefinition(
    name="farid_simoncelli_5",
    default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
    smooth_kernel_1d=_PREFILTER_5,
    derivative_kernel_1d=_DERIVATIVE_KERNEL_5,
    support="separable",
    symmetry="odd",
    operator_family="farid_simoncelli",
    references=("Farid2004Differentiation",),
)
_DEFINITION_7 = GradientFilterDefinition(
    name="farid_simoncelli_7",
    default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
    smooth_kernel_1d=_PREFILTER_7,
    derivative_kernel_1d=_DERIVATIVE_KERNEL_7,
    support="separable",
    symmetry="odd",
    operator_family="farid_simoncelli",
    references=("Farid2004Differentiation",),
)
FILTER_SPECS = (
    {
        "name": "farid_simoncelli_5",
        "definition_factory": "farid_simoncelli_5_definition",
        "description": "Farid-Simoncelli 5-tap",
        "exports": ("farid_simoncelli_5", "farid_simoncelli_5_definition"),
        "smoke_path": "separable",
    },
    {
        "name": "farid_simoncelli_7",
        "definition_factory": "farid_simoncelli_7_definition",
        "description": "Farid-Simoncelli 7-tap",
        "exports": ("farid_simoncelli_7", "farid_simoncelli_7_definition"),
        "smoke_path": "separable",
    },
)


def farid_simoncelli_5_definition() -> GradientFilterDefinition:
    return _DEFINITION_5


def farid_simoncelli_7_definition() -> GradientFilterDefinition:
    return _DEFINITION_7


def farid_simoncelli_5(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION_5, image, path=path, boundary=boundary)


def farid_simoncelli_7(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION_7, image, path=path, boundary=boundary)
