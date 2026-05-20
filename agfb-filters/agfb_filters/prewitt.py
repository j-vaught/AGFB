"""Prewitt-3 (separable, replicate padding)."""

from __future__ import annotations

import torch

from agfb_filters.definitions import GradientFilterDefinition
from agfb_filters.execution import ExecutionPath, ExecutionPlan
from agfb_filters.runner import run_filter

_SMOOTH_KERNEL = torch.tensor([1.0, 1.0, 1.0]) / 3.0
_DERIVATIVE_KERNEL = torch.tensor([-1.0, 0.0, 1.0]) / 2.0
_DEFINITION = GradientFilterDefinition(
    name="prewitt_3",
    padding_mode="replicate",
    smooth_kernel_1d=_SMOOTH_KERNEL,
    derivative_kernel_1d=_DERIVATIVE_KERNEL,
    support="separable",
    symmetry="odd",
)


def prewitt_3_definition() -> GradientFilterDefinition:
    return _DEFINITION


def prewitt_3(
    image: torch.Tensor,
    *,
    path: ExecutionPath | ExecutionPlan | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION, image, path=path)
