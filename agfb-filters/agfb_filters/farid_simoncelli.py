"""Farid-Simoncelli 5-tap matched-pair derivative.

From Farid & Simoncelli, "Differentiation of Discrete Multidimensional
Signals" (IEEE TIP 2004). The 5-tap interpolation and derivative filters are
jointly optimized to recover spatial derivatives with low cross-orientation
bias.
"""

from __future__ import annotations

import torch

from agfb_filters.definitions import GradientFilterDefinition
from agfb_filters.execution import ExecutionPath, ExecutionPlan
from agfb_filters.runner import run_filter

_PREFILTER = torch.tensor([0.030320, 0.249724, 0.439911, 0.249724, 0.030320])
_DERIVATIVE_KERNEL = torch.tensor([-0.104550, -0.292315, 0.0, 0.292315, 0.104550])
_DEFINITION = GradientFilterDefinition(
    name="farid_simoncelli_5",
    padding_mode="replicate",
    smooth_kernel_1d=_PREFILTER,
    derivative_kernel_1d=_DERIVATIVE_KERNEL,
    support="separable",
    symmetry="odd",
)


def farid_simoncelli_5_definition() -> GradientFilterDefinition:
    return _DEFINITION


def farid_simoncelli_5(
    image: torch.Tensor,
    *,
    path: ExecutionPath | ExecutionPlan | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION, image, path=path)
