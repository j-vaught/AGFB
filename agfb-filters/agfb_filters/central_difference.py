"""Central finite-difference baseline (no smoothing).

The 3-tap kernel `[-1, 0, 1] / 2` applied along the column axis gives
`gradient_x`. Applied along the row axis it gives `gradient_y`.
"""

from __future__ import annotations

import torch

from agfb_filters.definitions import ExecutionStrategy, GradientFilterDefinition
from agfb_filters.runner import run_filter

_SMOOTH_KERNEL = torch.tensor([1.0])  # identity along the smoothing axis
_DERIVATIVE_KERNEL = torch.tensor([-1.0, 0.0, 1.0]) / 2.0
_DEFINITION = GradientFilterDefinition(
    name="central_difference",
    padding_mode="replicate",
    smooth_kernel_1d=_SMOOTH_KERNEL,
    derivative_kernel_1d=_DERIVATIVE_KERNEL,
    strategy_hint=ExecutionStrategy.SEPARABLE,
    support="separable",
)


def central_difference_definition() -> GradientFilterDefinition:
    return _DEFINITION


def central_difference(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION, image)
