"""Scharr-3 (separable, replicate padding).

Optimized for rotational symmetry. The horizontal-gradient kernel uses
`[3, 10, 3] / 16` along the smoothing axis and `[-1, 0, 1] / 2` along the
differentiation axis. The combined 3x3 kernel is the standard Scharr operator.
"""

from __future__ import annotations

import torch

from agfb_filters.definitions import ExecutionStrategy, GradientFilterDefinition
from agfb_filters.runner import run_filter

_SMOOTH_KERNEL = torch.tensor([3.0, 10.0, 3.0]) / 16.0
_DERIVATIVE_KERNEL = torch.tensor([-1.0, 0.0, 1.0]) / 2.0
_DEFINITION = GradientFilterDefinition(
    name="scharr_3",
    padding_mode="replicate",
    smooth_kernel_1d=_SMOOTH_KERNEL,
    derivative_kernel_1d=_DERIVATIVE_KERNEL,
    strategy_hint=ExecutionStrategy.SEPARABLE,
    support="separable",
)


def scharr_3_definition() -> GradientFilterDefinition:
    return _DEFINITION


def scharr_3(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION, image)
