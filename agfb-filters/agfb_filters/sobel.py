"""Sobel-3 / Sobel-5 / Sobel-7 (separable, replicate padding).

The 5- and 7-tap versions recursively convolve the 3-tap Sobel derivative
with the binomial smoother `[1, 2, 1] / 4`.
"""

from __future__ import annotations

from functools import cache

import torch

from agfb_filters.base import linear_convolution_1d
from agfb_filters.definitions import GradientFilterDefinition
from agfb_filters.execution import ExecutionPath, ExecutionPlan
from agfb_filters.runner import run_filter

_SMOOTH_KERNEL_3 = torch.tensor([1.0, 2.0, 1.0]) / 4.0
_DERIVATIVE_KERNEL_3 = torch.tensor([-1.0, 0.0, 1.0]) / 2.0


def _build_kernels(kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"Sobel kernel size must be odd and >= 3, got {kernel_size}")
    smooth = _SMOOTH_KERNEL_3
    derivative = _DERIVATIVE_KERNEL_3
    for _ in range((kernel_size - 3) // 2):
        smooth = linear_convolution_1d(smooth, _SMOOTH_KERNEL_3)
        derivative = linear_convolution_1d(derivative, _SMOOTH_KERNEL_3)
    return smooth, derivative


@cache
def sobel_definition(kernel_size: int) -> GradientFilterDefinition:
    smooth_kernel, derivative_kernel = _build_kernels(kernel_size)
    return GradientFilterDefinition(
        name=f"sobel_{kernel_size}",
        padding_mode="replicate",
        smooth_kernel_1d=smooth_kernel,
        derivative_kernel_1d=derivative_kernel,
        support="separable",
        symmetry="odd",
        metadata={"kernel_size": kernel_size},
    )


def sobel_3(
    image: torch.Tensor,
    *,
    path: ExecutionPath | ExecutionPlan | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(sobel_definition(3), image, path=path)


def sobel_5(
    image: torch.Tensor,
    *,
    path: ExecutionPath | ExecutionPlan | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(sobel_definition(5), image, path=path)


def sobel_7(
    image: torch.Tensor,
    *,
    path: ExecutionPath | ExecutionPlan | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(sobel_definition(7), image, path=path)
