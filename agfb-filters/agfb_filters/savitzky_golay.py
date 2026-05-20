"""Savitzky-Golay 2-D gradient with a square polynomial fit window."""

from __future__ import annotations

import torch

from agfb_filters.definitions import ExecutionStrategy, GradientFilterDefinition
from agfb_filters.polynomial import build_polynomial_gradient_kernels
from agfb_filters.runner import run_filter


def savitzky_golay_kernels(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build square-window Savitzky-Golay cross-correlation kernels."""
    return build_polynomial_gradient_kernels(
        radius=radius,
        degree=degree,
        support="square",
        device=device,
    )


def savitzky_golay_definition(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> GradientFilterDefinition:
    kernel_x, kernel_y = savitzky_golay_kernels(radius=radius, degree=degree, device=device)
    return GradientFilterDefinition(
        name="savitzky_golay",
        padding_mode="reflect",
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        strategy_hint=ExecutionStrategy.AUTO,
        support="square",
        metadata={"radius": int(radius), "degree": int(degree)},
    )


class SavitzkyGolay:
    """Holds prebuilt kernels for one `(radius, degree)` configuration."""

    def __init__(self, radius: int, degree: int, device: torch.device | None = None) -> None:
        self.radius = int(radius)
        self.degree = int(degree)
        self.definition = savitzky_golay_definition(radius=radius, degree=degree, device=device)

    def apply(
        self,
        image: torch.Tensor,
        *,
        strategy: ExecutionStrategy | str = ExecutionStrategy.AUTO,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, strategy=strategy)
