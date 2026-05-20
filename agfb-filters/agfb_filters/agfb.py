"""AGFB disc-polynomial gradient filter.

This filter fits a polynomial least-squares surface over a discrete disc and
uses the linear polynomial terms as horizontal and vertical gradient kernels.
Kernel construction is vectorized with torch operations so the solve can run
on the requested device.
"""

from __future__ import annotations

import torch

from agfb_filters.definitions import ExecutionStrategy, GradientFilterDefinition
from agfb_filters.polynomial import build_polynomial_gradient_kernels
from agfb_filters.runner import run_filter


def agfb_kernels(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build AGFB disc-polynomial cross-correlation kernels.

    Returns `(kernel_x, kernel_y)`. Each kernel has shape
    `(2 * radius + 1, 2 * radius + 1)` and dtype float32.
    """
    return build_polynomial_gradient_kernels(
        radius=radius,
        degree=degree,
        support="disc",
        device=device,
    )


def agfb_definition(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> GradientFilterDefinition:
    kernel_x, kernel_y = agfb_kernels(radius=radius, degree=degree, device=device)
    return GradientFilterDefinition(
        name="agfb",
        padding_mode="reflect",
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        strategy_hint=ExecutionStrategy.AUTO,
        support="disc",
        metadata={"radius": int(radius), "degree": int(degree)},
    )


class AGFB:
    """Holds prebuilt AGFB kernels for one `(radius, degree)` configuration."""

    def __init__(self, radius: int, degree: int, device: torch.device | None = None) -> None:
        self.radius = int(radius)
        self.degree = int(degree)
        self.definition = agfb_definition(radius=radius, degree=degree, device=device)

    def apply(
        self,
        image: torch.Tensor,
        *,
        strategy: ExecutionStrategy | str = ExecutionStrategy.AUTO,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, strategy=strategy)
