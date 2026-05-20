"""Savitzky-Golay 2-D gradient with a square polynomial fit window."""

from __future__ import annotations

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition
from agfb_filters.filters.polynomial import build_polynomial_gradient_kernels
from agfb_filters.runtime.execution import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    ExecutionPlan,
)
from agfb_filters.runtime.runner import run_filter


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
        default_boundary=BoundaryCondition(BoundaryMode.REFLECT),
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        support="square",
        symmetry="odd",
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
        path: ExecutionPath | ExecutionPlan | str,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, path=path, boundary=boundary)
