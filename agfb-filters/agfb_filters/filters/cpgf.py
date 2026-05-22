"""Circular Polynomial Gradient Filter.

This filter fits a polynomial least-squares surface over a discrete disc and
uses the linear polynomial terms as horizontal and vertical gradient kernels.
Kernel construction is vectorized with torch operations so the solve can run
on the requested device.
"""

from __future__ import annotations

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition
from agfb_filters.filters.polynomial import build_polynomial_gradient_kernels
from agfb_filters.runtime.execution import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
)
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "cpgf",
        "definition_factory": "cpgf_definition",
        "description": "circular polynomial gradient filter",
        "exports": ("CPGF", "cpgf_definition", "cpgf_kernels"),
        "smoke_kwargs": {"radius": 2, "degree": 2},
        "smoke_path": "sparse_offsets",
    },
)


def cpgf_kernels(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CPGF disc-polynomial cross-correlation kernels.

    Returns `(kernel_x, kernel_y)`. Each kernel has shape
    `(2 * radius + 1, 2 * radius + 1)` and dtype float32.
    """
    return build_polynomial_gradient_kernels(
        radius=radius,
        degree=degree,
        support="disc",
        device=device,
    )


def cpgf_definition(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> GradientFilterDefinition:
    kernel_x, kernel_y = cpgf_kernels(radius=radius, degree=degree, device=device)
    return GradientFilterDefinition(
        name="cpgf",
        default_boundary=BoundaryCondition(BoundaryMode.REFLECT),
        kernel_x=kernel_x,
        kernel_y=kernel_y,
        support="disc",
        symmetry="odd",
        metadata={"radius": int(radius), "degree": int(degree)},
        operator_family="polynomial_least_squares",
        support_shape="disc",
        parameters={"radius": int(radius), "degree": int(degree)},
    )


class CPGF:
    """Holds prebuilt CPGF kernels for one `(radius, degree)` configuration."""

    def __init__(self, radius: int, degree: int, device: torch.device | None = None) -> None:
        self.radius = int(radius)
        self.degree = int(degree)
        self.definition = cpgf_definition(radius=radius, degree=degree, device=device)

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, path=path, boundary=boundary)
