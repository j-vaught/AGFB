"""Derivative-of-Gaussian filter.

Separable smoothing is applied along one axis and first-derivative filtering is
applied along the other. Kernel half-width is `ceil(truncate * sigma)`.

This is the first-derivative-of-Gaussian operator, not the bandpass difference
of two Gaussian smoothing kernels.
"""

from __future__ import annotations

import math

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition
from agfb_filters.runtime.execution import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
)
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "derivative_of_gaussian",
        "definition_factory": "derivative_of_gaussian_definition",
        "description": "first derivative of Gaussian",
        "exports": ("DerivativeOfGaussian", "derivative_of_gaussian_definition"),
        "smoke_kwargs": {"sigma": 1.0},
        "smoke_path": "separable",
    },
)


def derivative_of_gaussian_definition(
    sigma: float,
    truncate: float = 4.0,
) -> GradientFilterDefinition:
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = max(1, int(math.ceil(truncate * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
    gaussian_kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    derivative_kernel = (offsets / (sigma**2)) * gaussian_kernel
    derivative_kernel = derivative_kernel - derivative_kernel.mean()
    return GradientFilterDefinition(
        name="derivative_of_gaussian",
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        smooth_kernel_1d=gaussian_kernel.to(torch.float32),
        derivative_kernel_1d=derivative_kernel.to(torch.float32),
        support="separable",
        symmetry="odd",
        metadata={"sigma": float(sigma), "truncate": float(truncate), "radius": radius},
        operator_family="gaussian_derivative",
        references=("Young1995Recursive",),
    )


class DerivativeOfGaussian:
    """Holds prebuilt 1-D smoothing and derivative kernels."""

    def __init__(self, sigma: float, truncate: float = 4.0) -> None:
        self.sigma = float(sigma)
        self.definition = derivative_of_gaussian_definition(sigma=sigma, truncate=truncate)
        self.radius = int(self.definition.metadata["radius"])

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, path=path, boundary=boundary)
