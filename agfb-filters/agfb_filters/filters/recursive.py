"""Recursive Gaussian derivative filters."""

from __future__ import annotations

from functools import cache

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition, define_recursive_filter
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "deriche_recursive_gaussian_derivative",
        "definition_factory": "deriche_recursive_gaussian_derivative_definition",
        "description": "Deriche-style recursive Gaussian derivative",
        "exports": (
            "DericheRecursiveGaussianDerivative",
            "deriche_recursive_gaussian_derivative",
            "deriche_recursive_gaussian_derivative_definition",
        ),
        "smoke_kwargs": {"sigma": 1.0},
        "smoke_path": "recursive",
    },
)


@cache
def deriche_recursive_gaussian_derivative_definition(
    sigma: float = 1.0,
) -> GradientFilterDefinition:
    return define_recursive_filter(
        name="deriche_recursive_gaussian_derivative",
        sigma=float(sigma),
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        metadata={"sigma": float(sigma)},
        references=("Young1995Recursive",),
        supported_boundaries=(BoundaryMode.REPLICATE,),
    )


def deriche_recursive_gaussian_derivative(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    sigma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    definition = deriche_recursive_gaussian_derivative_definition(sigma=sigma)
    return run_filter(definition, image, path=path, boundary=boundary)


class DericheRecursiveGaussianDerivative:
    """Holds one recursive Gaussian derivative configuration."""

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = float(sigma)
        self.definition = deriche_recursive_gaussian_derivative_definition(sigma=sigma)

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, path=path, boundary=boundary)
