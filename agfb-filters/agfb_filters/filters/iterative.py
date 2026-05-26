"""Iterative diffusion gradient filters."""

from __future__ import annotations

from functools import cache

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition, define_iterative_filter
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "perona_malik_gradient",
        "definition_factory": "perona_malik_gradient_definition",
        "description": "Perona-Malik diffusion followed by central gradient",
        "exports": (
            "PeronaMalikGradient",
            "perona_malik_gradient",
            "perona_malik_gradient_definition",
        ),
        "smoke_kwargs": {"iterations": 2, "step_size": 0.15, "kappa": 0.2},
        "smoke_path": "iterative",
    },
)


@cache
def perona_malik_gradient_definition(
    iterations: int = 5,
    step_size: float = 0.15,
    kappa: float = 0.2,
    conduction: str = "exponential",
    derivative_radius: int = 1,
) -> GradientFilterDefinition:
    return define_iterative_filter(
        name="perona_malik_gradient",
        iterations=int(iterations),
        step_size=float(step_size),
        kappa=float(kappa),
        conduction=str(conduction),
        derivative_radius=int(derivative_radius),
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        metadata={
            "iterations": int(iterations),
            "step_size": float(step_size),
            "kappa": float(kappa),
            "conduction": str(conduction),
            "derivative_radius": int(derivative_radius),
        },
        references=("Perona1990ScaleSpace",),
    )


def perona_malik_gradient(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    iterations: int = 5,
    step_size: float = 0.15,
    kappa: float = 0.2,
    conduction: str = "exponential",
    derivative_radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    definition = perona_malik_gradient_definition(
        iterations=iterations,
        step_size=step_size,
        kappa=kappa,
        conduction=conduction,
        derivative_radius=derivative_radius,
    )
    return run_filter(definition, image, path=path, boundary=boundary)


class PeronaMalikGradient:
    """Holds one Perona-Malik gradient configuration."""

    def __init__(
        self,
        iterations: int = 5,
        step_size: float = 0.15,
        kappa: float = 0.2,
        conduction: str = "exponential",
        derivative_radius: int = 1,
    ) -> None:
        self.definition = perona_malik_gradient_definition(
            iterations=iterations,
            step_size=step_size,
            kappa=kappa,
            conduction=conduction,
            derivative_radius=derivative_radius,
        )

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, path=path, boundary=boundary)
