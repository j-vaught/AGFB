"""First-order Riesz transform filters."""

from __future__ import annotations

from functools import cache

import torch

from agfb_filters.filters.definitions import GradientFilterDefinition, define_riesz_filter
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import run_filter

FILTER_SPECS = (
    {
        "name": "riesz_transform",
        "definition_factory": "riesz_transform_definition",
        "description": "first-order Riesz transform vector filter",
        "exports": ("RieszTransform", "riesz_transform", "riesz_transform_definition"),
        "smoke_kwargs": {},
        "smoke_path": "fft",
    },
)


@cache
def riesz_transform_definition(epsilon: float = 1.0e-12) -> GradientFilterDefinition:
    return define_riesz_filter(
        name="riesz_transform",
        default_boundary=BoundaryCondition(BoundaryMode.CIRCULAR),
        epsilon=float(epsilon),
        metadata={"epsilon": float(epsilon)},
        references=("Felsberg2001Monogenic",),
        supported_boundaries=(BoundaryMode.CIRCULAR,),
    )


def riesz_transform(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str = ExecutionPath.FFT,
    boundary: BoundaryCondition | None = None,
    epsilon: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    definition = riesz_transform_definition(epsilon=epsilon)
    return run_filter(
        definition,
        image,
        path=path,
        boundary=definition.default_boundary if boundary is None else boundary,
    )


class RieszTransform:
    """Holds one first-order Riesz transform configuration."""

    def __init__(self, epsilon: float = 1.0e-12) -> None:
        self.epsilon = float(epsilon)
        self.definition = riesz_transform_definition(epsilon=epsilon)

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str = ExecutionPath.FFT,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(
            self.definition,
            image,
            path=path,
            boundary=self.definition.default_boundary if boundary is None else boundary,
        )
