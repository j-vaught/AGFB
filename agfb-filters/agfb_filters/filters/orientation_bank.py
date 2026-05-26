"""Rotated anisotropic Gaussian derivative orientation banks."""

from __future__ import annotations

import math
from functools import cache

import torch

from agfb_filters.filters.definitions import (
    GradientFilterDefinition,
    define_orientation_bank_filter,
)
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import OrientationBankResult, run_orientation_bank

FILTER_SPECS = (
    {
        "name": "anisotropic_gaussian_orientation_bank",
        "definition_factory": "anisotropic_gaussian_orientation_bank_definition",
        "description": "rotated anisotropic Gaussian derivative orientation bank",
        "exports": (
            "AnisotropicGaussianOrientationBank",
            "anisotropic_gaussian_orientation_bank",
            "anisotropic_gaussian_orientation_bank_definition",
        ),
        "smoke_kwargs": {"angle_count": 4, "sigma_parallel": 1.0, "sigma_perpendicular": 2.0},
        "smoke_path": "orientation_bank",
        "output_api": "orientation_bank",
    },
)


@cache
def anisotropic_gaussian_orientation_bank_definition(
    angle_count: int = 8,
    sigma_parallel: float = 1.0,
    sigma_perpendicular: float = 2.0,
    truncate: float = 3.0,
) -> GradientFilterDefinition:
    angle_count = int(angle_count)
    if angle_count < 1:
        raise ValueError("angle_count must be >= 1")
    angles = torch.linspace(0.0, math.pi, steps=angle_count + 1, dtype=torch.float32)[:-1]
    return define_orientation_bank_filter(
        name="anisotropic_gaussian_orientation_bank",
        angles=angles,
        sigma_parallel=float(sigma_parallel),
        sigma_perpendicular=float(sigma_perpendicular),
        truncate=float(truncate),
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        metadata={
            "angle_count": angle_count,
            "sigma_parallel": float(sigma_parallel),
            "sigma_perpendicular": float(sigma_perpendicular),
            "truncate": float(truncate),
        },
        references=("Freeman1991Steerable",),
    )


def anisotropic_gaussian_orientation_bank(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    angle_count: int = 8,
    sigma_parallel: float = 1.0,
    sigma_perpendicular: float = 2.0,
    truncate: float = 3.0,
) -> OrientationBankResult:
    definition = anisotropic_gaussian_orientation_bank_definition(
        angle_count=angle_count,
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
        truncate=truncate,
    )
    return run_orientation_bank(definition, image, path=path, boundary=boundary)


class AnisotropicGaussianOrientationBank:
    """Holds one rotated anisotropic Gaussian derivative bank."""

    def __init__(
        self,
        angle_count: int = 8,
        sigma_parallel: float = 1.0,
        sigma_perpendicular: float = 2.0,
        truncate: float = 3.0,
    ) -> None:
        self.definition = anisotropic_gaussian_orientation_bank_definition(
            angle_count=angle_count,
            sigma_parallel=sigma_parallel,
            sigma_perpendicular=sigma_perpendicular,
            truncate=truncate,
        )

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None = None,
    ) -> OrientationBankResult:
        return run_orientation_bank(self.definition, image, path=path, boundary=boundary)
