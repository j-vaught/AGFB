"""First-order orientable filter banks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache

import torch

from agfb_filters.filters.definitions import (
    FilterImplementationKind,
    GradientFilterDefinition,
    GradientFilterImplementation,
)
from agfb_filters.filters.derivative_of_gaussian import derivative_of_gaussian_definition
from agfb_filters.filters.farid_simoncelli import (
    farid_simoncelli_5_definition,
    farid_simoncelli_7_definition,
)
from agfb_filters.filters.recursive import deriche_recursive_gaussian_derivative_definition
from agfb_filters.filters.riesz import riesz_transform_definition
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import (
    OrientationBankResult,
    orientation_angles,
    run_orientation_bank,
    run_steered_filter_bank,
)

FILTER_SPECS = (
    {
        "name": "steered_gaussian_derivative_orientation_bank",
        "definition_factory": "steered_gaussian_derivative_orientation_bank_definition",
        "description": "steered first derivative of Gaussian orientation bank",
        "exports": (
            "steered_gaussian_derivative_orientation_bank",
            "steered_gaussian_derivative_orientation_bank_definition",
        ),
        "smoke_kwargs": {"angle_count": 4, "sigma": 1.0},
        "smoke_path": "orientation_bank",
        "output_api": "orientation_bank",
    },
    {
        "name": "farid_simoncelli_5_orientation_bank",
        "definition_factory": "farid_simoncelli_5_orientation_bank_definition",
        "description": "Farid-Simoncelli 5-tap first-derivative orientation bank",
        "exports": (
            "farid_simoncelli_5_orientation_bank",
            "farid_simoncelli_5_orientation_bank_definition",
        ),
        "smoke_kwargs": {"angle_count": 4},
        "smoke_path": "orientation_bank",
        "output_api": "orientation_bank",
    },
    {
        "name": "farid_simoncelli_7_orientation_bank",
        "definition_factory": "farid_simoncelli_7_orientation_bank_definition",
        "description": "Farid-Simoncelli 7-tap first-derivative orientation bank",
        "exports": (
            "farid_simoncelli_7_orientation_bank",
            "farid_simoncelli_7_orientation_bank_definition",
        ),
        "smoke_kwargs": {"angle_count": 4},
        "smoke_path": "orientation_bank",
        "output_api": "orientation_bank",
    },
    {
        "name": "matched_edge_orientation_bank",
        "definition_factory": "matched_edge_orientation_bank_definition",
        "description": "rotated first-derivative matched edge orientation bank",
        "exports": (
            "matched_edge_derivative_kernels",
            "matched_edge_orientation_bank",
            "matched_edge_orientation_bank_definition",
        ),
        "smoke_kwargs": {"angle_count": 4, "sigma_across": 1.0, "sigma_along": 3.0},
        "smoke_path": "orientation_bank",
        "output_api": "orientation_bank",
    },
)


@cache
def steered_gaussian_derivative_orientation_bank_definition(
    angle_count: int = 8,
    sigma: float = 1.0,
    truncate: float = 4.0,
) -> GradientFilterDefinition:
    source = derivative_of_gaussian_definition(sigma=float(sigma), truncate=float(truncate))
    angles = orientation_angles(int(angle_count))
    return _steered_kernel_bank_definition(
        name="steered_gaussian_derivative_orientation_bank",
        source=source,
        angles=angles,
        operator_family="steered_gaussian_derivative",
        support_shape="isotropic_steered",
        parameters={
            "angle_count": int(angle_count),
            "sigma": float(sigma),
            "truncate": float(truncate),
        },
        references=("Freeman1991Steerable", "Canny1986EdgeDetection"),
    )


def steered_gaussian_derivative_orientation_bank(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    angle_count: int = 8,
    sigma: float = 1.0,
    truncate: float = 4.0,
) -> OrientationBankResult:
    definition = steered_gaussian_derivative_orientation_bank_definition(
        angle_count=angle_count,
        sigma=sigma,
        truncate=truncate,
    )
    return run_orientation_bank(
        definition,
        image,
        path=path,
        boundary=definition.default_boundary if boundary is None else boundary,
    )


@cache
def farid_simoncelli_5_orientation_bank_definition(
    angle_count: int = 8,
) -> GradientFilterDefinition:
    angles = orientation_angles(int(angle_count))
    return _steered_kernel_bank_definition(
        name="farid_simoncelli_5_orientation_bank",
        source=farid_simoncelli_5_definition(),
        angles=angles,
        operator_family="farid_simoncelli_orientation_bank",
        support_shape="separable_steered",
        parameters={"angle_count": int(angle_count), "tap_count": 5},
        references=("Farid2004Differentiation",),
    )


@cache
def farid_simoncelli_7_orientation_bank_definition(
    angle_count: int = 8,
) -> GradientFilterDefinition:
    angles = orientation_angles(int(angle_count))
    return _steered_kernel_bank_definition(
        name="farid_simoncelli_7_orientation_bank",
        source=farid_simoncelli_7_definition(),
        angles=angles,
        operator_family="farid_simoncelli_orientation_bank",
        support_shape="separable_steered",
        parameters={"angle_count": int(angle_count), "tap_count": 7},
        references=("Farid2004Differentiation",),
    )


def farid_simoncelli_5_orientation_bank(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    angle_count: int = 8,
) -> OrientationBankResult:
    definition = farid_simoncelli_5_orientation_bank_definition(angle_count=angle_count)
    return run_orientation_bank(
        definition,
        image,
        path=path,
        boundary=definition.default_boundary if boundary is None else boundary,
    )


def farid_simoncelli_7_orientation_bank(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    angle_count: int = 8,
) -> OrientationBankResult:
    definition = farid_simoncelli_7_orientation_bank_definition(angle_count=angle_count)
    return run_orientation_bank(
        definition,
        image,
        path=path,
        boundary=definition.default_boundary if boundary is None else boundary,
    )


@cache
def matched_edge_orientation_bank_definition(
    angle_count: int = 8,
    sigma_across: float = 1.0,
    sigma_along: float = 6.0,
    truncate: float = 3.0,
) -> GradientFilterDefinition:
    angles = orientation_angles(int(angle_count))
    kernels = matched_edge_derivative_kernels(
        angles,
        sigma_across=float(sigma_across),
        sigma_along=float(sigma_along),
        truncate=float(truncate),
    )
    radius = int(kernels.shape[-1] // 2)
    return GradientFilterDefinition(
        name="matched_edge_orientation_bank",
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        implementation=GradientFilterImplementation(
            kind=FilterImplementationKind.ORIENTATION_BANK,
            orientation_kernels=kernels,
            angles=angles,
            spatial_padding=(radius, radius, radius, radius),
        ),
        support="orientation_bank",
        symmetry="odd",
        metadata={
            "angle_count": int(angle_count),
            "sigma_across": float(sigma_across),
            "sigma_along": float(sigma_along),
            "truncate": float(truncate),
        },
        operator_family="matched_edge_derivative",
        orientation_model="orientation_bank",
        support_shape="rotated_matched_edge",
        parameters={
            "angle_count": int(angle_count),
            "sigma_across": float(sigma_across),
            "sigma_along": float(sigma_along),
            "truncate": float(truncate),
        },
        references=("Chaudhuri1989MatchedFilter",),
    )


def matched_edge_orientation_bank(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
    angle_count: int = 8,
    sigma_across: float = 1.0,
    sigma_along: float = 6.0,
    truncate: float = 3.0,
) -> OrientationBankResult:
    definition = matched_edge_orientation_bank_definition(
        angle_count=angle_count,
        sigma_across=sigma_across,
        sigma_along=sigma_along,
        truncate=truncate,
    )
    return run_orientation_bank(
        definition,
        image,
        path=path,
        boundary=definition.default_boundary if boundary is None else boundary,
    )


def recursive_gaussian_derivative_orientation_bank(
    image: torch.Tensor,
    *,
    angle_count: int = 8,
    sigma: float = 1.0,
    path: ExecutionPath | str = ExecutionPath.RECURSIVE,
    boundary: BoundaryCondition | None = None,
) -> OrientationBankResult:
    definition = deriche_recursive_gaussian_derivative_definition(sigma=float(sigma))
    return run_steered_filter_bank(
        definition,
        image,
        angles=orientation_angles(int(angle_count), dtype=image.dtype, device=image.device),
        path=path,
        boundary=definition.default_boundary if boundary is None else boundary,
    )


def multiscale_gaussian_derivative_orientation_banks(
    image: torch.Tensor,
    *,
    sigmas: Sequence[float] = (1.0, 2.0, 4.0),
    angle_count: int = 8,
    truncate: float = 4.0,
    path: ExecutionPath | str = ExecutionPath.SEPARABLE,
    boundary: BoundaryCondition | None = None,
) -> tuple[OrientationBankResult, ...]:
    angles = orientation_angles(int(angle_count), dtype=image.dtype, device=image.device)
    results: list[OrientationBankResult] = []
    for sigma in sigmas:
        definition = derivative_of_gaussian_definition(sigma=float(sigma), truncate=float(truncate))
        result = run_steered_filter_bank(
            definition,
            image,
            angles=angles,
            path=path,
            boundary=definition.default_boundary if boundary is None else boundary,
        )
        results.append(
            OrientationBankResult(
                responses=result.responses,
                angles=result.angles,
                definition_name=f"derivative_of_gaussian_sigma_{float(sigma):g}_steered",
            )
        )
    return tuple(results)


def riesz_orientation_bank(
    image: torch.Tensor,
    *,
    angle_count: int = 8,
    path: ExecutionPath | str = ExecutionPath.FFT,
    boundary: BoundaryCondition | None = None,
    epsilon: float = 1.0e-12,
) -> OrientationBankResult:
    definition = riesz_transform_definition(epsilon=float(epsilon))
    return run_steered_filter_bank(
        definition,
        image,
        angles=orientation_angles(int(angle_count), dtype=image.dtype, device=image.device),
        path=path,
        boundary=definition.default_boundary if boundary is None else boundary,
    )


def matched_edge_derivative_kernels(
    angles: torch.Tensor,
    *,
    sigma_across: float,
    sigma_along: float,
    truncate: float,
) -> torch.Tensor:
    sigma_across = _positive_float(sigma_across, name="sigma_across")
    sigma_along = _positive_float(sigma_along, name="sigma_along")
    truncate = _positive_float(truncate, name="truncate")
    radius = max(1, int(math.ceil(truncate * max(sigma_across, sigma_along))))
    coordinates = torch.arange(-radius, radius + 1, dtype=angles.dtype, device=angles.device)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernels: list[torch.Tensor] = []
    for theta in angles:
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        normal = columns * cos_theta + rows * sin_theta
        tangent = -columns * sin_theta + rows * cos_theta
        envelope = torch.exp(-0.5 * ((normal / sigma_across) ** 2 + (tangent / sigma_along) ** 2))
        basis_x = columns * envelope
        basis_y = rows * envelope
        moment_matrix = torch.stack(
            (
                torch.stack((torch.sum(basis_x * columns), torch.sum(basis_y * columns))),
                torch.stack((torch.sum(basis_x * rows), torch.sum(basis_y * rows))),
            )
        )
        target_moments = torch.stack((cos_theta, sin_theta))
        coefficients = torch.linalg.solve(moment_matrix, target_moments)
        kernel = coefficients[0] * basis_x + coefficients[1] * basis_y
        kernel = kernel - kernel.mean()
        kernels.append(kernel)
    return torch.stack(kernels, dim=0)


def _steered_kernel_bank_definition(
    *,
    name: str,
    source: GradientFilterDefinition,
    angles: torch.Tensor,
    operator_family: str,
    support_shape: str,
    parameters: dict[str, int | float],
    references: tuple[str, ...],
) -> GradientFilterDefinition:
    kernels = _steered_kernels_from_source(source, angles)
    kernel_height = int(kernels.shape[-2])
    kernel_width = int(kernels.shape[-1])
    return GradientFilterDefinition(
        name=name,
        default_boundary=source.default_boundary,
        implementation=GradientFilterImplementation(
            kind=FilterImplementationKind.ORIENTATION_BANK,
            orientation_kernels=kernels,
            angles=angles,
            spatial_padding=(
                kernel_width // 2,
                kernel_width // 2,
                kernel_height // 2,
                kernel_height // 2,
            ),
        ),
        support="orientation_bank",
        symmetry="odd",
        metadata=parameters,
        operator_family=operator_family,
        orientation_model="steered_first_derivative",
        support_shape=support_shape,
        parameters=parameters,
        references=references,
        supported_boundaries=source.supported_boundaries,
    )


def _steered_kernels_from_source(
    source: GradientFilterDefinition,
    angles: torch.Tensor,
) -> torch.Tensor:
    kernel_x, kernel_y = source.dense_kernels()
    dtype = torch.promote_types(kernel_x.dtype, angles.dtype)
    kernel_x = kernel_x.to(dtype=dtype, device=angles.device)
    kernel_y = kernel_y.to(dtype=dtype, device=angles.device)
    cosines = torch.cos(angles).view(-1, 1, 1)
    sines = torch.sin(angles).view(-1, 1, 1)
    return (cosines * kernel_x.unsqueeze(0) + sines * kernel_y.unsqueeze(0)).contiguous()


def _positive_float(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
