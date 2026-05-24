"""Ando consistent gradient operators.

From Ando, "Consistent Gradient Operators" (IEEE TPAMI 2000). The operators are
derived by an orthogonal decomposition of the difference between the continuous
gradient and the discrete operator into an intrinsic smoothing term and a
self-inconsistency term, then minimizing only the self-inconsistency. The
resulting masks estimate gradient direction exactly for locally one-dimensional
patterns, independent of orientation, spectral content, and sub-pixel shift.

Each operator is a smoothing kernel applied along one axis and a central
difference along the other. The 3x3 mask is Ando's exact optimal solution; the
4x4 and 5x5 masks shipped here are the standard separable approximations of
Ando's optimal (non-separable) operators. The smoothing kernels are normalized
to unit sum and the difference kernels are scaled for unit gradient gain.

The 4x4 operator is even-sized, so it estimates the gradient on the dual grid
and carries an inherent half-pixel localization offset; it has no separable
execution path and runs as a dense kernel with explicit padding.
"""

from __future__ import annotations

import torch

from agfb_filters.filters.definitions import (
    GradientFilterDefinition,
    define_dense_filter,
    define_separable_filter,
)
from agfb_filters.runtime.execution import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
)
from agfb_filters.runtime.runner import run_filter

_REPLICATE = BoundaryCondition(BoundaryMode.REPLICATE)

# 3x3 (exact optimal). Smoothing renormalized to unit sum, central difference.
_SMOOTH_3 = torch.tensor([0.225474, 0.549052, 0.225474])
_DERIVATIVE_3 = torch.tensor([-0.5, 0.0, 0.5])

# 5x5 (separable approximation). Difference scaled by 0.784406 for unit gain.
_SMOOTH_5 = torch.tensor([0.0357338, 0.248861, 0.43081, 0.248861, 0.0357338])
_DERIVATIVE_5 = torch.tensor([-0.137424, -0.362576, 0.0, 0.362576, 0.137424]) * 0.784406

# 4x4 (separable approximation, even support). Difference scaled by 1.46205884
# for unit gain. Built as a dense outer product because even kernels have no
# separable execution path in this library.
_SMOOTH_4 = torch.tensor([0.0919833, 0.408017, 0.408017, 0.0919833])
_DERIVATIVE_4 = torch.tensor([-0.0919833, -0.408017, 0.408017, 0.0919833]) * 1.46205884
_KERNEL_X_4 = torch.outer(_SMOOTH_4, _DERIVATIVE_4)
_KERNEL_Y_4 = torch.outer(_DERIVATIVE_4, _SMOOTH_4)

_DEFINITION_3 = define_separable_filter(
    name="ando_3",
    smooth_kernel_1d=_SMOOTH_3,
    derivative_kernel_1d=_DERIVATIVE_3,
    default_boundary=_REPLICATE,
    operator_family="ando_consistent",
    metadata={"size": 3, "approximation": "exact"},
    references=("Ando2000Consistent",),
)
_DEFINITION_4 = define_dense_filter(
    name="ando_4",
    kernel_x=_KERNEL_X_4,
    kernel_y=_KERNEL_Y_4,
    default_boundary=_REPLICATE,
    spatial_padding=(1, 2, 1, 2),
    symmetry="odd",
    operator_family="ando_consistent",
    metadata={"size": 4, "approximation": "separable", "dual_grid": True},
    references=("Ando2000Consistent",),
)
_DEFINITION_5 = define_separable_filter(
    name="ando_5",
    smooth_kernel_1d=_SMOOTH_5,
    derivative_kernel_1d=_DERIVATIVE_5,
    default_boundary=_REPLICATE,
    operator_family="ando_consistent",
    metadata={"size": 5, "approximation": "separable"},
    references=("Ando2000Consistent",),
)
FILTER_SPECS = (
    {
        "name": "ando_3",
        "definition_factory": "ando_3_definition",
        "description": "Ando consistent 3x3",
        "exports": ("ando_3", "ando_3_definition"),
        "smoke_path": "separable",
    },
    {
        "name": "ando_4",
        "definition_factory": "ando_4_definition",
        "description": "Ando consistent 4x4",
        "exports": ("ando_4", "ando_4_definition"),
        "smoke_path": "spatial_dense",
    },
    {
        "name": "ando_5",
        "definition_factory": "ando_5_definition",
        "description": "Ando consistent 5x5",
        "exports": ("ando_5", "ando_5_definition"),
        "smoke_path": "separable",
    },
)


def ando_3_definition() -> GradientFilterDefinition:
    return _DEFINITION_3


def ando_4_definition() -> GradientFilterDefinition:
    return _DEFINITION_4


def ando_5_definition() -> GradientFilterDefinition:
    return _DEFINITION_5


def ando_3(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION_3, image, path=path, boundary=boundary)


def ando_4(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION_4, image, path=path, boundary=boundary)


def ando_5(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_filter(_DEFINITION_5, image, path=path, boundary=boundary)
