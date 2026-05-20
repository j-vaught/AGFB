"""Freeman-Adelson G1 steerable basis.

The G1 basis consists of the two first-order Gaussian derivatives along the
cardinal axes. Any oriented G1 response is a cosine and sine weighted
combination of those axes, so reporting `(gradient_x, gradient_y)` is
equivalent to producing the steerable gradient vector at every pixel.

For first-order edge detection this collapses to a separable
derivative-of-Gaussian, so `FreemanAdelsonG1` is numerically equivalent to
`DerivativeOfGaussian` at the same sigma. It is retained as a distinct filter
name for benchmark reporting.

The higher-order G2/H2 basis (for second-derivative steering and oriented
energy) is not implemented in v1.
"""

from __future__ import annotations

import torch

from agfb_filters.definitions import GradientFilterDefinition
from agfb_filters.derivative_of_gaussian import derivative_of_gaussian_definition
from agfb_filters.execution import BoundaryCondition, ExecutionPath, ExecutionPlan
from agfb_filters.runner import run_filter


def freeman_adelson_g1_definition(
    sigma: float,
    truncate: float = 4.0,
) -> GradientFilterDefinition:
    derivative_definition = derivative_of_gaussian_definition(sigma=sigma, truncate=truncate)
    return GradientFilterDefinition(
        name="freeman_adelson_g1",
        default_boundary=derivative_definition.default_boundary,
        smooth_kernel_1d=derivative_definition.smooth_kernel_1d,
        derivative_kernel_1d=derivative_definition.derivative_kernel_1d,
        support=derivative_definition.support,
        symmetry=derivative_definition.symmetry,
        metadata=derivative_definition.metadata,
    )


class FreemanAdelsonG1:
    def __init__(self, sigma: float, truncate: float = 4.0) -> None:
        self.sigma = float(sigma)
        self.definition = freeman_adelson_g1_definition(sigma=sigma, truncate=truncate)

    def apply(
        self,
        image: torch.Tensor,
        *,
        path: ExecutionPath | ExecutionPlan | str,
        boundary: BoundaryCondition | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return run_filter(self.definition, image, path=path, boundary=boundary)
