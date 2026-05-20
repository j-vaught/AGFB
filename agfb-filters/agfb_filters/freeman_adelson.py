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

from agfb_filters.derivative_of_gaussian import DerivativeOfGaussian


class FreemanAdelsonG1:
    def __init__(self, sigma: float, truncate: float = 4.0) -> None:
        self.sigma = float(sigma)
        self._derivative_of_gaussian = DerivativeOfGaussian(sigma=sigma, truncate=truncate)

    def apply(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._derivative_of_gaussian.apply(image)
