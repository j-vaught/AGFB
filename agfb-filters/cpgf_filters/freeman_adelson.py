"""Freeman–Adelson G1 steerable basis.

The G1 basis consists of the two first-order Gaussian derivatives along the
cardinal axes: `G1a = ∂G/∂x` and `G1b = ∂G/∂y`. Any oriented G1 response at
angle θ is the linear combination `cos(θ) G1a + sin(θ) G1b`, so reporting
`(g_x, g_y) = (G1a * I, G1b * I)` is equivalent to producing the steerable
gradient vector at every pixel.

For first-order edge detection this collapses to a separable
derivative-of-Gaussian, so `FreemanAdelsonG1` is numerically equivalent to
`DoG` at the same σ. It is retained as a distinct filter name for
benchmark-reporting reasons (the §1.3 table compares the two by construction).

The higher-order G2/H2 basis (for second-derivative steering and oriented
energy) is not implemented in v1; if §1.4 needs it we add it then.
"""

from __future__ import annotations

import torch

from cpgf_filters.derivative_of_gaussian import DoG


class FreemanAdelsonG1:
    def __init__(self, sigma: float, truncate: float = 4.0) -> None:
        self.sigma = float(sigma)
        self._dog = DoG(sigma=sigma, truncate=truncate)

    def apply(self, I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dog.apply(I)
