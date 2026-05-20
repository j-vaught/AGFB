"""Derivative-of-Gaussian filter.

Separable smoothing is applied along one axis and first-derivative filtering is
applied along the other. Kernel half-width is `ceil(truncate * sigma)`.

This is the first-derivative-of-Gaussian operator, not the bandpass difference
of two Gaussian smoothing kernels.
"""

from __future__ import annotations

import math

import torch

from agfb_filters.base import check_input, separable_gradient


class DerivativeOfGaussian:
    """Holds prebuilt 1-D smoothing and derivative kernels."""

    def __init__(self, sigma: float, truncate: float = 4.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        radius = max(1, int(math.ceil(truncate * sigma)))
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
        gaussian_kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        derivative_kernel = -(offsets / (sigma**2)) * gaussian_kernel
        derivative_kernel = derivative_kernel - derivative_kernel.mean()
        self.sigma = float(sigma)
        self.radius = radius
        self.smooth_kernel = gaussian_kernel.to(torch.float32)
        self.derivative_kernel = derivative_kernel.to(torch.float32)

    def apply(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = check_input(image)
        smooth_kernel = self.smooth_kernel.to(device=image.device, dtype=image.dtype)
        derivative_kernel = self.derivative_kernel.to(device=image.device, dtype=image.dtype)
        return separable_gradient(
            image,
            smooth_kernel_1d=smooth_kernel,
            derivative_kernel_1d=derivative_kernel,
        )
