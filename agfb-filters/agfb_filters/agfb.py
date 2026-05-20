"""AGFB disc-polynomial gradient filter.

This filter fits a polynomial least-squares surface over a discrete disc and
uses the linear polynomial terms as horizontal and vertical gradient kernels.
Kernel construction is vectorized with torch operations so the solve can run
on the requested device.
"""

from __future__ import annotations

import torch

from agfb_filters.base import check_input, fft_cross_correlation
from agfb_filters.polynomial import build_polynomial_gradient_kernels


def agfb_kernels(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build AGFB disc-polynomial cross-correlation kernels.

    Returns `(kernel_x, kernel_y)`. Each kernel has shape
    `(2 * radius + 1, 2 * radius + 1)` and dtype float32.
    """
    return build_polynomial_gradient_kernels(
        radius=radius,
        degree=degree,
        support="disc",
        device=device,
    )


class AGFB:
    """Holds prebuilt AGFB kernels for one `(radius, degree)` configuration."""

    def __init__(self, radius: int, degree: int, device: torch.device | None = None) -> None:
        self.radius = int(radius)
        self.degree = int(degree)
        self.kernel_x, self.kernel_y = agfb_kernels(radius=radius, degree=degree, device=device)

    def apply(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = check_input(image)
        kernel_x = self.kernel_x.to(device=image.device, dtype=image.dtype)
        kernel_y = self.kernel_y.to(device=image.device, dtype=image.dtype)
        return fft_cross_correlation(image, (kernel_x, kernel_y), pad_mode="reflect")
