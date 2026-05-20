"""Savitzky-Golay 2-D gradient with a square polynomial fit window."""

from __future__ import annotations

import torch

from agfb_filters.base import check_input, fft_cross_correlation
from agfb_filters.polynomial import build_polynomial_gradient_kernels


def savitzky_golay_kernels(
    radius: int,
    degree: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build square-window Savitzky-Golay cross-correlation kernels."""
    return build_polynomial_gradient_kernels(
        radius=radius,
        degree=degree,
        support="square",
        device=device,
    )


class SavitzkyGolay:
    """Holds prebuilt kernels for one `(radius, degree)` configuration."""

    def __init__(self, radius: int, degree: int, device: torch.device | None = None) -> None:
        self.radius = int(radius)
        self.degree = int(degree)
        self.kernel_x, self.kernel_y = savitzky_golay_kernels(
            radius=radius,
            degree=degree,
            device=device,
        )

    def apply(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = check_input(image)
        kernel_x = self.kernel_x.to(device=image.device, dtype=image.dtype)
        kernel_y = self.kernel_y.to(device=image.device, dtype=image.dtype)
        return fft_cross_correlation(image, (kernel_x, kernel_y), pad_mode="reflect")
