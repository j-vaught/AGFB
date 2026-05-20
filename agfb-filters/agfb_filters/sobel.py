"""Sobel-3 / Sobel-5 / Sobel-7 (separable, replicate padding).

The 5- and 7-tap versions recursively convolve the 3-tap Sobel derivative
with the binomial smoother `[1, 2, 1] / 4`.
"""

from __future__ import annotations

import torch

from agfb_filters.base import check_input, linear_convolution_1d, separable_gradient

_SMOOTH_KERNEL_3 = torch.tensor([1.0, 2.0, 1.0]) / 4.0
_DERIVATIVE_KERNEL_3 = torch.tensor([-1.0, 0.0, 1.0]) / 2.0


def _build_kernels(kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"Sobel kernel size must be odd and >= 3, got {kernel_size}")
    smooth = _SMOOTH_KERNEL_3
    derivative = _DERIVATIVE_KERNEL_3
    for _ in range((kernel_size - 3) // 2):
        smooth = linear_convolution_1d(smooth, _SMOOTH_KERNEL_3)
        derivative = linear_convolution_1d(derivative, _SMOOTH_KERNEL_3)
    return smooth, derivative


def _apply(image: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = check_input(image)
    smooth_kernel, derivative_kernel = _build_kernels(kernel_size)
    smooth_kernel = smooth_kernel.to(device=image.device, dtype=image.dtype)
    derivative_kernel = derivative_kernel.to(device=image.device, dtype=image.dtype)
    return separable_gradient(
        image,
        smooth_kernel_1d=smooth_kernel,
        derivative_kernel_1d=derivative_kernel,
    )


def sobel_3(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply(image, 3)


def sobel_5(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply(image, 5)


def sobel_7(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply(image, 7)
