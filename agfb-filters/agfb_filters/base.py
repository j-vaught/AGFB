"""Shared tensor conventions and helpers for AGFB filters.

Input convention.
    `image` is a `(batch, height, width)` float32 tensor on any device.

Output convention.
    Every filter returns `(gradient_x, gradient_y)` tensors with shape
    `(batch, height, width)`.
    `gradient_x` is the horizontal gradient along image columns.
    `gradient_y` is the vertical gradient along image rows.

Padding defaults to `replicate`. Individual filter definitions may override
that when a different boundary convention is part of the filter.
"""

from __future__ import annotations

import torch
import torch.fft as torch_fft
import torch.nn.functional as F


def check_input(image: torch.Tensor) -> torch.Tensor:
    """Validate and contiguous-ify a filter input."""
    if image.ndim != 3:
        raise ValueError(
            f"filter input must be (batch, height, width), got shape {tuple(image.shape)}"
        )
    if image.dtype != torch.float32:
        image = image.to(torch.float32)
    return image.contiguous()


def directional_derivative(
    image: torch.Tensor,
    *,
    smooth_kernel_1d: torch.Tensor,
    derivative_kernel_1d: torch.Tensor,
    axis: int,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Apply a separable directional derivative to one image batch.

    Smoothing is applied perpendicular to `axis`, then differentiation is
    applied along `axis`.

    `axis = 0` differentiates along rows and returns the vertical gradient.
    `axis = 1` differentiates along columns and returns the horizontal gradient.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    smooth_radius = smooth_kernel_1d.shape[0] // 2
    derivative_radius = derivative_kernel_1d.shape[0] // 2
    image_channels = image.unsqueeze(1)
    if axis == 1:
        # smooth along rows (y), differentiate along cols (x)
        smooth_kernel = smooth_kernel_1d.view(1, 1, -1, 1)
        derivative_kernel = derivative_kernel_1d.view(1, 1, 1, -1)
        image_channels = F.pad(
            image_channels,
            (derivative_radius, derivative_radius, smooth_radius, smooth_radius),
            mode=pad_mode,
        )
    else:
        # smooth along cols (x), differentiate along rows (y)
        smooth_kernel = smooth_kernel_1d.view(1, 1, 1, -1)
        derivative_kernel = derivative_kernel_1d.view(1, 1, -1, 1)
        image_channels = F.pad(
            image_channels,
            (smooth_radius, smooth_radius, derivative_radius, derivative_radius),
            mode=pad_mode,
        )
    smoothed_image = F.conv2d(image_channels, smooth_kernel)
    return F.conv2d(smoothed_image, derivative_kernel).squeeze(1)


def separable_gradient(
    image: torch.Tensor,
    *,
    smooth_kernel_1d: torch.Tensor,
    derivative_kernel_1d: torch.Tensor,
    pad_mode: str = "replicate",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(gradient_x, gradient_y)` with smooth-first ordering."""
    gradient_x = directional_derivative(
        image,
        smooth_kernel_1d=smooth_kernel_1d,
        derivative_kernel_1d=derivative_kernel_1d,
        axis=1,
        pad_mode=pad_mode,
    )
    gradient_y = directional_derivative(
        image,
        smooth_kernel_1d=smooth_kernel_1d,
        derivative_kernel_1d=derivative_kernel_1d,
        axis=0,
        pad_mode=pad_mode,
    )
    return gradient_x, gradient_y


def dense_convolution_2d(
    image: torch.Tensor,
    kernel: torch.Tensor,
    *,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Apply an odd-sized dense 2-D kernel with `F.conv2d`."""
    kernel_height, kernel_width = kernel.shape
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError(f"dense kernel dims must be odd, got {kernel_height}x{kernel_width}")
    radius_height = kernel_height // 2
    radius_width = kernel_width // 2
    image_channels = image.unsqueeze(1)
    padded_image = F.pad(
        image_channels,
        (radius_width, radius_width, radius_height, radius_height),
        mode=pad_mode,
    )
    convolution_kernel = kernel.view(1, 1, kernel_height, kernel_width)
    return F.conv2d(padded_image, convolution_kernel).squeeze(1)


def fft_cross_correlation(
    image: torch.Tensor,
    kernels: tuple[torch.Tensor, torch.Tensor],
    *,
    pad_mode: str = "replicate",
    spatial_padding: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-correlate `image` with horizontal and vertical kernels via FFT.

    Returns `(gradient_x, gradient_y)` with the same padding and output shape
    semantics as dense spatial cross-correlation.
    """
    kernel_x, kernel_y = kernels
    if kernel_x.shape != kernel_y.shape:
        raise ValueError(f"kernel shapes must match, got {kernel_x.shape} vs {kernel_y.shape}")
    kernel_height, kernel_width = kernel_x.shape
    if spatial_padding is None:
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("even-sized kernels require explicit spatial padding")
        spatial_padding = (
            kernel_width // 2,
            kernel_width // 2,
            kernel_height // 2,
            kernel_height // 2,
        )

    _, image_height, image_width = image.shape
    padded_image = F.pad(image.unsqueeze(1), spatial_padding, mode=pad_mode).squeeze(1)
    padded_height = int(padded_image.shape[-2])
    padded_width = int(padded_image.shape[-1])
    output_height = padded_height - kernel_height + 1
    output_width = padded_width - kernel_width + 1
    if output_height != image_height or output_width != image_width:
        raise ValueError(
            "spatial padding must preserve input shape, "
            f"got output {output_height}x{output_width} for input {image_height}x{image_width}"
        )

    fft_shape = (padded_height + kernel_height - 1, padded_width + kernel_width - 1)
    padded_kernel_x = torch.zeros(fft_shape, dtype=image.dtype, device=image.device)
    padded_kernel_y = torch.zeros(fft_shape, dtype=image.dtype, device=image.device)
    padded_kernel_x[:kernel_height, :kernel_width] = torch.flip(kernel_x, dims=(0, 1))
    padded_kernel_y[:kernel_height, :kernel_width] = torch.flip(kernel_y, dims=(0, 1))

    image_spectrum = torch_fft.rfft2(padded_image, s=fft_shape)
    gradient_x_full = torch_fft.irfft2(
        image_spectrum * torch_fft.rfft2(padded_kernel_x, s=fft_shape),
        s=fft_shape,
    )
    gradient_y_full = torch_fft.irfft2(
        image_spectrum * torch_fft.rfft2(padded_kernel_y, s=fft_shape),
        s=fft_shape,
    )
    return (
        gradient_x_full[
            ...,
            kernel_height - 1 : kernel_height - 1 + image_height,
            kernel_width - 1 : kernel_width - 1 + image_width,
        ].contiguous(),
        gradient_y_full[
            ...,
            kernel_height - 1 : kernel_height - 1 + image_height,
            kernel_width - 1 : kernel_width - 1 + image_width,
        ].contiguous(),
    )


def linear_convolution_1d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Discrete linear convolution of two 1-D tensors (full output)."""
    if signal.ndim != 1 or kernel.ndim != 1:
        raise ValueError("linear_convolution_1d requires 1-D inputs")
    convolved = F.conv1d(
        signal.view(1, 1, -1),
        kernel.flip(0).view(1, 1, -1),
        padding=kernel.shape[0] - 1,
    )
    return convolved.view(-1)
