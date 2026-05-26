"""Shared tensor conventions and helpers for AGFB filters.

Input convention.
    `image` is a `(batch, height, width)` floating-point tensor on any device.

Output convention.
    Every filter returns `(gradient_x, gradient_y)` tensors with shape
    `(batch, height, width)`.
    `gradient_x` is the horizontal gradient along image columns.
    `gradient_y` is the vertical gradient along image rows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode


def check_input(image: torch.Tensor) -> torch.Tensor:
    """Validate and contiguous-ify a filter input."""
    if image.ndim != 3:
        raise ValueError(
            f"filter input must be (batch, height, width), got shape {tuple(image.shape)}"
        )
    if not image.dtype.is_floating_point:
        raise ValueError(f"filter input must use a floating-point dtype, got {image.dtype}")
    return image.contiguous()


def pad_with_boundary(
    tensor: torch.Tensor,
    padding: tuple[int, int, int, int],
    boundary: BoundaryCondition,
) -> torch.Tensor:
    """Pad a 4-D image tensor using the runner boundary contract."""
    if tensor.ndim != 4:
        raise ValueError(f"boundary padding expects a 4-D tensor, got {tensor.ndim} dimensions")
    if len(padding) != 4:
        raise ValueError("padding must be (left, right, top, bottom)")
    if any(amount < 0 for amount in padding):
        raise ValueError(f"padding amounts must be nonnegative, got {padding}")

    boundary = _require_boundary_condition(boundary)
    if boundary.mode == BoundaryMode.REFLECT:
        _require_reflect_padding_supported(tensor, padding)
    if boundary.mode == BoundaryMode.CONSTANT:
        return F.pad(tensor, padding, mode=boundary.mode.value, value=boundary.value)
    return F.pad(tensor, padding, mode=boundary.mode.value)


def directional_derivative(
    image: torch.Tensor,
    *,
    smooth_kernel_1d: torch.Tensor,
    derivative_kernel_1d: torch.Tensor,
    axis: int,
    boundary: BoundaryCondition,
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
        image_channels = pad_with_boundary(
            image_channels,
            (derivative_radius, derivative_radius, smooth_radius, smooth_radius),
            boundary,
        )
    else:
        # smooth along cols (x), differentiate along rows (y)
        smooth_kernel = smooth_kernel_1d.view(1, 1, 1, -1)
        derivative_kernel = derivative_kernel_1d.view(1, 1, -1, 1)
        image_channels = pad_with_boundary(
            image_channels,
            (smooth_radius, smooth_radius, derivative_radius, derivative_radius),
            boundary,
        )
    smoothed_image = F.conv2d(image_channels, smooth_kernel)
    return F.conv2d(smoothed_image, derivative_kernel).squeeze(1)


def separable_gradient(
    image: torch.Tensor,
    *,
    smooth_kernel_1d: torch.Tensor,
    derivative_kernel_1d: torch.Tensor,
    boundary: BoundaryCondition,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(gradient_x, gradient_y)` with smooth-first ordering."""
    gradient_x = directional_derivative(
        image,
        smooth_kernel_1d=smooth_kernel_1d,
        derivative_kernel_1d=derivative_kernel_1d,
        axis=1,
        boundary=boundary,
    )
    gradient_y = directional_derivative(
        image,
        smooth_kernel_1d=smooth_kernel_1d,
        derivative_kernel_1d=derivative_kernel_1d,
        axis=0,
        boundary=boundary,
    )
    return gradient_x, gradient_y


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


def _require_boundary_condition(boundary: BoundaryCondition) -> BoundaryCondition:
    if not isinstance(boundary, BoundaryCondition):
        raise ValueError("boundary must be a BoundaryCondition")
    return boundary


def _require_reflect_padding_supported(
    tensor: torch.Tensor,
    padding: tuple[int, int, int, int],
) -> None:
    left, right, top, bottom = padding
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    if left >= width or right >= width or top >= height or bottom >= height:
        raise ValueError(
            "reflect boundary requires left/right padding to be smaller than input width "
            "and top/bottom padding to be smaller than input height, "
            f"got padding {padding} for input {height}x{width}"
        )
