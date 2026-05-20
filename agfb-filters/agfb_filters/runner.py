"""Hardware-aware filter execution."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agfb_filters.base import check_input, fft_cross_correlation, separable_gradient
from agfb_filters.definitions import ExecutionStrategy, GradientFilterDefinition


def run_filter(
    definition: GradientFilterDefinition,
    image: torch.Tensor,
    *,
    strategy: ExecutionStrategy | str = ExecutionStrategy.AUTO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a filter definition against an image batch."""
    image = check_input(image)
    selected_strategy = _resolve_strategy(definition, image, ExecutionStrategy(strategy))

    if selected_strategy == ExecutionStrategy.SEPARABLE:
        smooth_kernel, derivative_kernel = definition.separable_kernels()
        return separable_gradient(
            image,
            smooth_kernel_1d=smooth_kernel.to(device=image.device, dtype=image.dtype),
            derivative_kernel_1d=derivative_kernel.to(device=image.device, dtype=image.dtype),
            pad_mode=definition.padding_mode,
        )

    kernel_x, kernel_y = definition.dense_kernels()
    dense_kernels = (
        kernel_x.to(device=image.device, dtype=image.dtype),
        kernel_y.to(device=image.device, dtype=image.dtype),
    )
    if selected_strategy == ExecutionStrategy.SPATIAL:
        return _spatial_cross_correlation(
            image,
            dense_kernels,
            padding_mode=definition.padding_mode,
            spatial_padding=definition.spatial_padding,
        )
    if selected_strategy == ExecutionStrategy.FFT:
        return fft_cross_correlation(image, dense_kernels, pad_mode=definition.padding_mode)
    raise ValueError(f"unsupported execution strategy {selected_strategy}")


def _resolve_strategy(
    definition: GradientFilterDefinition,
    image: torch.Tensor,
    requested_strategy: ExecutionStrategy,
) -> ExecutionStrategy:
    if requested_strategy != ExecutionStrategy.AUTO:
        return requested_strategy
    if definition.strategy_hint != ExecutionStrategy.AUTO:
        return definition.strategy_hint
    if definition.has_separable_kernels:
        return ExecutionStrategy.SEPARABLE
    if definition.spatial_padding is not None:
        return ExecutionStrategy.SPATIAL

    kernel_x, _ = definition.dense_kernels()
    kernel_height, kernel_width = kernel_x.shape
    kernel_size = max(kernel_height, kernel_width)
    image_area = image.shape[-2] * image.shape[-1]
    if image.device.type == "cpu":
        fft_threshold = 15 if image_area < 512 * 512 else 9
    else:
        fft_threshold = 9 if image_area < 512 * 512 else 7
    if kernel_size >= fft_threshold:
        return ExecutionStrategy.FFT
    return ExecutionStrategy.SPATIAL


def _spatial_cross_correlation(
    image: torch.Tensor,
    kernels: tuple[torch.Tensor, torch.Tensor],
    *,
    padding_mode: str,
    spatial_padding: tuple[int, int, int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x, kernel_y = kernels
    if kernel_x.shape != kernel_y.shape:
        raise ValueError(f"kernel shapes must match, got {kernel_x.shape} vs {kernel_y.shape}")

    kernel_height, kernel_width = kernel_x.shape
    if spatial_padding is None:
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("even-sized kernels require explicit spatial padding")
        padding_height = kernel_height // 2
        padding_width = kernel_width // 2
        spatial_padding = (padding_width, padding_width, padding_height, padding_height)

    image_channels = image.unsqueeze(1)
    padded_image = F.pad(image_channels, spatial_padding, mode=padding_mode)
    kernel_stack = torch.stack((kernel_x, kernel_y), dim=0).unsqueeze(1)
    gradients = F.conv2d(padded_image, kernel_stack)
    return gradients[:, 0].contiguous(), gradients[:, 1].contiguous()
