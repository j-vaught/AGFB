"""Scharr-3 (separable, replicate padding).

Optimized for rotational symmetry. The horizontal-gradient kernel uses
`[3, 10, 3] / 16` along the smoothing axis and `[-1, 0, 1] / 2` along the
differentiation axis. The combined 3x3 kernel is the standard Scharr operator.
"""

from __future__ import annotations

import torch

from agfb_filters.base import check_input, separable_gradient

_SMOOTH_KERNEL = torch.tensor([3.0, 10.0, 3.0]) / 16.0
_DERIVATIVE_KERNEL = torch.tensor([-1.0, 0.0, 1.0]) / 2.0


def scharr_3(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    image = check_input(image)
    smooth_kernel = _SMOOTH_KERNEL.to(device=image.device, dtype=image.dtype)
    derivative_kernel = _DERIVATIVE_KERNEL.to(device=image.device, dtype=image.dtype)
    return separable_gradient(
        image,
        smooth_kernel_1d=smooth_kernel,
        derivative_kernel_1d=derivative_kernel,
    )
