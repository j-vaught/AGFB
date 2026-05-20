"""Central finite-difference baseline (no smoothing).

The 3-tap kernel `[-1, 0, 1] / 2` applied along the column axis gives
`gradient_x`. Applied along the row axis it gives `gradient_y`.
"""

from __future__ import annotations

import torch

from agfb_filters.base import check_input, separable_gradient

_SMOOTH_KERNEL = torch.tensor([1.0])  # identity along the smoothing axis
_DERIVATIVE_KERNEL = torch.tensor([-1.0, 0.0, 1.0]) / 2.0


def central_difference(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    image = check_input(image)
    smooth_kernel = _SMOOTH_KERNEL.to(device=image.device, dtype=image.dtype)
    derivative_kernel = _DERIVATIVE_KERNEL.to(device=image.device, dtype=image.dtype)
    return separable_gradient(
        image,
        smooth_kernel_1d=smooth_kernel,
        derivative_kernel_1d=derivative_kernel,
    )
