"""Farid-Simoncelli 5-tap matched-pair derivative.

From Farid & Simoncelli, "Differentiation of Discrete Multidimensional
Signals" (IEEE TIP 2004). The 5-tap interpolation and derivative filters are
jointly optimized to recover spatial derivatives with low cross-orientation
bias.
"""

from __future__ import annotations

import torch

from agfb_filters.base import check_input, separable_gradient

_PREFILTER = torch.tensor([0.030320, 0.249724, 0.439911, 0.249724, 0.030320])
_DERIVATIVE_KERNEL = torch.tensor([-0.104550, -0.292315, 0.0, 0.292315, 0.104550])


def farid_simoncelli_5(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    image = check_input(image)
    prefilter_kernel = _PREFILTER.to(device=image.device, dtype=image.dtype)
    derivative_kernel = _DERIVATIVE_KERNEL.to(device=image.device, dtype=image.dtype)
    return separable_gradient(
        image,
        smooth_kernel_1d=prefilter_kernel,
        derivative_kernel_1d=derivative_kernel,
    )
