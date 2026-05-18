"""Central finite-difference baseline (no smoothing).

The 3-tap kernel `[-1, 0, 1] / 2` applied along the column axis gives `g_x`;
applied along the row axis gives `g_y`. This is the second-order accurate
discrete approximation of `∂I/∂x` at the pixel center. It is the floor that
every other comparator filter in §1.3 must beat once noise is added; on clean
inputs it often wins on A1 NRMSE because any smoothing introduces pure bias.
"""

from __future__ import annotations

import torch

from cpgf_filters.base import check_input, separable_gradient

_SMOOTH = torch.tensor([1.0])  # identity along the smoothing axis
_DIFF = torch.tensor([-1.0, 0.0, 1.0]) / 2.0


def central_difference(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    I = check_input(I)
    s = _SMOOTH.to(device=I.device, dtype=I.dtype)
    d = _DIFF.to(device=I.device, dtype=I.dtype)
    return separable_gradient(I, smooth_1d=s, diff_1d=d)
