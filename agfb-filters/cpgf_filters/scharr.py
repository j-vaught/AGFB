"""Scharr-3 (separable, replicate padding).

Optimized for rotational symmetry: gx kernel is `[3, 10, 3] / 16` along the
smoothing axis and `[-1, 0, 1] / 2` along the differentiation axis. The
combined 3x3 kernel is the standard Scharr operator (sums of |c| = 32).
"""

from __future__ import annotations

import torch

from cpgf_filters.base import check_input, separable_gradient

_SMOOTH = torch.tensor([3.0, 10.0, 3.0]) / 16.0
_DIFF = torch.tensor([-1.0, 0.0, 1.0]) / 2.0


def scharr_3(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    I = check_input(I)
    smooth = _SMOOTH.to(device=I.device, dtype=I.dtype)
    diff = _DIFF.to(device=I.device, dtype=I.dtype)
    return separable_gradient(I, smooth_1d=smooth, diff_1d=diff)
