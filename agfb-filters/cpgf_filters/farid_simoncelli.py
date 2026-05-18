"""Farid–Simoncelli 5-tap matched-pair derivative.

From Farid & Simoncelli, "Differentiation of Discrete Multidimensional
Signals" (IEEE TIP 2004). The 5-tap interpolation `p` and derivative `d`
filters are jointly optimized so that `d(I) * p(I)^T` recovers the spatial
derivative with the least cross-orientation bias.
"""

from __future__ import annotations

import torch

from cpgf_filters.base import check_input, separable_gradient

_PREFILTER = torch.tensor([0.030320, 0.249724, 0.439911, 0.249724, 0.030320])
_DERIV = torch.tensor([-0.104550, -0.292315, 0.0, 0.292315, 0.104550])


def farid_simoncelli_5(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    I = check_input(I)
    p = _PREFILTER.to(device=I.device, dtype=I.dtype)
    d = _DERIV.to(device=I.device, dtype=I.dtype)
    return separable_gradient(I, smooth_1d=p, diff_1d=d)
