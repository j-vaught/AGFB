"""99th-percentile spurious gradient on flat regions.

Designed for flat-region masks where true gradients are zero and high-percentile
background gradient magnitude is the quantity of interest.

`tail_spurious_grad = P_99({|grad_filter(p)| : p in F})`

Independent of `sigma_n`. The value a downstream gradient cutoff must be set
above to suppress flat-region gradient artifacts. Tracks noise gain for
Gaussian-output filters and diverges when the noise output has heavy tails.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import check_grad_pair, magnitude, masked_quantile_per_image


def tail_spurious_grad(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    flat_mask: torch.Tensor,
    *,
    q: float = 0.99,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    if flat_mask.shape != g_x.shape:
        raise ValueError(f"flat_mask {flat_mask.shape} must match (B, H, W) {g_x.shape}")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")

    mag_f = magnitude(g_x, g_y)

    return masked_quantile_per_image(mag_f, flat_mask, q)
