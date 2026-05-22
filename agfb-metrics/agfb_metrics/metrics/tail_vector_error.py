"""95th-percentile gradient-vector error on signal pixels.

Designed for AGFB-style signal masks where rare large gradient errors matter.
Inputs are matched filter and truth gradient tensors.

For each `p in E`, the scalar magnitude `|e(p)|` of the per-pixel vector
error from NRMSE; report the 95th-percentile of `{|e(p)| : p in E}` per image.

Captures the rare-disaster pixels that NRMSE's mean hides. Downstream gradient
pipelines tend to break on the single worst pixel.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import check_grad_pair, masked_quantile_per_image


def tail_vector_error(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor | None,
    *,
    q: float = 0.95,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask is not None and signal_mask.shape != g_x.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x.shape}")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")

    err_mag = torch.sqrt((g_x - g_x_t) ** 2 + (g_y - g_y_t) ** 2)

    return masked_quantile_per_image(err_mag, signal_mask, q)
