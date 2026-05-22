"""Metric A.3 -- 95th-percentile gradient-vector error on edge pixels.

For each `p in E`, the scalar magnitude `|e(p)|` of the per-pixel vector
error from A.1; report the 95th-percentile of `{|e(p)| : p in E}` per image.

Captures the rare-disaster pixels that A.1's mean hides. Downstream pipelines
(edge linking, ridge tracing) tend to break on the single worst pixel.
"""

from __future__ import annotations

import torch

from agfb_metrics.base import check_grad_pair


def a3_tail_vector_error(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    q: float = 0.95,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask.shape != g_x.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x.shape}")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")

    err_mag = torch.sqrt((g_x - g_x_t) ** 2 + (g_y - g_y_t) ** 2)

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i in range(B):
        m = signal_mask[i]
        if not bool(m.any()):
            out[i] = float("nan")
            continue
        out[i] = float(torch.quantile(err_mag[i][m], q))
    return out
