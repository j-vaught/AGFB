"""Metric B.3 -- Magnitude bias on edge pixels.

`B_3 = <|grad_filter|>_E / <|grad_true|>_E - 1`

Signed: positive means the filter is over-reading the edge magnitude
(sharpeners), negative means under-reading (smoothers spread the step's
gradient across many pixels and reduce peak height).
"""

from __future__ import annotations

import torch

from agfb_metrics.base import check_grad_pair, magnitude


def b3_magnitude_bias(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask.shape != g_x.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x.shape}")

    mag_f = magnitude(g_x, g_y)
    mag_t = magnitude(g_x_t, g_y_t)

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i in range(B):
        m = signal_mask[i]
        if not bool(m.any()):
            out[i] = float("nan")
            continue
        num = mag_f[i][m].mean()
        den = mag_t[i][m].mean().clamp_min(1e-30)
        out[i] = float(num / den - 1.0)
    return out
