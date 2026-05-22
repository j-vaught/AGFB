"""Magnitude bias on signal pixels.

Designed for signal masks where the average true-gradient magnitude is a
meaningful reference scale.

`magnitude_bias = <|grad_filter|>_E / <|grad_true|>_E - 1`

Signed: positive means the filter is over-reading the signal magnitude
(sharpeners), negative means under-reading (smoothers spread the step's
gradient across many pixels and reduce peak height).
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import (
    check_grad_pair,
    magnitude,
    masked_count_per_image,
    masked_sum_per_image,
)


def magnitude_bias(
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

    count = masked_count_per_image(signal_mask)
    num = masked_sum_per_image(mag_f, signal_mask)
    den = masked_sum_per_image(mag_t, signal_mask).clamp_min(1e-30)
    out = num / den - 1.0
    return torch.where(count > 0, out, torch.full_like(out, float("nan")))
