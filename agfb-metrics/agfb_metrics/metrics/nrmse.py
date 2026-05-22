"""NRMSE on signal pixels.

Designed for AGFB-style gradient benchmarks with known true gradients and a
non-empty signal mask. Inputs are gradient tensors, not raw intensity images.

`NRMSE = sqrt(<|e|^2>_E) / <|grad_true|>_E`

where `e = grad_filter - grad_true`, `E` is the signal (true-gradient signal) mask, and
both expectations are means over `E`. Note this is the *spec* definition: the
numerator is RMS, the denominator is the linear mean of true-gradient
magnitude. (The existing PGF_paper prototype `mini.nrmse_vector` uses a
sqrt-of-mean-square denominator; that form is preserved in the prototype
itself for backward compatibility but is *not* what the Section 1.3 spec asks for.)

Returns one value per image: shape `(B,)`, float32, NaN if the signal mask
is empty for that image.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import (
    check_grad_pair,
    magnitude,
    masked_count_per_image,
    masked_mean_per_image,
    masked_sum_per_image,
)


def nrmse(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor | None,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask is not None and signal_mask.shape != g_x.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x.shape}")

    err_sq = (g_x - g_x_t) ** 2 + (g_y - g_y_t) ** 2
    mag_true = magnitude(g_x_t, g_y_t)

    count = masked_count_per_image(signal_mask, err_sq)
    num = torch.sqrt(masked_sum_per_image(err_sq, signal_mask) / count.clamp_min(1.0))
    den = masked_mean_per_image(mag_true, signal_mask).clamp_min(1e-30)
    out = num / den
    return torch.where(count > 0, out, torch.full_like(out, float("nan")))
