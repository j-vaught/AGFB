"""Angular MAE on signal pixels (degrees).

Designed for signal pixels where filter and truth gradients have meaningful
nonzero directions. Degenerate zero-magnitude pixels are skipped.

At each `p in E`,
    theta_p = arccos((grad_filter . grad_true) / (|grad_filter| |grad_true|))

reported as the mean of `theta_p` over `E`, in degrees. The arccos argument
is clamped to exactly `[-1, 1]` so float32 round-off (e.g. 1.0000001 for
perfectly aligned vectors) doesn't drift outside the domain. The endpoints
themselves are well-defined: arccos(1) = 0, arccos(-1) = pi.

Signal pixels at which either gradient has zero magnitude are skipped (the
angle is undefined there). If every signal pixel in an image is degenerate
the metric returns NaN for that image.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import check_grad_pair, magnitude, masked_mean_per_image


def angular_mae(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask.shape != g_x.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x.shape}")

    dot = g_x * g_x_t + g_y * g_y_t
    mag_f = magnitude(g_x, g_y)
    mag_t = magnitude(g_x_t, g_y_t)
    denom = (mag_f * mag_t).clamp_min(eps)
    cos_theta = (dot / denom).clamp(-1.0, 1.0)
    theta_deg = torch.arccos(cos_theta) * (180.0 / torch.pi)
    valid = signal_mask & (mag_f > eps) & (mag_t > eps)
    return masked_mean_per_image(theta_deg, valid)
