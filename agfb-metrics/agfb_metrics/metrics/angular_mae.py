"""Angular MAE on edge pixels (degrees).

At each `p in E`,
    theta_p = arccos((grad_filter . grad_true) / (|grad_filter| |grad_true|))

reported as the mean of `theta_p` over `E`, in degrees. The arccos argument
is clamped to exactly `[-1, 1]` so float32 round-off (e.g. 1.0000001 for
perfectly aligned vectors) doesn't drift outside the domain. The endpoints
themselves are well-defined: arccos(1) = 0, arccos(-1) = pi.

Edge pixels at which either gradient has zero magnitude are skipped (the
angle is undefined there). If every edge pixel in an image is degenerate
the metric returns NaN for that image.
"""

from __future__ import annotations

import math

import torch

from agfb_metrics.metrics.base import check_grad_pair, magnitude


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
    theta_deg = torch.arccos(cos_theta) * (180.0 / math.pi)
    valid = signal_mask & (mag_f > eps) & (mag_t > eps)

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i in range(B):
        m = valid[i]
        if not bool(m.any()):
            out[i] = float("nan")
            continue
        out[i] = float(theta_deg[i][m].mean())
    return out
