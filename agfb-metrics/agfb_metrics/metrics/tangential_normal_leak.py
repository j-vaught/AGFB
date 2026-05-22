"""Tangential-to-normal leak (dB).

Designed for oriented-gradient signal regions with reliable true-gradient
normals. The signal mask should exclude flat or near-zero truth gradients.

At each `p in E` with true unit normal `n_hat_p` and tangent `t_hat_p`:
    g_n(p) = grad_filter(p) . n_hat_p
    g_t(p) = grad_filter(p) . t_hat_p
    E_n = <g_n^2>_E,  E_t = <g_t^2>_E
    leak ratio = 10 log10(E_t / E_n)

Square-support filters at oblique edges leak energy into the tangential
direction; circular-support reference filters in the AGFB suite do not.
More-negative is better. Tangent is the 90 deg CCW rotation of the normal:
`(t_x, t_y) = (-n_y, n_x)`.

When `E_n` is zero (filter output identically zero on every signal pixel) the
metric is returned as `-inf` since `E_t` is also zero in that case - the
filter has no energy at all and the leak ratio is degenerate. The `B,`
output dtype is float32, so the actual value is `-3.4028e+38`; the sweep
aggregator should treat -inf and NaN as failure flags.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import (
    check_grad_pair,
    masked_count_per_image,
    masked_sum_per_image,
    unit_normal_from_truth,
)


def tangential_normal_leak(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor | None,
    *,
    eps: float = 1e-30,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask is not None and signal_mask.shape != g_x.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x.shape}")

    n_x, n_y = unit_normal_from_truth(g_x_t, g_y_t)
    t_x, t_y = -n_y, n_x

    g_n = g_x * n_x + g_y * n_y
    g_t = g_x * t_x + g_y * t_y

    count = masked_count_per_image(signal_mask, g_n)
    e_n = masked_sum_per_image(g_n * g_n, signal_mask) / count.clamp_min(1.0)
    e_t = masked_sum_per_image(g_t * g_t, signal_mask) / count.clamp_min(1.0)
    finite = 10.0 * torch.log10(e_t / e_n)
    out = torch.where(e_n < eps, torch.where(e_t < eps, -torch.inf, torch.inf), finite)
    out = torch.where((e_n >= eps) & (e_t < eps), -torch.inf, out)
    return torch.where(count > 0, out, torch.full_like(out, float("nan")))
