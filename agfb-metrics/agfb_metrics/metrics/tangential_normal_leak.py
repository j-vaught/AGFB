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

import math

import torch

from agfb_metrics.metrics.base import check_grad_pair, unit_normal_from_truth


def tangential_normal_leak(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    eps: float = 1e-30,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask.shape != g_x.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x.shape}")

    n_x, n_y = unit_normal_from_truth(g_x_t, g_y_t)
    t_x, t_y = -n_y, n_x

    g_n = g_x * n_x + g_y * n_y
    g_t = g_x * t_x + g_y * t_y

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i in range(B):
        m = signal_mask[i]
        if not bool(m.any()):
            out[i] = float("nan")
            continue
        e_n = float((g_n[i][m] ** 2).mean())
        e_t = float((g_t[i][m] ** 2).mean())
        if e_n < eps:
            out[i] = float("-inf") if e_t < eps else float("inf")
            continue
        if e_t < eps:
            out[i] = float("-inf")
            continue
        out[i] = 10.0 * math.log10(e_t / e_n)
    return out
