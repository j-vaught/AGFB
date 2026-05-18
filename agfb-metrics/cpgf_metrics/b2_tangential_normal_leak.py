"""Metric B.2 -- Tangential-to-normal leak (dB).

At each `p in E` with true unit normal `n̂_p` and tangent `t̂_p`:
    g_n(p) = grad_filter(p) . n̂_p
    g_t(p) = grad_filter(p) . t̂_p
    E_n = <g_n^2>_E,  E_t = <g_t^2>_E
    B_2 = 10 log10(E_t / E_n)

Square-support filters at oblique edges leak energy into the tangential
direction; circular-support filters (CPGF, DoG, FreemanAdelson) do not.
More-negative is better. Tangent is the 90 deg CCW rotation of the normal:
`(t_x, t_y) = (-n_y, n_x)`.

When `E_n` is zero (filter output identically zero on every edge pixel) the
metric is returned as `-inf` since `E_t` is also zero in that case — the
filter has no energy at all and the leak ratio is degenerate. The `B,`
output dtype is float32, so the actual value is `-3.4028e+38`; the sweep
aggregator should treat -inf and NaN as failure flags.
"""

from __future__ import annotations

import math

import torch

from cpgf_metrics.base import check_grad_pair, unit_normal_from_truth


def b2_tangential_normal_leak(
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
        out[i] = 10.0 * math.log10(e_t / e_n)
    return out
