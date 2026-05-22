"""Background noise gain.

Designed for flat-region gradient outputs from images with known injected AWGN
standard deviation `sigma_n`.

`noise_gain = std(|grad_filter|)_F / sigma_n`

where `F` is the flat-region mask and `sigma_n` is the known input AWGN
standard deviation. Smoothers score below 1.0, raw derivatives above. This
is the direct empirical match to the analytic noise-gain expression in the
theory section.

`sigma_n` is required and must be > 0. The benchmark always knows the
injected noise std a priori, so silent fallback would only hide bugs.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import check_grad_pair, magnitude


def noise_gain(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    flat_mask: torch.Tensor,
    sigma_n: float,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    if flat_mask.shape != g_x.shape:
        raise ValueError(f"flat_mask {flat_mask.shape} must match (B, H, W) {g_x.shape}")
    if not sigma_n > 0:
        raise ValueError(f"sigma_n must be positive; got {sigma_n}")

    mag_f = magnitude(g_x, g_y)

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i in range(B):
        m = flat_mask[i]
        if int(m.sum()) < 2:
            out[i] = float("nan")
            continue
        out[i] = float(torch.std(mag_f[i][m], unbiased=False) / sigma_n)
    return out
