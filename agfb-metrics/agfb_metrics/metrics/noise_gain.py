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

from agfb_metrics.metrics.base import check_grad_pair, magnitude, masked_std_per_image


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

    return masked_std_per_image(mag_f, flat_mask, min_count=2) / sigma_n
