"""Edge FWHM (response sharpness, pixels).

Designed for smooth, single-lobe cross-signal gradient profiles. Profiles that
do not cross half-peak on both sides are skipped.

On the cross-signal `|grad_filter|` profile for each signal pixel, find the
peak value and the two interpolated positions on either side of the peak
where the profile first drops to half the peak height. The FWHM is the
distance between those positions; the metric is the mean FWHM across signal
pixels. Sub-pixel crossings use linear interpolation between adjacent
samples.

Signal pixels whose profile never reaches half-peak on one or both sides
within `[-r_max, r_max]` are skipped - the spec doesn't define FWHM in
that case. Images with no usable signal pixels return NaN.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics._cross_edge_profile import cross_edge_profile
from agfb_metrics.metrics.base import check_grad_pair, magnitude


def _fwhm_profiles(prof: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Vectorized FWHM for `(N, K)` profiles; unusable profiles yield NaN."""
    N, K = prof.shape
    if N == 0:
        return torch.empty((0,), dtype=torch.float32, device=prof.device)

    rows = torch.arange(N, device=prof.device)
    idx = torch.arange(K, device=prof.device)
    peak, k_peak = torch.max(prof, dim=1)
    half = 0.5 * peak

    before_peak = idx.unsqueeze(0) < k_peak.unsqueeze(1)
    below_half = prof < half.unsqueeze(1)

    left_candidates = before_peak & below_half
    has_left = left_candidates.any(dim=1)
    left_lo = torch.where(left_candidates, idx.unsqueeze(0), -torch.ones_like(idx).unsqueeze(0))
    left_lo = left_lo.max(dim=1).values
    left_hi = (left_lo + 1).clamp_max(K - 1)

    after_peak = idx.unsqueeze(0) > k_peak.unsqueeze(1)
    right_candidates = after_peak & below_half
    has_right = right_candidates.any(dim=1)
    right_lo = torch.where(right_candidates, idx.unsqueeze(0), torch.full_like(idx, K).unsqueeze(0))
    right_lo = right_lo.min(dim=1).values
    right_hi = (right_lo - 1).clamp_min(0)

    left_p_hi = prof[rows, left_hi]
    left_p_lo = prof[rows, left_lo.clamp_min(0)]
    left_den = (left_p_hi - left_p_lo).clamp_min(1e-30)
    left_frac = (left_p_hi - half) / left_den
    left = t[left_hi] - left_frac * (t[left_hi] - t[left_lo.clamp_min(0)])

    right_p_hi = prof[rows, right_hi]
    right_p_lo = prof[rows, right_lo.clamp_max(K - 1)]
    right_den = (right_p_hi - right_p_lo).clamp_min(1e-30)
    right_frac = (right_p_hi - half) / right_den
    right = t[right_hi] + right_frac * (t[right_lo.clamp_max(K - 1)] - t[right_hi])

    widths = right - left
    valid = (peak > 0.0) & has_left & has_right
    return torch.where(valid, widths, torch.full_like(widths, float("nan")))


def edge_fwhm(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    r_max: float = 16.0,
    step: float = 0.5,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    profiles, t, _ = cross_edge_profile(
        magnitude(g_x, g_y), g_x_t, g_y_t, signal_mask, r_max=r_max, step=step
    )

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i, prof in enumerate(profiles):
        if prof.shape[0] == 0:
            out[i] = float("nan")
            continue
        widths = _fwhm_profiles(prof, t)
        valid = ~torch.isnan(widths)
        count = valid.sum()
        total = torch.where(valid, widths, torch.zeros_like(widths)).sum()
        out[i] = torch.where(
            count > 0, total / count.clamp_min(1), torch.tensor(float("nan"), device=prof.device)
        )
    return out
