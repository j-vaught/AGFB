"""Edge FWHM (response sharpness, pixels).

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


def _fwhm_one_profile(prof: torch.Tensor, t: torch.Tensor) -> float | None:
    K = prof.shape[0]
    k_peak = int(torch.argmax(prof).item())
    peak = float(prof[k_peak].item())
    if peak <= 0.0:
        return None
    half = 0.5 * peak

    # Walk left from peak until we cross below half.
    left = None
    for k in range(k_peak, 0, -1):
        if prof[k - 1].item() < half:
            p_hi = prof[k].item()
            p_lo = prof[k - 1].item()
            frac = (p_hi - half) / max(p_hi - p_lo, 1e-30)
            left = float(t[k].item() - frac * (t[k].item() - t[k - 1].item()))
            break
    if left is None:
        return None

    right = None
    for k in range(k_peak, K - 1):
        if prof[k + 1].item() < half:
            p_hi = prof[k].item()
            p_lo = prof[k + 1].item()
            frac = (p_hi - half) / max(p_hi - p_lo, 1e-30)
            right = float(t[k].item() + frac * (t[k + 1].item() - t[k].item()))
            break
    if right is None:
        return None
    return right - left


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
        widths: list[float] = []
        for n in range(prof.shape[0]):
            w = _fwhm_one_profile(prof[n], t)
            if w is not None:
                widths.append(w)
        if not widths:
            out[i] = float("nan")
            continue
        out[i] = float(sum(widths) / len(widths))
    return out
