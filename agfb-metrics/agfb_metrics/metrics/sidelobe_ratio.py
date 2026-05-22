"""Side-lobe ratio (dB).

Designed for sampled cross-signal profiles where a dominant main lobe can be
separated from smaller secondary lobes.

On the cross-signal `|grad_filter|` profile for each signal pixel, identify the
main lobe as the contiguous region around the peak that extends outward in
each direction until the profile first reaches a local minimum. Outside
the main lobe, compute the maximum value `|grad|_outside`, and form

    r_p = |grad|_outside / |grad|_peak

The metric is `20/|E| * sum_p log10(r_p)`, in dB (the spec form). Pixels
with no detectable side-lobe - i.e., the main lobe spans the whole
`[-r_max, r_max]` window - are skipped, since the spec is silent on what
their `r_p` should be. Images with no usable signal pixels return NaN.
"""

from __future__ import annotations

import torch

from agfb_metrics.metrics._cross_edge_profile import cross_edge_profile
from agfb_metrics.metrics.base import check_grad_pair, magnitude


def _sidelobe_ratios(prof: torch.Tensor) -> torch.Tensor:
    """Vectorized side-lobe ratios for `(N, K)` profiles; unusable rows yield NaN."""
    N, K = prof.shape
    if N == 0:
        return torch.empty((0,), dtype=torch.float32, device=prof.device)

    idx = torch.arange(K, device=prof.device)
    peak, k_peak = torch.max(prof, dim=1)

    left_idx = idx[1:].unsqueeze(0)
    left_violation = prof[:, :-1] > prof[:, 1:]
    left_violation = left_violation & (left_idx <= k_peak.unsqueeze(1))
    left_bound = torch.where(left_violation, left_idx, torch.zeros_like(left_idx)).max(dim=1).values

    right_idx = idx[:-1].unsqueeze(0)
    right_violation = prof[:, 1:] > prof[:, :-1]
    right_violation = right_violation & (right_idx >= k_peak.unsqueeze(1))
    right_bound = torch.where(right_violation, right_idx, torch.full_like(right_idx, K - 1))
    right_bound = right_bound.min(dim=1).values

    outside = (idx.unsqueeze(0) < left_bound.unsqueeze(1)) | (
        idx.unsqueeze(0) > right_bound.unsqueeze(1)
    )
    outside_values = torch.where(outside, prof, torch.full_like(prof, -torch.inf))
    outside_max = outside_values.max(dim=1).values
    ratio = outside_max / peak.clamp_min(1e-30)
    valid = (peak > 0.0) & outside.any(dim=1) & (outside_max > 0.0)
    return torch.where(valid, ratio, torch.full_like(ratio, float("nan")))


def sidelobe_ratio(
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
    profiles, _, _ = cross_edge_profile(
        magnitude(g_x, g_y), g_x_t, g_y_t, signal_mask, r_max=r_max, step=step
    )

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i, prof in enumerate(profiles):
        if prof.shape[0] == 0:
            out[i] = float("nan")
            continue
        ratios = _sidelobe_ratios(prof)
        valid = ~torch.isnan(ratios)
        log_ratios = torch.where(valid, torch.log10(ratios), torch.zeros_like(ratios))
        count = valid.sum()
        out[i] = torch.where(
            count > 0,
            20.0 * log_ratios.sum() / count.clamp_min(1),
            torch.tensor(float("nan"), device=prof.device),
        )
    return out
