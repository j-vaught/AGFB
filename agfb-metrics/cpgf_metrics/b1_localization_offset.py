"""Metric B.1 -- Localization offset (pixels).

For each true edge pixel `p` with unit normal `n̂_p`, sample
`|grad_filter|` and `|grad_truth|` along the perpendicular line
`p + t * n̂_p` for `t in [-r_max, r_max]`. Locate each profile's peak
position to sub-pixel precision via parabolic refinement around the
discretely-sampled argmax, then report

    B_1 = <|t*_filter(p) - t*_truth(p)|>_E    pixels.

Subtracting the truth peak position is required because the signal mask is
a band (§1.1) rather than a thin ridge — for an off-ridge edge pixel, even
the truth's |grad| profile has its peak away from t=0. B.1 isolates the
filter-induced localization shift relative to where the truth field itself
peaks along the normal.

With `step=0.5` and r_max=16, the parabolic refinement is good to ~0.05 px.
"""

from __future__ import annotations

import torch

from cpgf_metrics._cross_edge_profile import cross_edge_profile
from cpgf_metrics.base import check_grad_pair, magnitude


def _parabolic_subpixel_offset(
    p_left: torch.Tensor, p_mid: torch.Tensor, p_right: torch.Tensor
) -> torch.Tensor:
    """Sub-sample offset (in units of `step`) of the true peak relative to
    `p_mid`, via parabolic fit through three consecutive samples. Returns
    values in `[-1, 1]`; clamps degenerate (flat / non-concave) triples to 0.
    """
    denom = p_left - 2.0 * p_mid + p_right
    safe = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
    delta = 0.5 * (p_left - p_right) / safe
    delta = torch.where(denom.abs() > 1e-12, delta, torch.zeros_like(delta))
    return delta.clamp(-1.0, 1.0)


def _peak_positions(profiles: torch.Tensor, t: torch.Tensor, step: float) -> torch.Tensor:
    """Sub-pixel peak positions along the `t` axis for each profile row."""
    K = t.shape[0]
    argmax = torch.argmax(profiles, dim=1)
    argmax_c = argmax.clamp(1, K - 2)
    rows = torch.arange(profiles.shape[0], device=profiles.device)
    p_mid = profiles[rows, argmax_c]
    p_left = profiles[rows, argmax_c - 1]
    p_right = profiles[rows, argmax_c + 1]
    subpx = _parabolic_subpixel_offset(p_left, p_mid, p_right)
    return t[argmax_c] + subpx * step


def b1_localization_offset(
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

    mag_f = magnitude(g_x, g_y)
    mag_t = magnitude(g_x_t, g_y_t)
    filt_profiles, t, _ = cross_edge_profile(
        mag_f, g_x_t, g_y_t, signal_mask, r_max=r_max, step=step
    )
    true_profiles, _, _ = cross_edge_profile(
        mag_t, g_x_t, g_y_t, signal_mask, r_max=r_max, step=step
    )

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i in range(B):
        f_prof = filt_profiles[i]
        t_prof = true_profiles[i]
        if f_prof.shape[0] == 0:
            out[i] = float("nan")
            continue
        t_filter = _peak_positions(f_prof, t, step)
        t_truth = _peak_positions(t_prof, t, step)
        out[i] = float((t_filter - t_truth).abs().mean())
    return out
