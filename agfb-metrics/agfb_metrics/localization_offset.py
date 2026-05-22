"""Localization offset (pixels).

For each true edge pixel `p` with unit normal `n_hat_p`, sample
`|grad_filter|` along the perpendicular line `p + t * n_hat_p` for
`t in [-r_max, r_max]`, locate the peak position to sub-pixel precision
via parabolic refinement, and report the mean absolute offset across edge
pixels.

Two modes are supported because the Section 1.3 spec is ambiguous about which
mask `E` to use:

* `mode="truth_anchored"` (default) -- uses the band signal mask. Because
  even `|grad_truth|` does not peak at `t=0` for off-ridge band pixels,
  the metric subtracts the truth's argmax position so the answer is 0
  when filter == truth:
      localization offset = <|t*_filter(p) - t*_truth(p)|>_signal_mask.

* `mode="ridge"` -- thins the signal mask to a 1-pixel-wide ridge by
  non-max-suppressing `|grad_truth|` along its own normal (see
  `ridge_mask_from_truth`). The literal spec form
      localization offset = <|t*_filter(p)|>_ridge_mask
  then makes sense because `t*_truth(p) = 0` on the ridge by construction.

Both modes rank filters identically on noiseless input; the truth-anchored
mode uses more pixels per image (a band has ~15x as many pixels as the
ridge it surrounds), so it produces a tighter estimate in the Section 1.3 sweep.

With `step=0.5` and r_max=16, parabolic refinement is good to ~0.05 px.
"""

from __future__ import annotations

from typing import Literal

import torch

from agfb_metrics._cross_edge_profile import cross_edge_profile
from agfb_metrics.base import check_grad_pair, magnitude, ridge_mask_from_truth


def _parabolic_subpixel_offset(
    p_left: torch.Tensor, p_mid: torch.Tensor, p_right: torch.Tensor
) -> torch.Tensor:
    denom = p_left - 2.0 * p_mid + p_right
    safe = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
    delta = 0.5 * (p_left - p_right) / safe
    delta = torch.where(denom.abs() > 1e-12, delta, torch.zeros_like(delta))
    return delta.clamp(-1.0, 1.0)


def _peak_positions(profiles: torch.Tensor, t: torch.Tensor, step: float) -> torch.Tensor:
    K = t.shape[0]
    argmax = torch.argmax(profiles, dim=1)
    argmax_c = argmax.clamp(1, K - 2)
    rows = torch.arange(profiles.shape[0], device=profiles.device)
    p_mid = profiles[rows, argmax_c]
    p_left = profiles[rows, argmax_c - 1]
    p_right = profiles[rows, argmax_c + 1]
    subpx = _parabolic_subpixel_offset(p_left, p_mid, p_right)
    return t[argmax_c] + subpx * step


def localization_offset(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    mode: Literal["truth_anchored", "ridge"] = "truth_anchored",
    r_max: float = 16.0,
    step: float = 0.5,
) -> torch.Tensor:
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if mode not in ("truth_anchored", "ridge"):
        raise ValueError(f"mode must be 'truth_anchored' or 'ridge'; got {mode!r}")

    mag_f = magnitude(g_x, g_y)
    mask = ridge_mask_from_truth(g_x_t, g_y_t, signal_mask) if mode == "ridge" else signal_mask
    filt_profiles, t, _ = cross_edge_profile(mag_f, g_x_t, g_y_t, mask, r_max=r_max, step=step)

    truth_profiles: list[torch.Tensor] | None = None
    if mode == "truth_anchored":
        mag_t = magnitude(g_x_t, g_y_t)
        truth_profiles, _, _ = cross_edge_profile(mag_t, g_x_t, g_y_t, mask, r_max=r_max, step=step)

    B = g_x.shape[0]
    out = torch.empty(B, dtype=torch.float32, device=g_x.device)
    for i in range(B):
        f_prof = filt_profiles[i]
        if f_prof.shape[0] == 0:
            out[i] = float("nan")
            continue
        t_filter = _peak_positions(f_prof, t, step)
        if truth_profiles is not None:
            t_truth = _peak_positions(truth_profiles[i], t, step)
            out[i] = float((t_filter - t_truth).abs().mean())
        else:
            out[i] = float(t_filter.abs().mean())
    return out
