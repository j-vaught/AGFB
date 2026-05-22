"""Side-lobe ratio (dB).

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

import math

import torch

from agfb_metrics.metrics._cross_edge_profile import cross_edge_profile
from agfb_metrics.metrics.base import check_grad_pair, magnitude


def _main_lobe_bounds(prof: torch.Tensor, k_peak: int) -> tuple[int, int]:
    """Return (k_left, k_right) inclusive bounds of the main lobe, defined as
    the contiguous region around `k_peak` extending outward until the profile
    first reaches a local minimum (i.e. starts rising again)."""
    K = prof.shape[0]
    k_left = k_peak
    while k_left > 0 and prof[k_left - 1].item() <= prof[k_left].item():
        k_left -= 1
    k_right = k_peak
    while k_right < K - 1 and prof[k_right + 1].item() <= prof[k_right].item():
        k_right += 1
    return k_left, k_right


def _sidelobe_ratio_one_profile(prof: torch.Tensor) -> float | None:
    K = prof.shape[0]
    k_peak = int(torch.argmax(prof).item())
    peak = float(prof[k_peak].item())
    if peak <= 0.0:
        return None
    k_left, k_right = _main_lobe_bounds(prof, k_peak)
    if k_left == 0 and k_right == K - 1:
        return None  # main lobe fills the window; no side-lobe to measure
    outside_max = 0.0
    if k_left > 0:
        outside_max = max(outside_max, float(prof[:k_left].max().item()))
    if k_right < K - 1:
        outside_max = max(outside_max, float(prof[k_right + 1 :].max().item()))
    if outside_max <= 0.0:
        return None  # nothing outside the main lobe; r=0 -> -inf, skip
    return outside_max / peak


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
        ratios: list[float] = []
        for n in range(prof.shape[0]):
            r = _sidelobe_ratio_one_profile(prof[n])
            if r is not None:
                ratios.append(r)
        if not ratios:
            out[i] = float("nan")
            continue
        out[i] = float(20.0 * sum(math.log10(r) for r in ratios) / len(ratios))
    return out
