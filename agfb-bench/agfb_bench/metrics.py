"""Metric evaluation — Chapter 4 of BENCHMARK_DESIGN.md.

The ten metrics split into a cheap **pixel set** (7, collected on every run) and
an expensive **profile set** (3, collected only on the clean pass; spec 4.5).
Region masks come from ``masks(gx_t, gy_t, dilate_px, rel_eps)`` and are computed
once per clean cell, then reused across all noise levels and filters (spec 4.1).
"""

from __future__ import annotations

import math

import torch

from agfb_bench.config import (
    ALL_METRICS,
    MASK_REL_EPS,
    PIXEL_METRICS,
    PROFILE_METRICS,
    PROFILE_R_MAX,
    PROFILE_STEP,
    mask_dilate_px,
)

# noise_gain / tail_spurious_grad need the flat mask; the profile set needs the
# signal mask. nrmse and the rest need the signal mask.
_FLAT_METRICS = frozenset({"noise_gain", "tail_spurious_grad"})


def build_masks(gx_t: torch.Tensor, gy_t: torch.Tensor, image_size: int) -> dict[str, torch.Tensor]:
    """Signal / flat region masks for a clean cell (reused across noise+filters)."""
    import agfb_metrics

    return agfb_metrics.masks(
        gx_t, gy_t, dilate_px=mask_dilate_px(image_size), rel_eps=MASK_REL_EPS
    )


def metric_names(profile: str) -> tuple[str, ...]:
    """Resolve a metric profile to its ordered metric tuple."""
    if profile == "pixel":
        return PIXEL_METRICS
    if profile == "profile":
        return PROFILE_METRICS
    if profile == "all":
        return ALL_METRICS
    raise ValueError(f"unknown metric profile {profile!r}")


def evaluate(
    gx: torch.Tensor,
    gy: torch.Tensor,
    gx_t: torch.Tensor,
    gy_t: torch.Tensor,
    *,
    names: tuple[str, ...],
    signal_mask: torch.Tensor,
    flat_mask: torch.Tensor,
    sigma_n: float,
) -> dict[str, float]:
    """Evaluate the requested metrics for a single (batch-1) field.

    ``noise_gain`` is undefined without injected noise; on the clean pass
    (``sigma_n <= 0``) it is reported as NaN rather than dividing by zero.
    """
    import agfb_metrics

    requested = list(names)
    drop_noise_gain = "noise_gain" in requested and sigma_n <= 0.0
    to_compute = [m for m in requested if not (m == "noise_gain" and drop_noise_gain)]

    scores: dict[str, float] = {}
    if to_compute:
        raw = agfb_metrics.evaluate_metrics(
            gx,
            gy,
            gx_t,
            gy_t,
            metrics=tuple(to_compute),
            signal_mask=signal_mask,
            flat_mask=flat_mask,
            sigma_n=(sigma_n if sigma_n > 0 else None),
            r_max=PROFILE_R_MAX,
            step=PROFILE_STEP,
        )
        for name, value in raw.items():
            scores[name] = float(value.detach().reshape(-1)[0].item())

    if drop_noise_gain:
        scores["noise_gain"] = math.nan
    return {name: scores.get(name, math.nan) for name in requested}
