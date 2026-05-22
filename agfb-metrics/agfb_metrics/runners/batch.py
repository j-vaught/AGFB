"""Batch runner for the standard AGFB metric set."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from agfb_metrics.metrics import (
    angular_mae,
    edge_fwhm,
    localization_offset,
    magnitude_bias,
    masks,
    noise_gain,
    nrmse,
    sidelobe_ratio,
    tail_spurious_grad,
    tail_vector_error,
    tangential_normal_leak,
)

MetricInput = Literal["edge", "flat", "flat_with_sigma"]


@dataclass(frozen=True)
class MetricSpec:
    """Describe how a metric function is called by the batch runner."""

    name: str
    fn: Callable[..., torch.Tensor]
    input_kind: MetricInput


DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("nrmse", nrmse, "edge"),
    MetricSpec("angular_mae", angular_mae, "edge"),
    MetricSpec("tail_vector_error", tail_vector_error, "edge"),
    MetricSpec("localization_offset", localization_offset, "edge"),
    MetricSpec("tangential_normal_leak", tangential_normal_leak, "edge"),
    MetricSpec("magnitude_bias", magnitude_bias, "edge"),
    MetricSpec("edge_fwhm", edge_fwhm, "edge"),
    MetricSpec("sidelobe_ratio", sidelobe_ratio, "edge"),
    MetricSpec("noise_gain", noise_gain, "flat_with_sigma"),
    MetricSpec("tail_spurious_grad", tail_spurious_grad, "flat"),
)


def run_metric_set(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    *,
    signal_mask: torch.Tensor | None = None,
    flat_mask: torch.Tensor | None = None,
    sigma_n: float | None = None,
    metric_specs: Iterable[MetricSpec] = DEFAULT_METRICS,
    dilate_px: int = 8,
    rel_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Run a metric set and return one `(B,)` tensor per metric name.

    Metric functions stay in `agfb_metrics.metrics`; this module only handles
    mask construction and dispatch. Frequency-response workflows should use
    this runner pattern with existing scalar metrics instead of defining a
    duplicate frequency-response metric.
    """
    if signal_mask is None or flat_mask is None:
        computed_masks = masks(g_x_t, g_y_t, dilate_px=dilate_px, rel_eps=rel_eps)
        if signal_mask is None:
            signal_mask = computed_masks["signal"]
        if flat_mask is None:
            flat_mask = computed_masks["flat"]

    out: dict[str, torch.Tensor] = {}
    for spec in metric_specs:
        if spec.input_kind == "edge":
            out[spec.name] = spec.fn(g_x, g_y, g_x_t, g_y_t, signal_mask)
        elif spec.input_kind == "flat":
            out[spec.name] = spec.fn(g_x, g_y, flat_mask)
        elif spec.input_kind == "flat_with_sigma":
            if sigma_n is None:
                raise ValueError(f"sigma_n is required to run {spec.name}")
            out[spec.name] = spec.fn(g_x, g_y, flat_mask, sigma_n)
        else:
            raise ValueError(f"unknown metric input kind: {spec.input_kind}")
    return out


def run_all_metrics(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    *,
    signal_mask: torch.Tensor | None = None,
    flat_mask: torch.Tensor | None = None,
    sigma_n: float | None = None,
    dilate_px: int = 8,
    rel_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Run the default metric set."""
    return run_metric_set(
        g_x,
        g_y,
        g_x_t,
        g_y_t,
        signal_mask=signal_mask,
        flat_mask=flat_mask,
        sigma_n=sigma_n,
        dilate_px=dilate_px,
        rel_eps=rel_eps,
    )
