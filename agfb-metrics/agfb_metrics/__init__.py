"""Batched, GPU-accelerated benchmark metrics for the AGFB suite."""

from agfb_metrics.metrics import (
    angular_mae,
    edge_fwhm,
    localization_offset,
    magnitude,
    magnitude_bias,
    masks,
    noise_gain,
    nrmse,
    ridge_mask_from_truth,
    sidelobe_ratio,
    tail_spurious_grad,
    tail_vector_error,
    tangential_normal_leak,
    unit_normal_from_truth,
)
from agfb_metrics.runners import DEFAULT_METRICS, MetricSpec, run_all_metrics, run_metric_set

__all__ = [
    "nrmse",
    "angular_mae",
    "tail_vector_error",
    "localization_offset",
    "tangential_normal_leak",
    "magnitude_bias",
    "edge_fwhm",
    "sidelobe_ratio",
    "noise_gain",
    "tail_spurious_grad",
    "magnitude",
    "masks",
    "ridge_mask_from_truth",
    "unit_normal_from_truth",
    "MetricSpec",
    "DEFAULT_METRICS",
    "run_metric_set",
    "run_all_metrics",
]
