"""Metric definitions and shared metric helpers."""

from agfb_metrics.metrics.angular_mae import angular_mae
from agfb_metrics.metrics.base import (
    magnitude,
    masks,
    ridge_mask_from_truth,
    unit_normal_from_truth,
)
from agfb_metrics.metrics.edge_fwhm import edge_fwhm
from agfb_metrics.metrics.evaluator import (
    ALL_METRICS,
    PIXEL_METRICS,
    MetricEvaluator,
    MetricName,
    evaluate_all_metrics,
    evaluate_metrics,
)
from agfb_metrics.metrics.localization_offset import localization_offset
from agfb_metrics.metrics.magnitude_bias import magnitude_bias
from agfb_metrics.metrics.noise_gain import noise_gain
from agfb_metrics.metrics.nrmse import nrmse
from agfb_metrics.metrics.sidelobe_ratio import sidelobe_ratio
from agfb_metrics.metrics.tail_spurious_grad import tail_spurious_grad
from agfb_metrics.metrics.tail_vector_error import tail_vector_error
from agfb_metrics.metrics.tangential_normal_leak import tangential_normal_leak

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
    "MetricName",
    "MetricEvaluator",
    "ALL_METRICS",
    "PIXEL_METRICS",
    "evaluate_metrics",
    "evaluate_all_metrics",
    "magnitude",
    "masks",
    "ridge_mask_from_truth",
    "unit_normal_from_truth",
]
