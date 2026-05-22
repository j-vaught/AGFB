"""Batched, GPU-accelerated benchmark metrics for the AGFB suite."""

from agfb_metrics.angular_mae import angular_mae
from agfb_metrics.base import (
    magnitude,
    masks,
    ridge_mask_from_truth,
    unit_normal_from_truth,
)
from agfb_metrics.edge_fwhm import edge_fwhm
from agfb_metrics.localization_offset import localization_offset
from agfb_metrics.magnitude_bias import magnitude_bias
from agfb_metrics.noise_gain import noise_gain
from agfb_metrics.nrmse import nrmse
from agfb_metrics.sidelobe_ratio import sidelobe_ratio
from agfb_metrics.tail_spurious_grad import tail_spurious_grad
from agfb_metrics.tail_vector_error import tail_vector_error
from agfb_metrics.tangential_normal_leak import tangential_normal_leak

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
]
