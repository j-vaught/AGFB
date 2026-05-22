"""Batched, GPU-accelerated benchmark metrics for the AGFB suite."""

from agfb_metrics.a1_nrmse import a1_nrmse
from agfb_metrics.a2_angular_mae import a2_angular_mae
from agfb_metrics.a3_tail_vector_error import a3_tail_vector_error
from agfb_metrics.b1_localization_offset import b1_localization_offset
from agfb_metrics.b2_tangential_normal_leak import b2_tangential_normal_leak
from agfb_metrics.b3_magnitude_bias import b3_magnitude_bias
from agfb_metrics.b4_edge_fwhm import b4_edge_fwhm
from agfb_metrics.b5_sidelobe_ratio import b5_sidelobe_ratio
from agfb_metrics.base import (
    magnitude,
    masks,
    ridge_mask_from_truth,
    unit_normal_from_truth,
)
from agfb_metrics.c1_noise_gain import c1_noise_gain
from agfb_metrics.c2_tail_spurious_grad import c2_tail_spurious_grad

__all__ = [
    "a1_nrmse",
    "a2_angular_mae",
    "a3_tail_vector_error",
    "b1_localization_offset",
    "b2_tangential_normal_leak",
    "b3_magnitude_bias",
    "b4_edge_fwhm",
    "b5_sidelobe_ratio",
    "c1_noise_gain",
    "c2_tail_spurious_grad",
    "magnitude",
    "masks",
    "ridge_mask_from_truth",
    "unit_normal_from_truth",
]
