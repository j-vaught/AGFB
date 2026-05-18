"""Batched, GPU-accelerated benchmark metrics for the CPGF gradient-filter suite."""

from cpgf_metrics.a1_nrmse import a1_nrmse
from cpgf_metrics.a2_angular_mae import a2_angular_mae
from cpgf_metrics.a3_tail_vector_error import a3_tail_vector_error
from cpgf_metrics.b1_localization_offset import b1_localization_offset
from cpgf_metrics.b2_tangential_normal_leak import b2_tangential_normal_leak
from cpgf_metrics.b3_magnitude_bias import b3_magnitude_bias
from cpgf_metrics.b4_edge_fwhm import b4_edge_fwhm
from cpgf_metrics.b5_sidelobe_ratio import b5_sidelobe_ratio
from cpgf_metrics.base import (
    magnitude,
    masks,
    unit_normal_from_truth,
)
from cpgf_metrics.c1_noise_gain import c1_noise_gain
from cpgf_metrics.c2_tail_spurious_grad import c2_tail_spurious_grad

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
    "unit_normal_from_truth",
]
