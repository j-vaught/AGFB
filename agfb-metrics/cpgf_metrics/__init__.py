"""Batched, GPU-accelerated benchmark metrics for the CPGF gradient-filter suite."""

from cpgf_metrics.a1_nrmse import a1_nrmse
from cpgf_metrics.base import (
    magnitude,
    masks,
    unit_normal_from_truth,
)

__all__ = [
    "a1_nrmse",
    "magnitude",
    "masks",
    "unit_normal_from_truth",
]
