"""Batched, GPU-accelerated benchmark metrics for the CPGF gradient-filter suite."""

from cpgf_metrics.base import (
    magnitude,
    masks,
    unit_normal_from_truth,
)

__all__ = [
    "magnitude",
    "masks",
    "unit_normal_from_truth",
]
