"""Batched GPU-accelerated synthetic generators (§1.1 of the CPGF benchmark)."""

from cpgf_generators.base import Frame, coord_grid
from cpgf_generators.composite import CompositeRect, composite
from cpgf_generators.curved_arc import curved_arc
from cpgf_generators.diagnostics import constant_field, contrast_ramp, multi_freq_grating
from cpgf_generators.gaussian_blob import gaussian_blob
from cpgf_generators.gaussian_ridge import gaussian_ridge
from cpgf_generators.hard_step import hard_step
from cpgf_generators.polynomial import polynomial
from cpgf_generators.sinusoid import sinusoid
from cpgf_generators.smoothed_bar import smoothed_bar
from cpgf_generators.smoothed_step import smoothed_step

__all__ = [
    "CompositeRect",
    "Frame",
    "composite",
    "constant_field",
    "contrast_ramp",
    "coord_grid",
    "curved_arc",
    "gaussian_blob",
    "gaussian_ridge",
    "hard_step",
    "multi_freq_grating",
    "polynomial",
    "sinusoid",
    "smoothed_bar",
    "smoothed_step",
]
