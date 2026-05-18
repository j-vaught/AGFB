"""Batched GPU-accelerated synthetic generators for the AGFB benchmark."""

from agfb_generators.base import Frame, coord_grid
from agfb_generators.composite import CompositeRect, composite
from agfb_generators.curved_arc import curved_arc
from agfb_generators.gaussian_blob import gaussian_blob
from agfb_generators.gaussian_ridge import gaussian_ridge
from agfb_generators.hard_step import hard_step
from agfb_generators.polynomial import polynomial
from agfb_generators.sinusoid import sinusoid
from agfb_generators.smoothed_bar import smoothed_bar
from agfb_generators.smoothed_step import smoothed_step

__all__ = [
    "CompositeRect",
    "Frame",
    "composite",
    "coord_grid",
    "curved_arc",
    "gaussian_blob",
    "gaussian_ridge",
    "hard_step",
    "polynomial",
    "sinusoid",
    "smoothed_bar",
    "smoothed_step",
]
