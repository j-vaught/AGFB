"""Regression checks for the notebook display helper module."""

from __future__ import annotations

import torch

from agfb_generators.notebook import set_color_scheme, show_color_scheme, show_image


def test_notebook_color_scheme_smoke() -> None:
    """Check that user-defined notebook color schemes can be applied."""
    color_scheme = {
        "intensity": [(0.0, "#000000"), (1.0, "#FFFFFF")],
        "magnitude": [(0.0, "#000000"), (1.0, "#FFFFFF")],
        "signed": [(0.0, "#000000"), (0.5, "#A2A2A2"), (1.0, "#FFFFFF")],
        "mask": [(0.0, "#000000"), (1.0, "#FFFFFF")],
    }

    set_color_scheme(color_scheme)
    show_color_scheme(color_scheme)


def test_notebook_show_image_smoke() -> None:
    """Check the smallest tensor display path used by the notebook."""
    image = torch.zeros(1, 8, 8)
    image[0, 2:6, 2:6] = 1.0

    show_image(image[0], "test image")
