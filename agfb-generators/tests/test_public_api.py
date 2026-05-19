"""Public package export checks."""

from __future__ import annotations

import agfb_generators


def test_public_exports_include_generator_families() -> None:
    """Check that every public generator family is exported from the package."""
    expected = {
        "Frame",
        "anisotropic_blob",
        "asymmetric_ridge",
        "chirp",
        "coord_grid",
        "curved_arc",
        "curved_ridge",
        "finite_ramp",
        "gabor_packet",
        "gaussian_blob",
        "gaussian_ridge",
        "hard_l_junction",
        "hard_t_junction",
        "hard_x_junction",
        "hard_y_junction",
        "junction_mask",
        "mach_band",
        "polynomial",
        "roof_profile",
        "sinusoid",
        "smoothed_bar",
        "smoothed_l_junction",
        "smoothed_ramp",
        "smoothed_step",
        "smoothed_t_junction",
        "smoothed_x_junction",
        "smoothed_y_junction",
        "vessel_bifurcation",
        "vessel_bifurcation_truth",
        "vessel_crossing",
        "vessel_crossing_truth",
    }

    assert set(agfb_generators.__all__) == expected
    for name in expected:
        assert hasattr(agfb_generators, name), name


def test_composite_api_is_removed() -> None:
    """Check that the old rectangular composite API is no longer public."""
    assert "composite" not in agfb_generators.__all__
    assert "CompositeRect" not in agfb_generators.__all__
    assert not hasattr(agfb_generators, "composite")
    assert not hasattr(agfb_generators, "CompositeRect")
