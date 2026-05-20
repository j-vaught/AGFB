"""Input validation tests for generator scale parameters."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from agfb_generators import (
    anisotropic_blob,
    asymmetric_ridge,
    curved_arc,
    curved_ridge,
    finite_ramp,
    gabor_packet,
    gaussian_blob,
    gaussian_ridge,
    mach_band,
    roof_profile,
    smoothed_bar,
    smoothed_l_junction,
    smoothed_ramp,
    smoothed_step,
    smoothed_t_junction,
    smoothed_x_junction,
    vessel_bifurcation,
    vessel_bifurcation_truth,
    vessel_crossing,
    vessel_crossing_truth,
)


@pytest.mark.parametrize(
    ("label", "render"),
    [
        ("smoothed_step_sigma", lambda: smoothed_step(16, 16, angle_rad=0.0, edge_sigma=0.0)),
        ("smoothed_bar_sigma", lambda: smoothed_bar(16, 16, edge_sigma=0.0)),
        (
            "smoothed_ramp_sigma",
            lambda: smoothed_ramp(16, 16, ramp_width=8.0, angle_rad=0.0, edge_sigma=0.0),
        ),
        ("curved_arc_sigma", lambda: curved_arc(16, 16, radius=8.0, edge_sigma=0.0)),
        ("mach_band_edge_sigma", lambda: mach_band(16, 16, edge_sigma=0.0)),
        ("mach_band_shoulder_sigma", lambda: mach_band(16, 16, shoulder_sigma=0.0)),
        (
            "smoothed_l_junction_sigma",
            lambda: smoothed_l_junction(16, 16, arm_width=8.0, edge_sigma=0.0),
        ),
        (
            "smoothed_t_junction_sigma",
            lambda: smoothed_t_junction(16, 16, arm_width=8.0, edge_sigma=0.0),
        ),
        (
            "smoothed_x_junction_sigma",
            lambda: smoothed_x_junction(16, 16, arm_width=8.0, edge_sigma=0.0),
        ),
        ("gaussian_blob_sigma", lambda: gaussian_blob(16, 16, scale_sigma=0.0)),
        ("gaussian_ridge_sigma", lambda: gaussian_ridge(16, 16, width_sigma=0.0, angle_rad=0.0)),
        (
            "gaussian_ridge_tensor_sigma",
            lambda: gaussian_ridge(
                16,
                16,
                width_sigma=torch.tensor([4.0, 0.0]),
                angle_rad=torch.tensor([0.0, 0.25]),
            ),
        ),
        (
            "asymmetric_ridge_sigma",
            lambda: asymmetric_ridge(
                16,
                16,
                negative_sigma=4.0,
                positive_sigma=0.0,
                angle_rad=0.0,
            ),
        ),
        (
            "curved_ridge_sigma",
            lambda: curved_ridge(16, 16, width_sigma=0.0, angle_rad=0.0, curvature=0.002),
        ),
        (
            "anisotropic_blob_sigma",
            lambda: anisotropic_blob(
                16,
                16,
                length_sigma=4.0,
                width_sigma=0.0,
                angle_rad=0.0,
            ),
        ),
        (
            "gabor_packet_sigma",
            lambda: gabor_packet(
                16,
                16,
                carrier_frequency=0.03,
                angle_rad=0.0,
                envelope_length_sigma=8.0,
                envelope_width_sigma=0.0,
            ),
        ),
        (
            "vessel_crossing_sigma",
            lambda: vessel_crossing(
                16,
                16,
                branch_a_width_sigma=0.0,
                branch_b_width_sigma=4.0,
                branch_a_normal_angle_rad=0.0,
                branch_b_normal_angle_rad=math.pi / 2.0,
            ),
        ),
        (
            "vessel_crossing_truth_sigma",
            lambda: vessel_crossing_truth(
                16,
                16,
                branch_a_width_sigma=0.0,
                branch_b_width_sigma=4.0,
                branch_a_normal_angle_rad=0.0,
                branch_b_normal_angle_rad=math.pi / 2.0,
            ),
        ),
        (
            "vessel_bifurcation_gate_sigma",
            lambda: vessel_bifurcation(
                16,
                16,
                trunk_width_sigma=4.0,
                left_width_sigma=4.0,
                right_width_sigma=4.0,
                trunk_tangent_angle_rad=math.pi / 2.0,
                left_tangent_angle_rad=math.pi / 4.0,
                right_tangent_angle_rad=3.0 * math.pi / 4.0,
                branch_gate_sigma=0.0,
            ),
        ),
        (
            "vessel_bifurcation_truth_gate_sigma",
            lambda: vessel_bifurcation_truth(
                16,
                16,
                trunk_width_sigma=4.0,
                left_width_sigma=4.0,
                right_width_sigma=4.0,
                trunk_tangent_angle_rad=math.pi / 2.0,
                left_tangent_angle_rad=math.pi / 4.0,
                right_tangent_angle_rad=3.0 * math.pi / 4.0,
                branch_gate_sigma=0.0,
            ),
        ),
        ("finite_ramp_width", lambda: finite_ramp(16, 16, ramp_width=0.0, angle_rad=0.0)),
        ("roof_profile_width", lambda: roof_profile(16, 16, roof_width=0.0)),
    ],
)
def test_nonpositive_scale_parameters_raise(
    label: str,
    render: Callable[[], object],
) -> None:
    del label
    with pytest.raises(ValueError, match="greater than zero"):
        render()
