from __future__ import annotations

import math

import pytest
import torch

from agfb_filters import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    farid_simoncelli_7_definition,
    get_filter_definition,
    multiscale_gaussian_derivative_orientation_banks,
    orientation_angles,
    recursive_gaussian_derivative_orientation_bank,
    riesz_orientation_bank,
    run_filter,
    run_orientation_bank,
    run_steered_filter_bank,
    steer_gradient,
)


def test_steer_gradient_projects_gradient_pair() -> None:
    gradient_x = torch.full((1, 4, 5), 2.0)
    gradient_y = torch.full((1, 4, 5), -1.0)
    angles = torch.tensor([0.0, math.pi / 2], dtype=torch.float32)

    result = steer_gradient(gradient_x, gradient_y, angles, definition_name="manual")

    assert result.definition_name == "manual"
    assert result.responses.shape == (1, 2, 4, 5)
    assert torch.allclose(result.responses[:, 0], gradient_x)
    assert torch.allclose(result.responses[:, 1], gradient_y)


def test_static_first_order_orientation_banks_zero_on_constants() -> None:
    image = torch.ones(2, 17, 19)
    cases = (
        ("steered_gaussian_derivative_orientation_bank", {"angle_count": 4, "sigma": 1.0}),
        ("farid_simoncelli_5_orientation_bank", {"angle_count": 4}),
        ("farid_simoncelli_7_orientation_bank", {"angle_count": 4}),
        (
            "matched_edge_orientation_bank",
            {"angle_count": 4, "sigma_across": 1.0, "sigma_along": 3.0},
        ),
    )

    for name, kwargs in cases:
        definition = get_filter_definition(name, **kwargs)
        result = run_orientation_bank(
            definition,
            image,
            path=ExecutionPath.ORIENTATION_BANK,
            boundary=definition.default_boundary,
        )
        assert result.responses.shape == (2, 4, 17, 19)
        assert torch.allclose(result.responses, torch.zeros_like(result.responses), atol=1e-5)


def test_steered_farid_bank_matches_gradient_projection() -> None:
    image = torch.randn(1, 16, 18)
    angles = orientation_angles(6)
    source = farid_simoncelli_7_definition()
    definition = get_filter_definition("farid_simoncelli_7_orientation_bank", angle_count=6)

    expected = run_steered_filter_bank(
        source,
        image,
        angles=angles,
        path=ExecutionPath.SEPARABLE,
        boundary=source.default_boundary,
    )
    actual = run_orientation_bank(
        definition,
        image,
        path=ExecutionPath.ORIENTATION_BANK,
        boundary=definition.default_boundary,
    )

    assert torch.allclose(actual.responses, expected.responses, atol=1e-6)


def test_matched_edge_bank_recovers_ramp_directional_derivative() -> None:
    rows = torch.arange(41, dtype=torch.float32).view(1, 41, 1)
    columns = torch.arange(43, dtype=torch.float32).view(1, 1, 43)
    image = 3.0 * columns + 4.0 * rows
    definition = get_filter_definition(
        "matched_edge_orientation_bank",
        angle_count=8,
        sigma_across=1.0,
        sigma_along=3.0,
    )

    result = run_orientation_bank(
        definition,
        image,
        path=ExecutionPath.ORIENTATION_BANK,
        boundary=definition.default_boundary,
    )
    expected = 3.0 * torch.cos(result.angles) + 4.0 * torch.sin(result.angles)

    assert torch.allclose(result.responses[:, :, 10:-10, 10:-10].mean(dim=(0, 2, 3)), expected)


def test_gxgy_orientation_wrappers_return_expected_shapes() -> None:
    image = torch.ones(1, 15, 17)

    recursive = recursive_gaussian_derivative_orientation_bank(
        image,
        angle_count=5,
        sigma=1.2,
    )
    assert recursive.responses.shape == (1, 5, 15, 17)
    assert torch.allclose(recursive.responses, torch.zeros_like(recursive.responses), atol=1e-5)

    multiscale = multiscale_gaussian_derivative_orientation_banks(
        image,
        sigmas=(1.0, 2.0),
        angle_count=5,
    )
    assert len(multiscale) == 2
    assert all(result.responses.shape == (1, 5, 15, 17) for result in multiscale)
    assert all(
        torch.allclose(result.responses, torch.zeros_like(result.responses), atol=1e-5)
        for result in multiscale
    )

    riesz = riesz_orientation_bank(image, angle_count=5)
    assert riesz.responses.shape == (1, 5, 15, 17)
    assert torch.allclose(riesz.responses, torch.zeros_like(riesz.responses), atol=1e-5)


def test_riesz_transform_requires_fft_path_and_circular_boundary() -> None:
    image = torch.randn(1, 12, 14)
    definition = get_filter_definition("riesz_transform")

    with pytest.raises(ValueError, match="fft path"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.SPATIAL_DENSE,
            boundary=definition.default_boundary,
        )

    with pytest.raises(ValueError, match="does not support replicate boundary"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.FFT,
            boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        )
