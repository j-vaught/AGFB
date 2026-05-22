from __future__ import annotations

import pytest
import torch

from agfb_filters import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    GradientFilterDefinition,
    cpgf_definition,
    define_dense_filter,
    define_separable_filter,
    get_filter_definition,
    register_filter,
    registered_filters,
    run_filter,
)


def test_dense_filter_helper_supports_custom_filter_without_package_edits() -> None:
    definition = define_dense_filter(
        name="unit_test_dense_difference",
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        kernel_x=[
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ],
        kernel_y=[
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        metadata={"source": "unit_test"},
    )
    image = torch.arange(25, dtype=torch.float32).view(1, 5, 5)

    gradient_x, gradient_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=definition.default_boundary,
    )

    assert gradient_x.shape == image.shape
    assert gradient_y.shape == image.shape
    assert torch.allclose(gradient_x[:, 1:-1, 1:-1], torch.ones(1, 3, 3))
    assert torch.allclose(gradient_y[:, 1:-1, 1:-1], torch.full((1, 3, 3), 5.0))


def test_separable_filter_helper_generates_dense_kernels() -> None:
    definition = define_separable_filter(
        name="unit_test_separable_difference",
        smooth_kernel_1d=[1.0],
        derivative_kernel_1d=[-0.5, 0.0, 0.5],
    )

    assert definition.has_separable_kernels
    assert definition.has_dense_kernels
    kernel_x, kernel_y = definition.dense_kernels()
    assert kernel_x.shape == (3, 3)
    assert kernel_y.shape == (3, 3)


def test_registry_builds_built_in_and_custom_filter_definitions() -> None:
    custom_definition = define_separable_filter(
        name="unit_test_registered_difference",
        smooth_kernel_1d=[1.0],
        derivative_kernel_1d=[-0.5, 0.0, 0.5],
    )
    register_filter(
        "unit_test_registered_difference",
        lambda: custom_definition,
        description="unit test registered filter",
        replace=True,
    )

    built_in_definition = get_filter_definition("sobel_3")
    registered_definition = get_filter_definition("unit_test_registered_difference")

    assert isinstance(built_in_definition, GradientFilterDefinition)
    assert built_in_definition.name == "sobel_3"
    assert registered_definition is custom_definition
    assert "unit_test_registered_difference" in registered_filters()


def test_registry_rejects_duplicate_names_without_replace() -> None:
    definition = define_separable_filter(
        name="unit_test_duplicate",
        smooth_kernel_1d=[1.0],
        derivative_kernel_1d=[-0.5, 0.0, 0.5],
    )
    register_filter("unit_test_duplicate", lambda: definition, replace=True)

    with pytest.raises(ValueError, match="already registered"):
        register_filter("unit_test_duplicate", lambda: definition)


def test_custom_filter_validation_fails_early_for_bad_dense_shape() -> None:
    with pytest.raises(ValueError, match="kernel shapes must match"):
        define_dense_filter(
            name="unit_test_bad_dense",
            kernel_x=[[1.0, 0.0, -1.0]],
            kernel_y=[[1.0], [0.0], [-1.0]],
        )


def test_custom_filter_validation_requires_padding_for_even_dense_kernels() -> None:
    with pytest.raises(ValueError, match="spatial_padding"):
        define_dense_filter(
            name="unit_test_bad_even_dense",
            kernel_x=[[-1.0, 1.0], [-1.0, 1.0]],
            kernel_y=[[-1.0, -1.0], [1.0, 1.0]],
        )


def test_polynomial_filter_validation_rejects_underdetermined_fits() -> None:
    with pytest.raises(ValueError, match="underdetermined"):
        cpgf_definition(radius=1, degree=2)


def test_cpgf_large_high_degree_fit_is_well_conditioned() -> None:
    definition = cpgf_definition(radius=200, degree=11)
    kernel_x, kernel_y = definition.dense_kernels()

    coords = torch.arange(-200, 201, dtype=kernel_x.dtype)
    row_grid, column_grid = torch.meshgrid(coords, coords, indexing="ij")

    assert kernel_x.shape == (401, 401)
    assert kernel_y.shape == (401, 401)
    assert torch.isfinite(kernel_x).all()
    assert torch.isfinite(kernel_y).all()
    assert torch.isclose((kernel_x * column_grid).sum(), torch.tensor(1.0))
    assert torch.isclose((kernel_y * row_grid).sum(), torch.tensor(1.0))
    assert torch.isclose((kernel_x * row_grid).sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose((kernel_y * column_grid).sum(), torch.tensor(0.0), atol=1e-6)
