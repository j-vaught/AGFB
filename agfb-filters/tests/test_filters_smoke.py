from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest
import torch

from agfb_filters import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    GradientFilterDefinition,
    cpgf_definition,
    get_filter_definition,
    get_filter_registration,
    run_filter,
    run_orientation_bank,
    shipped_filter_specs,
    sobel_definition,
)


def test_gradient_filters_return_gradient_pair_with_input_shape() -> None:
    image = torch.randn(2, 8, 9)

    for spec in shipped_filter_specs():
        definition = get_filter_definition(spec.name, **dict(spec.smoke_kwargs))
        if spec.output_api != "gradient":
            continue
        gradient_x, gradient_y = run_filter(
            definition,
            image,
            path=ExecutionPath(spec.smoke_path),
            boundary=definition.default_boundary,
        )
        assert gradient_x.shape == image.shape
        assert gradient_y.shape == image.shape
        assert gradient_x.dtype == image.dtype
        assert gradient_y.dtype == image.dtype


def test_orientation_bank_filters_return_raw_bank_with_input_shape() -> None:
    image = torch.randn(2, 8, 9)

    for spec in shipped_filter_specs():
        if spec.output_api != "orientation_bank":
            continue
        definition = get_filter_definition(spec.name, **dict(spec.smoke_kwargs))
        result = run_orientation_bank(
            definition,
            image,
            path=ExecutionPath(spec.smoke_path),
            boundary=definition.default_boundary,
        )
        assert result.responses.shape == (2, int(definition.parameters["angle_count"]), 8, 9)
        assert result.angles.shape == (int(definition.parameters["angle_count"]),)
        assert result.definition_name == definition.name


def test_shipped_filter_catalog_entries_are_complete() -> None:
    import agfb_filters
    import agfb_filters.filters

    for spec in shipped_filter_specs():
        module = import_module(spec.module)
        factory = getattr(module, spec.definition_factory)
        definition = get_filter_definition(spec.name, **dict(spec.smoke_kwargs))
        registration = get_filter_registration(spec.name)

        assert callable(factory)
        assert registration.name == spec.name
        assert isinstance(definition, GradientFilterDefinition)
        assert definition.name == spec.name
        assert definition.operator_family
        assert definition.stage_count >= 1
        assert definition.support_shape
        assert definition.orientation_model
        assert definition.shape_model == "same"
        assert isinstance(definition.parameters, dict)
        for export_name in spec.exports:
            assert getattr(agfb_filters, export_name) is getattr(
                agfb_filters.filters,
                export_name,
            )


def test_split_package_import_paths_are_available() -> None:
    from agfb_filters.filters import sobel_definition as filters_sobel_definition
    from agfb_filters.runtime import run_filter as runtime_run_filter

    assert filters_sobel_definition is sobel_definition
    assert runtime_run_filter is run_filter


def test_dense_runner_paths_match_for_cpgf_definition() -> None:
    image = torch.randn(1, 16, 17)
    definition = cpgf_definition(radius=2, degree=2)

    reference_x, reference_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=definition.default_boundary,
    )
    for path in (
        ExecutionPath.FFT,
        ExecutionPath.SPARSE_OFFSETS,
        ExecutionPath.ANTIPODAL_PAIRS,
    ):
        gradient_x, gradient_y = run_filter(
            definition,
            image,
            path=path,
            boundary=definition.default_boundary,
        )
        assert torch.allclose(reference_x, gradient_x, atol=1e-5)
        assert torch.allclose(reference_y, gradient_y, atol=1e-5)


def test_run_filter_requires_explicit_path() -> None:
    image = torch.randn(1, 8, 9)
    untyped_run_filter: Any = run_filter
    with pytest.raises(TypeError, match="path"):
        untyped_run_filter(sobel_definition(3), image)


def test_run_filter_rejects_automatic_or_unknown_paths() -> None:
    image = torch.randn(1, 8, 9)
    definition = sobel_definition(3)
    with pytest.raises(ValueError, match="unsupported execution path"):
        run_filter(definition, image, path="auto", boundary=definition.default_boundary)


def test_run_filter_requires_explicit_boundary() -> None:
    image = torch.randn(1, 8, 9)
    with pytest.raises(ValueError, match="boundary"):
        run_filter(sobel_definition(3), image, path=ExecutionPath.SEPARABLE)


def test_run_filter_rejects_invalid_kernel_path_combination() -> None:
    image = torch.randn(1, 8, 9)
    definition = cpgf_definition(radius=2, degree=2)
    with pytest.raises(ValueError, match="requires separable kernels"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.SEPARABLE,
            boundary=definition.default_boundary,
        )
    with pytest.raises(ValueError, match="stencil path"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.STENCIL,
            boundary=definition.default_boundary,
        )


def test_boundary_validation_rejects_unknown_mode() -> None:
    bad_mode: Any = "mirror"
    with pytest.raises(ValueError, match="unsupported boundary mode"):
        BoundaryCondition(bad_mode)


def test_boundary_validation_rejects_nonzero_value_for_non_constant_mode() -> None:
    with pytest.raises(ValueError, match="constant mode"):
        BoundaryCondition(BoundaryMode.REFLECT, value=1.0)


def test_run_filter_preserves_float64() -> None:
    image = torch.randn(1, 8, 9, dtype=torch.float64)
    definition = sobel_definition(3)

    gradient_x, gradient_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=definition.default_boundary,
    )

    assert gradient_x.dtype == torch.float64
    assert gradient_y.dtype == torch.float64
    assert gradient_x.shape == image.shape
    assert gradient_y.shape == image.shape


def test_run_filter_rejects_integer_inputs() -> None:
    image = torch.ones(1, 8, 9, dtype=torch.int64)
    definition = sobel_definition(3)

    with pytest.raises(ValueError, match="floating-point dtype"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.SPATIAL_DENSE,
            boundary=definition.default_boundary,
        )


def test_constant_boundary_value_changes_edge_output() -> None:
    image = torch.zeros(1, 5, 5)
    definition = sobel_definition(3)
    zero_x, _ = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=BoundaryCondition(BoundaryMode.CONSTANT),
    )
    nonzero_x, _ = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=BoundaryCondition(BoundaryMode.CONSTANT, value=2.0),
    )

    assert not torch.allclose(zero_x, nonzero_x)
    assert float(nonzero_x.abs().max()) > 0.0


def test_reflect_boundary_rejects_padding_too_large_for_input() -> None:
    image = torch.randn(1, 1, 4)
    definition = sobel_definition(3)
    with pytest.raises(ValueError, match="reflect boundary"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.SPATIAL_DENSE,
            boundary=BoundaryCondition(BoundaryMode.REFLECT),
        )


def test_antipodal_path_rejects_non_symmetric_kernels() -> None:
    image = torch.randn(1, 8, 9)
    definition = GradientFilterDefinition(
        name="non_symmetric",
        default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
        kernel_x=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
        kernel_y=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="odd symmetry"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.ANTIPODAL_PAIRS,
            boundary=definition.default_boundary,
        )
