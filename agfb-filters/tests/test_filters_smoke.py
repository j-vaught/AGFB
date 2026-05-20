from __future__ import annotations

from typing import Any

import pytest
import torch

from agfb_filters import (
    CPGF,
    BoundaryCondition,
    BoundaryMode,
    DerivativeOfGaussian,
    ExecutionPath,
    ExecutionPlan,
    FreemanAdelsonG1,
    GradientFilterDefinition,
    InputSignature,
    SavitzkyGolay,
    central_difference,
    central_difference_definition,
    cpgf_definition,
    farid_simoncelli_5,
    farid_simoncelli_5_definition,
    prewitt_3,
    prewitt_3_definition,
    roberts,
    roberts_definition,
    run_filter,
    scharr_3,
    scharr_3_definition,
    sobel_3,
    sobel_5,
    sobel_7,
    sobel_definition,
)


def test_filters_return_gradient_pair_with_input_shape() -> None:
    image = torch.randn(2, 8, 9)
    cpgf = CPGF(radius=2, degree=2)
    derivative_of_gaussian = DerivativeOfGaussian(sigma=1.0)
    freeman_adelson = FreemanAdelsonG1(sigma=1.0)
    savitzky_golay = SavitzkyGolay(radius=2, degree=2)
    filters = [
        (central_difference, central_difference_definition(), ExecutionPath.SEPARABLE),
        (farid_simoncelli_5, farid_simoncelli_5_definition(), ExecutionPath.SEPARABLE),
        (prewitt_3, prewitt_3_definition(), ExecutionPath.SEPARABLE),
        (roberts, roberts_definition(), ExecutionPath.STENCIL),
        (scharr_3, scharr_3_definition(), ExecutionPath.SEPARABLE),
        (sobel_3, sobel_definition(3), ExecutionPath.SEPARABLE),
        (sobel_5, sobel_definition(5), ExecutionPath.SEPARABLE),
        (sobel_7, sobel_definition(7), ExecutionPath.SEPARABLE),
        (cpgf.apply, cpgf.definition, ExecutionPath.SPARSE_OFFSETS),
        (
            derivative_of_gaussian.apply,
            derivative_of_gaussian.definition,
            ExecutionPath.SEPARABLE,
        ),
        (freeman_adelson.apply, freeman_adelson.definition, ExecutionPath.SEPARABLE),
        (savitzky_golay.apply, savitzky_golay.definition, ExecutionPath.SPATIAL_DENSE),
    ]

    for apply_filter, definition, path in filters:
        gradient_x, gradient_y = apply_filter(
            image,
            path=path,
            boundary=definition.default_boundary,
        )
        assert gradient_x.shape == image.shape
        assert gradient_y.shape == image.shape
        assert gradient_x.dtype == torch.float32
        assert gradient_y.dtype == torch.float32


def test_runner_can_apply_definition_directly_with_explicit_path() -> None:
    image = torch.randn(2, 8, 9)
    definition = sobel_definition(3)
    gradient_x, gradient_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SEPARABLE,
        boundary=definition.default_boundary,
    )

    assert gradient_x.shape == image.shape
    assert gradient_y.shape == image.shape


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


def test_run_filter_rejects_boundary_mismatch_with_plan() -> None:
    image = torch.randn(1, 8, 9)
    definition = sobel_definition(3)
    plan = ExecutionPlan(
        path=ExecutionPath.SEPARABLE,
        input_signature=InputSignature.from_tensor(image),
        boundary=definition.default_boundary,
        filter_fingerprint=definition.fingerprint(),
        reason="test plan",
    )

    with pytest.raises(ValueError, match="boundary"):
        run_filter(
            definition,
            image,
            path=plan,
            boundary=BoundaryCondition(BoundaryMode.CONSTANT),
        )


def test_execution_plan_supplies_boundary_when_direct_boundary_is_absent() -> None:
    image = torch.randn(1, 8, 9)
    definition = sobel_definition(3)
    plan = ExecutionPlan(
        path=ExecutionPath.SEPARABLE,
        input_signature=InputSignature.from_tensor(image),
        boundary=definition.default_boundary,
        filter_fingerprint=definition.fingerprint(),
        reason="test plan",
    )

    gradient_x, gradient_y = run_filter(definition, image, path=plan)

    assert gradient_x.shape == image.shape
    assert gradient_y.shape == image.shape


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
