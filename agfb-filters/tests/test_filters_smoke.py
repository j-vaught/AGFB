from __future__ import annotations

from typing import Any

import pytest
import torch

from agfb_filters import (
    CPGF,
    DerivativeOfGaussian,
    ExecutionPath,
    FreemanAdelsonG1,
    GradientFilterDefinition,
    SavitzkyGolay,
    central_difference,
    cpgf_definition,
    farid_simoncelli_5,
    prewitt_3,
    roberts,
    run_filter,
    scharr_3,
    sobel_3,
    sobel_5,
    sobel_7,
    sobel_definition,
)


def test_filters_return_gradient_pair_with_input_shape() -> None:
    image = torch.randn(2, 8, 9)
    filters = [
        (central_difference, ExecutionPath.SEPARABLE),
        (farid_simoncelli_5, ExecutionPath.SEPARABLE),
        (prewitt_3, ExecutionPath.SEPARABLE),
        (roberts, ExecutionPath.STENCIL),
        (scharr_3, ExecutionPath.SEPARABLE),
        (sobel_3, ExecutionPath.SEPARABLE),
        (sobel_5, ExecutionPath.SEPARABLE),
        (sobel_7, ExecutionPath.SEPARABLE),
        (CPGF(radius=2, degree=2).apply, ExecutionPath.SPARSE_OFFSETS),
        (DerivativeOfGaussian(sigma=1.0).apply, ExecutionPath.SEPARABLE),
        (FreemanAdelsonG1(sigma=1.0).apply, ExecutionPath.SEPARABLE),
        (SavitzkyGolay(radius=2, degree=2).apply, ExecutionPath.SPATIAL_DENSE),
    ]

    for apply_filter, path in filters:
        gradient_x, gradient_y = apply_filter(image, path=path)
        assert gradient_x.shape == image.shape
        assert gradient_y.shape == image.shape
        assert gradient_x.dtype == torch.float32
        assert gradient_y.dtype == torch.float32


def test_runner_can_apply_definition_directly_with_explicit_path() -> None:
    image = torch.randn(2, 8, 9)
    gradient_x, gradient_y = run_filter(sobel_definition(3), image, path=ExecutionPath.SEPARABLE)

    assert gradient_x.shape == image.shape
    assert gradient_y.shape == image.shape


def test_dense_runner_paths_match_for_cpgf_definition() -> None:
    image = torch.randn(1, 16, 17)
    definition = cpgf_definition(radius=2, degree=2)

    reference_x, reference_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
    )
    for path in (
        ExecutionPath.FFT,
        ExecutionPath.SPARSE_OFFSETS,
        ExecutionPath.ANTIPODAL_PAIRS,
    ):
        gradient_x, gradient_y = run_filter(definition, image, path=path)
        assert torch.allclose(reference_x, gradient_x, atol=1e-5)
        assert torch.allclose(reference_y, gradient_y, atol=1e-5)


def test_run_filter_requires_explicit_path() -> None:
    image = torch.randn(1, 8, 9)
    untyped_run_filter: Any = run_filter
    with pytest.raises(TypeError, match="path"):
        untyped_run_filter(sobel_definition(3), image)


def test_run_filter_rejects_automatic_or_unknown_paths() -> None:
    image = torch.randn(1, 8, 9)
    with pytest.raises(ValueError, match="unsupported execution path"):
        run_filter(sobel_definition(3), image, path="auto")


def test_run_filter_rejects_invalid_kernel_path_combination() -> None:
    image = torch.randn(1, 8, 9)
    with pytest.raises(ValueError, match="requires separable kernels"):
        run_filter(cpgf_definition(radius=2, degree=2), image, path=ExecutionPath.SEPARABLE)
    with pytest.raises(ValueError, match="stencil path"):
        run_filter(cpgf_definition(radius=2, degree=2), image, path=ExecutionPath.STENCIL)


def test_antipodal_path_rejects_non_symmetric_kernels() -> None:
    image = torch.randn(1, 8, 9)
    definition = GradientFilterDefinition(
        name="non_symmetric",
        padding_mode="replicate",
        kernel_x=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
        kernel_y=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="odd symmetry"):
        run_filter(definition, image, path=ExecutionPath.ANTIPODAL_PAIRS)
