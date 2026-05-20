from __future__ import annotations

import torch

from agfb_filters import (
    CPGF,
    DerivativeOfGaussian,
    ExecutionStrategy,
    FreemanAdelsonG1,
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
        central_difference,
        farid_simoncelli_5,
        prewitt_3,
        roberts,
        scharr_3,
        sobel_3,
        sobel_5,
        sobel_7,
        CPGF(radius=2, degree=2).apply,
        DerivativeOfGaussian(sigma=1.0).apply,
        FreemanAdelsonG1(sigma=1.0).apply,
        SavitzkyGolay(radius=2, degree=2).apply,
    ]

    for apply_filter in filters:
        gradient_x, gradient_y = apply_filter(image)
        assert gradient_x.shape == image.shape
        assert gradient_y.shape == image.shape
        assert gradient_x.dtype == torch.float32
        assert gradient_y.dtype == torch.float32


def test_runner_can_apply_definition_directly() -> None:
    image = torch.randn(2, 8, 9)
    gradient_x, gradient_y = run_filter(sobel_definition(3), image)

    assert gradient_x.shape == image.shape
    assert gradient_y.shape == image.shape


def test_dense_runner_strategies_match_for_cpgf_definition() -> None:
    image = torch.randn(1, 16, 17)
    definition = cpgf_definition(radius=2, degree=2)

    spatial_gradient_x, spatial_gradient_y = run_filter(
        definition,
        image,
        strategy=ExecutionStrategy.SPATIAL,
    )
    fft_gradient_x, fft_gradient_y = run_filter(
        definition,
        image,
        strategy=ExecutionStrategy.FFT,
    )

    assert torch.allclose(spatial_gradient_x, fft_gradient_x, atol=1e-5)
    assert torch.allclose(spatial_gradient_y, fft_gradient_y, atol=1e-5)
