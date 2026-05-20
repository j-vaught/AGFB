from __future__ import annotations

import torch

from agfb_filters import (
    AGFB,
    DerivativeOfGaussian,
    FreemanAdelsonG1,
    SavitzkyGolay,
    central_difference,
    farid_simoncelli_5,
    prewitt_3,
    roberts,
    scharr_3,
    sobel_3,
    sobel_5,
    sobel_7,
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
        AGFB(radius=2, degree=2).apply,
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
