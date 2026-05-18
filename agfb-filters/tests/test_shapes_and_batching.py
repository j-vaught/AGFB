"""Every filter accepts (B, H, W) input and returns (B, H, W) g_x and g_y.

Batched output[i] must equal scalar output for each i (within float32 epsilon).
"""

from __future__ import annotations

import torch

from cpgf_filters import (
    CPGF,
    DoG,
    FreemanAdelsonG1,
    SavitzkyGolay,
    farid_simoncelli_5,
    prewitt_3,
    roberts,
    scharr_3,
    sobel_3,
    sobel_5,
    sobel_7,
)


def _filters():
    return [
        ("sobel_3", lambda I: sobel_3(I)),
        ("sobel_5", lambda I: sobel_5(I)),
        ("sobel_7", lambda I: sobel_7(I)),
        ("prewitt_3", lambda I: prewitt_3(I)),
        ("scharr_3", lambda I: scharr_3(I)),
        ("roberts", lambda I: roberts(I)),
        ("farid_simoncelli_5", lambda I: farid_simoncelli_5(I)),
        ("DoG(1.5)", DoG(sigma=1.5).apply),
        ("SavitzkyGolay(r=3,d=3)", SavitzkyGolay(r=3, d=3).apply),
        ("CPGF(r=5,d=3)", CPGF(r=5, d=3).apply),
        ("FreemanAdelsonG1(1.5)", FreemanAdelsonG1(sigma=1.5).apply),
    ]


def test_each_filter_preserves_shape() -> None:
    torch.manual_seed(0)
    I = torch.randn(4, 64, 96)
    for name, fn in _filters():
        gx, gy = fn(I)
        assert gx.shape == (4, 64, 96), name
        assert gy.shape == (4, 64, 96), name
        assert torch.isfinite(gx).all(), name
        assert torch.isfinite(gy).all(), name


def test_batched_matches_per_image() -> None:
    torch.manual_seed(1)
    batch = torch.randn(3, 48, 48)
    for name, fn in _filters():
        gx_b, gy_b = fn(batch)
        for i in range(3):
            gx_i, gy_i = fn(batch[i : i + 1])
            assert torch.allclose(gx_b[i], gx_i[0], atol=1e-5), f"{name} gx[{i}]"
            assert torch.allclose(gy_b[i], gy_i[0], atol=1e-5), f"{name} gy[{i}]"
