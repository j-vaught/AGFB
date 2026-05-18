"""Sobel-3 / Sobel-5 / Sobel-7 (separable, replicate padding).

`Sobel-3` matches the PGF_paper prototype exactly. The 5- and 7-tap versions
are the natural recursive extension: Sobel-n = Sobel-3 convolved
`(n - 3) // 2` more times with the 3-tap binomial smoother `[1, 2, 1] / 4`.
"""

from __future__ import annotations

import torch

from cpgf_filters.base import check_input, conv_1d, separable_gradient

_SMOOTH3 = torch.tensor([1.0, 2.0, 1.0]) / 4.0
_DIFF3 = torch.tensor([-1.0, 0.0, 1.0]) / 2.0


def _build_kernels(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n % 2 == 0 or n < 3:
        raise ValueError(f"Sobel kernel size must be odd and >= 3, got {n}")
    smooth = _SMOOTH3
    diff = _DIFF3
    for _ in range((n - 3) // 2):
        smooth = conv_1d(smooth, _SMOOTH3)
        diff = conv_1d(diff, _SMOOTH3)
    return smooth, diff


def _apply(I: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    I = check_input(I)
    smooth, diff = _build_kernels(n)
    smooth = smooth.to(device=I.device, dtype=I.dtype)
    diff = diff.to(device=I.device, dtype=I.dtype)
    return separable_gradient(I, smooth_1d=smooth, diff_1d=diff)


def sobel_3(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply(I, 3)


def sobel_5(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply(I, 5)


def sobel_7(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply(I, 7)
