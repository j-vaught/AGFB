"""Shared conventions and helpers for comparator filters.

Input convention:
    `I` is a `(B, H, W)` float32 tensor on any device.

Output convention:
    Every filter returns a `(g_x, g_y)` tuple of `(B, H, W)` tensors.
    Channel 0 = horizontal gradient (along `+x`, image column index).
    Channel 1 = vertical gradient (along `+y`, image row index).

Padding default is `replicate` (matches the PGF_paper prototype's Sobel
implementation). FFT-based filters use `reflect` because the FFT path needs a
boundary extension that doesn't alias.
"""

from __future__ import annotations

import torch
import torch.fft as fft
import torch.nn.functional as F


def check_input(I: torch.Tensor) -> torch.Tensor:
    """Validate and contiguous-ify a filter input."""
    if I.ndim != 3:
        raise ValueError(f"filter input must be (B, H, W), got shape {tuple(I.shape)}")
    if I.dtype != torch.float32:
        I = I.to(torch.float32)
    return I.contiguous()


def directional_derivative(
    I: torch.Tensor,
    *,
    smooth_1d: torch.Tensor,
    diff_1d: torch.Tensor,
    axis: int,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Separable directional derivative: smooth perpendicular to `axis`,
    differentiate along `axis`. Smoothing is applied first (matches the
    prototype's float32 convolution order exactly).

    `axis = 0` differentiates along rows (y, vertical, `g_y`).
    `axis = 1` differentiates along cols (x, horizontal, `g_x`).
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    r_s = smooth_1d.shape[0] // 2
    r_d = diff_1d.shape[0] // 2
    x = I.unsqueeze(1)
    if axis == 1:
        # smooth along rows (y), differentiate along cols (x)
        smooth_kernel = smooth_1d.view(1, 1, -1, 1)
        diff_kernel = diff_1d.view(1, 1, 1, -1)
        x = F.pad(x, (r_d, r_d, r_s, r_s), mode=pad_mode)
    else:
        # smooth along cols (x), differentiate along rows (y)
        smooth_kernel = smooth_1d.view(1, 1, 1, -1)
        diff_kernel = diff_1d.view(1, 1, -1, 1)
        x = F.pad(x, (r_s, r_s, r_d, r_d), mode=pad_mode)
    return F.conv2d(F.conv2d(x, smooth_kernel), diff_kernel).squeeze(1)


def separable_gradient(
    I: torch.Tensor,
    *,
    smooth_1d: torch.Tensor,
    diff_1d: torch.Tensor,
    pad_mode: str = "replicate",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both axes at once. Returns `(g_x, g_y)` with smooth-first ordering."""
    gx = directional_derivative(I, smooth_1d=smooth_1d, diff_1d=diff_1d, axis=1, pad_mode=pad_mode)
    gy = directional_derivative(I, smooth_1d=smooth_1d, diff_1d=diff_1d, axis=0, pad_mode=pad_mode)
    return gx, gy


def conv2d_dense(
    I: torch.Tensor,
    kernel: torch.Tensor,
    *,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Apply a dense 2-D kernel of shape `(kh, kw)` (odd dims) via `F.conv2d`."""
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"dense kernel dims must be odd, got {kh}x{kw}")
    rh, rw = kh // 2, kw // 2
    x = I.unsqueeze(1)
    x = F.pad(x, (rw, rw, rh, rh), mode=pad_mode)
    k = kernel.view(1, 1, kh, kw)
    return F.conv2d(x, k).squeeze(1)


def fft_xcorr(
    I: torch.Tensor,
    kernels: tuple[torch.Tensor, torch.Tensor],
    *,
    pad_mode: str = "reflect",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-correlate `I` with each of two dense kernels via `rfft2`.

    Matches the prototype FFT path: reflect-pad image to `(H + 2r, W + 2r)`,
    top-left zero-pad each kernel to the same shape, multiply by `conj(F[K])`,
    inverse-FFT, crop top-left `(H, W)`.

    Both kernels must have the same odd, square shape. Returns `(g_x, g_y)`.
    """
    K_x, K_y = kernels
    if K_x.shape != K_y.shape:
        raise ValueError(f"kernel shapes must match, got {K_x.shape} vs {K_y.shape}")
    kh, kw = K_x.shape
    if kh != kw or kh % 2 == 0:
        raise ValueError(f"FFT path requires odd square kernels, got {kh}x{kw}")
    r = kh // 2

    B, H, W = I.shape
    Hp, Wp = H + 2 * r, W + 2 * r

    x = I.unsqueeze(1)
    I_pad = F.pad(x, (r, r, r, r), mode=pad_mode).squeeze(1)
    K_x_pad = torch.zeros(Hp, Wp, dtype=I.dtype, device=I.device)
    K_y_pad = torch.zeros(Hp, Wp, dtype=I.dtype, device=I.device)
    K_x_pad[: 2 * r + 1, : 2 * r + 1] = K_x
    K_y_pad[: 2 * r + 1, : 2 * r + 1] = K_y

    FI = fft.rfft2(I_pad)
    gx_full = fft.irfft2(FI * fft.rfft2(K_x_pad).conj(), s=(Hp, Wp))
    gy_full = fft.irfft2(FI * fft.rfft2(K_y_pad).conj(), s=(Hp, Wp))
    return gx_full[..., :H, :W].contiguous(), gy_full[..., :H, :W].contiguous()


def conv_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Discrete linear convolution of two 1-D tensors (full output)."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("conv_1d requires 1-D inputs")
    out = F.conv1d(
        a.view(1, 1, -1),
        b.flip(0).view(1, 1, -1),
        padding=b.shape[0] - 1,
    )
    return out.view(-1)
