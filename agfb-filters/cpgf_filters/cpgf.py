"""Circular Polynomial Gradient Filter (CPGF).

Polynomial least-squares fit of degree `d` over a discrete disc of radius `r`,
returning the `(a_10, a_01)` linear coefficients as `(g_y, g_x)`.

Kernel construction is identical to the PGF_paper prototype:
    `cpgf_kernels(r, d)` builds the disc support, the monomial design matrix,
    solves `(X^T X) coef = X^T`, and packs the rows for `(0, 1)` (column
    derivative → g_x) and `(1, 0)` (row derivative → g_y) into 2-D kernels.

Application uses the shared FFT cross-correlation path: `F[I] · conj(F[K])`.
"""

from __future__ import annotations

import torch

from cpgf_filters.base import check_input, fft_xcorr


def cpgf_kernels(
    r: int, d: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CPGF cross-correlation kernels.

    Returns `(K_x, K_y)` each of shape `(2r+1, 2r+1)`, float32 on `device`.
    `K_x` is the polynomial weight extracting the column-direction gradient
    (linear in column offset `v`); `K_y` extracts the row-direction gradient.
    """
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    monos = [(i, j) for i in range(d + 1) for j in range(d + 1 - i)]
    coords = [(u, v) for u in range(-r, r + 1) for v in range(-r, r + 1) if u * u + v * v <= r * r]
    N, M = len(coords), len(monos)
    X = torch.zeros(N, M, dtype=torch.float64)
    for n, (u, v) in enumerate(coords):
        for m, (i, j) in enumerate(monos):
            X[n, m] = (float(u) ** i) * (float(v) ** j)
    A = X.t() @ X
    coef = torch.linalg.solve(A, X.t())
    idx_col_linear = monos.index((0, 1))  # g_x: linear in column offset v
    idx_row_linear = monos.index((1, 0))  # g_y: linear in row offset u
    K_x = torch.zeros(2 * r + 1, 2 * r + 1, dtype=torch.float64)
    K_y = torch.zeros(2 * r + 1, 2 * r + 1, dtype=torch.float64)
    for n, (u, v) in enumerate(coords):
        K_x[u + r, v + r] = coef[idx_col_linear, n]
        K_y[u + r, v + r] = coef[idx_row_linear, n]
    K_x = K_x.to(dtype=torch.float32)
    K_y = K_y.to(dtype=torch.float32)
    if device is not None:
        K_x = K_x.to(device=device)
        K_y = K_y.to(device=device)
    return K_x, K_y


class CPGF:
    """Holds prebuilt CPGF kernels for one `(r, d)` configuration."""

    def __init__(self, r: int, d: int, device: torch.device | None = None) -> None:
        self.r = int(r)
        self.d = int(d)
        self.K_x, self.K_y = cpgf_kernels(r, d, device=device)

    def apply(self, I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        I = check_input(I)
        K_x = self.K_x.to(device=I.device, dtype=I.dtype)
        K_y = self.K_y.to(device=I.device, dtype=I.dtype)
        return fft_xcorr(I, (K_x, K_y), pad_mode="reflect")
