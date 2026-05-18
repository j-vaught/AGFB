"""2D polynomial surface (§1.1 `polynomial`, sanity-only).

For CPGF basis recovery: a degree-`d_p` polynomial sampled on the grid is
exactly reconstructible by CPGF at degree `d >= d_p` to numerical precision.
"""

from __future__ import annotations

import torch

from cpgf_generators.base import Frame, coord_grid, pack


def polynomial(
    height: int,
    width: int,
    *,
    coeffs: torch.Tensor,
    scale: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched two-dimensional polynomial surface.

    CPGF uses this sanity-only generator to check exact polynomial basis
    recovery. Each `coeffs[b, i, j]` value is the coefficient of `x^i y^j`;
    terms with `i + j` outside the supported degree are ignored, and `scale`
    controls coordinate magnitude before the intensity and analytic gradient
    are evaluated.
    """
    if coeffs.ndim != 3:
        raise ValueError(f"coeffs must have shape (B, d_p+1, d_p+1), got {tuple(coeffs.shape)}")
    device = device or torch.device("cpu")
    coeffs = coeffs.to(device=device, dtype=dtype)
    B, D, D2 = coeffs.shape
    if D != D2:
        raise ValueError(f"coeffs second/third dims must match, got {D} vs {D2}")

    xx, yy = coord_grid(height, width, device, dtype)
    xx = xx / scale
    yy = yy / scale

    x_pows = [torch.ones_like(xx)]
    y_pows = [torch.ones_like(yy)]
    for _ in range(1, D):
        x_pows.append(x_pows[-1] * xx)
        y_pows.append(y_pows[-1] * yy)

    I = torch.zeros(B, height, width, device=device, dtype=dtype)
    gx = torch.zeros_like(I)
    gy = torch.zeros_like(I)
    for i in range(D):
        for j in range(D - i):
            c = coeffs[:, i, j].view(B, 1, 1)
            if torch.all(c == 0):
                continue
            term = x_pows[i] * y_pows[j]
            I = I + c * term
            if i > 0:
                gx = gx + (c * i / scale) * x_pows[i - 1] * y_pows[j]
            if j > 0:
                gy = gy + (c * j / scale) * x_pows[i] * y_pows[j - 1]
    return pack(I, gx, gy)
