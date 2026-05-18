"""Derivative-of-Gaussian (DoG).

Separable: smooth along one axis with `G(t; sigma)`, differentiate along the
other with `G'(t; sigma) = -t/sigma^2 * G(t; sigma)`. Kernel half-width is
`ceil(truncate * sigma)`; default truncation is 4σ.

Despite the name, this is the first-derivative-of-Gaussian operator (also
called "DroG" or "GaussianGradient" in some libraries), not the bandpass
difference of two Gaussians. The spec calls it `DoG` and that is the name
preserved here.
"""

from __future__ import annotations

import math

import torch

from cpgf_filters.base import check_input, separable_gradient


class DoG:
    """Holds prebuilt 1-D smoothing and derivative kernels for a given σ."""

    def __init__(self, sigma: float, truncate: float = 4.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        r = max(1, int(math.ceil(truncate * sigma)))
        t = torch.arange(-r, r + 1, dtype=torch.float64)
        g = torch.exp(-0.5 * (t / sigma) ** 2)
        g = g / g.sum()
        d = -(t / (sigma**2)) * g
        d = d - d.mean()  # remove residual DC from truncation
        self.sigma = float(sigma)
        self.radius = r
        self.smooth = g.to(torch.float32)
        self.diff = d.to(torch.float32)

    def apply(self, I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        I = check_input(I)
        smooth = self.smooth.to(device=I.device, dtype=I.dtype)
        diff = self.diff.to(device=I.device, dtype=I.dtype)
        return separable_gradient(I, smooth_1d=smooth, diff_1d=diff)
