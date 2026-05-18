"""Shared types and helpers for batched generators.

Conventions:
    Intensity tensor shape: (B, H, W), float32.
    Gradient tensor shape:  (B, 2, H, W), float32, channel order (g_x, g_y).
    Coordinates:            origin at the image center, +x right, +y down.

Every public generator function accepts scalar Python numbers OR 1-D tensors of
length B for its parameters. Scalars broadcast across the batch; tensors are
moved to the target device and reshaped to `(B, 1, 1)` before use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

Numeric = float | int | torch.Tensor


@dataclass(frozen=True)
class Frame:
    """Container for one rendered generator batch.

    Every CPGF generator returns a `Frame` so benchmark code, diagnostics, and
    tests can keep the intensity image and analytic ground-truth gradients
    together with a shared shape convention.
    """

    I: torch.Tensor  # (B, H, W) - image intensity.
    g: torch.Tensor  # (B, 2, H, W) - channel 0 is g_x, channel 1 is g_y.

    @property
    def gx(self) -> torch.Tensor:
        """Return the horizontal gradient channel.

        CPGF metrics and regression tests use this as the analytic `g_x`
        reference for a rendered frame.
        """
        return self.g[:, 0]

    @property
    def gy(self) -> torch.Tensor:
        """Return the vertical gradient channel.

        CPGF metrics and regression tests use this as the analytic `g_y`
        reference for a rendered frame.
        """
        return self.g[:, 1]

    @property
    def batch_size(self) -> int:
        """Return the number of images carried by this frame.

        Composite assembly and tests use this to distinguish scalar renders
        from batched generator output.
        """
        return int(self.I.shape[0])

    @property
    def height(self) -> int:
        """Return the image height in pixels.

        Callers use this with `width` when validating frame shape against a
        requested CPGF render size.
        """
        return int(self.I.shape[1])

    @property
    def width(self) -> int:
        """Return the image width in pixels.

        Callers use this with `height` when validating frame shape against a
        requested CPGF render size.
        """
        return int(self.I.shape[2])


def coord_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered pixel coordinate grids for a CPGF render.

    Generator functions use `xx` and `yy` to evaluate analytic intensity and
    gradient formulas on the shared convention where the origin is the image
    center, `+x` points right, and `+y` points down.
    """
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    ys = torch.arange(height, device=device, dtype=dtype) - cy
    xs = torch.arange(width, device=device, dtype=dtype) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def as_batch(
    value: Numeric,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert one scalar generator parameter into a broadcast batch tensor.

    Public CPGF generators use this after `infer_batch_size` so Python scalars
    and 1-D tensor parameters share the `(B, 1, 1)` shape needed for vectorized
    PyTorch formulas.
    """
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"parameter tensor must be scalar or shape ({batch_size},), got {tuple(t.shape)}"
            )
    else:
        t = torch.full((batch_size,), float(value), device=device, dtype=dtype)
    return t.view(batch_size, 1, 1)


def gauss_phi(u: torch.Tensor) -> torch.Tensor:
    """Evaluate the standard-normal probability density function.

    Smoothed edge and curved-arc generators use this as the closed-form
    derivative of the Gaussian cumulative distribution function.
    """
    return (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * u * u)


def gauss_Phi(u: torch.Tensor) -> torch.Tensor:
    """Evaluate the standard-normal cumulative distribution function.

    CPGF edge-like generators use this to turn signed distance fields into
    smoothly band-limited intensity transitions.
    """
    return 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))


def infer_batch_size(*params: Numeric) -> int:
    """Infer the render batch size from generator parameters.

    Public generators use the first 1-D tensor parameter as `B`; scalar-only
    calls default to one frame so the same code path handles scalar and batched
    CPGF scenes.
    """
    for p in params:
        if isinstance(p, torch.Tensor) and p.ndim == 1:
            return int(p.shape[0])
    return 1


def pack(I: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> Frame:
    """Package intensity and gradient tensors into the shared `Frame` type.

    Generators call this at return time so downstream CPGF metrics, diagnostics,
    and tests always receive contiguous tensors in the same channel order.
    """
    g = torch.stack((gx, gy), dim=1)
    return Frame(I=I.contiguous(), g=g.contiguous())
