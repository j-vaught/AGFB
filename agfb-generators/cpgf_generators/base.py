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
    """One batch of generator output."""

    I: torch.Tensor  # (B, H, W)
    g: torch.Tensor  # (B, 2, H, W) — channel 0 is g_x, channel 1 is g_y

    @property
    def gx(self) -> torch.Tensor:
        return self.g[:, 0]

    @property
    def gy(self) -> torch.Tensor:
        return self.g[:, 1]

    @property
    def batch_size(self) -> int:
        return int(self.I.shape[0])

    @property
    def height(self) -> int:
        return int(self.I.shape[1])

    @property
    def width(self) -> int:
        return int(self.I.shape[2])


def coord_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered pixel coordinates of shape (H, W) each.

    `xx[i, j]` is the horizontal offset of pixel (i, j) from the image center;
    `yy[i, j]` is the vertical offset. Both have the same shape `(H, W)`.
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
    """Broadcast `value` to a `(B, 1, 1)` tensor on the target device.

    Accepts a Python scalar (broadcast across the batch) or a 1-D tensor of
    length `B`. Anything else raises `ValueError`.
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
    """Standard-normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * u * u)


def gauss_Phi(u: torch.Tensor) -> torch.Tensor:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))


def infer_batch_size(*params: Numeric) -> int:
    """Pick the batch size from the first 1-D tensor parameter; default 1."""
    for p in params:
        if isinstance(p, torch.Tensor) and p.ndim == 1:
            return int(p.shape[0])
    return 1


def pack(I: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> Frame:
    """Stack intensity and the two gradient channels into a `Frame`."""
    g = torch.stack((gx, gy), dim=1)
    return Frame(I=I.contiguous(), g=g.contiguous())
