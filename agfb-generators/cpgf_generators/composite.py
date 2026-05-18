"""Composite scene rendered from disjoint rectangular regions (§1.1 `composite`).

The general partition is arbitrary in the spec; here we support a list of
axis-aligned rectangles, each owned by one component generator. Pixels outside
every rectangle are filled with a default `Frame` of zeros. Pixels within
`r_j = 3` of any boundary are returned in a junction mask of shape `(H, W)`.

Batched composites are not supported in v1 — composites are rare (3 × 8 = 24
frames total in production) and assembling them per-frame is cheaper than
designing a tensor-of-rectangles API. Pass a `list[CompositeRect]` per call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cpgf_generators.base import Frame, coord_grid


@dataclass(frozen=True)
class CompositeRect:
    """One rectangular component region of a composite frame.

    `frame` must already be rendered at `(1, H, W)` for the full image; the
    rectangle defines which pixels actually adopt its values. Coordinates are
    integer pixel indices; the rectangle is `[row_lo:row_hi, col_lo:col_hi]`.
    """

    row_lo: int
    row_hi: int
    col_lo: int
    col_hi: int
    frame: Frame


def composite(
    height: int,
    width: int,
    rects: list[CompositeRect],
    *,
    junction_radius: int = 3,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Frame, torch.Tensor]:
    """Assemble a single composite frame.

    Returns a `Frame` of batch size 1 plus a boolean junction mask `(H, W)`.
    """
    device = device or torch.device("cpu")
    I = torch.zeros(1, height, width, device=device, dtype=dtype)
    gx = torch.zeros_like(I)
    gy = torch.zeros_like(I)
    owner = torch.full((height, width), -1, device=device, dtype=torch.int32)

    for idx, r in enumerate(rects):
        if r.row_lo < 0 or r.col_lo < 0 or r.row_hi > height or r.col_hi > width:
            raise ValueError(f"rect #{idx} {r} is out of bounds for {height}x{width}")
        if r.frame.batch_size != 1:
            raise ValueError(f"rect #{idx} frame must have batch_size 1, got {r.frame.batch_size}")
        rs, re = r.row_lo, r.row_hi
        cs, ce = r.col_lo, r.col_hi
        I[:, rs:re, cs:ce] = r.frame.I[:, rs:re, cs:ce]
        gx[:, rs:re, cs:ce] = r.frame.gx[:, rs:re, cs:ce]
        gy[:, rs:re, cs:ce] = r.frame.gy[:, rs:re, cs:ce]
        owner[rs:re, cs:ce] = idx

    junction = _boundary_mask(owner, radius=junction_radius)
    xx, _ = coord_grid(height, width, device, dtype)
    del xx
    return Frame(I=I.contiguous(), g=torch.stack((gx, gy), dim=1).contiguous()), junction


def _boundary_mask(owner: torch.Tensor, *, radius: int) -> torch.Tensor:
    """Pixels within `radius` (Chebyshev) of any cell whose neighbor has a
    different owner. Computed with a single max-pool dilation."""
    if owner.ndim != 2:
        raise ValueError(f"owner must be 2-D, got shape {tuple(owner.shape)}")
    h, w = owner.shape
    o = owner
    diff = torch.zeros(h, w, dtype=torch.bool, device=o.device)
    diff[:-1, :] |= o[:-1, :] != o[1:, :]
    diff[1:, :] |= o[1:, :] != o[:-1, :]
    diff[:, :-1] |= o[:, :-1] != o[:, 1:]
    diff[:, 1:] |= o[:, 1:] != o[:, :-1]

    if radius <= 0:
        return diff
    k = 2 * radius + 1
    d_f = diff.float().unsqueeze(0).unsqueeze(0)
    dilated = torch.nn.functional.max_pool2d(d_f, kernel_size=k, stride=1, padding=radius)
    return dilated.squeeze(0).squeeze(0) > 0.5
