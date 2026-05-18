"""Composite frame rendered from disjoint rectangular regions.

The general partition is arbitrary in the spec; here we support a list of
axis-aligned rectangles, each owned by one component generator. Pixels outside
every rectangle are filled with a default `Frame` of zeros. Pixels within
`r_j = 3` of any boundary are returned in a junction mask of shape `(H, W)`.

Batched composites are not supported in v1. Assembling them per-frame keeps the
API simple and avoids a tensor-of-rectangles interface. Pass a
`list[CompositeRect]` per call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agfb_generators.base import Frame, coord_grid


@dataclass(frozen=True)
class CompositeRect:
    """Describe one rectangular region in a composite AGFB frame.

    `composite` consumes these records to copy pixels from pre-rendered
    component frames into an axis-aligned partition. The `frame` is rendered at
    full image size with batch size one, while the integer bounds select the
    rows and columns that adopt that component.
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
    """Assemble one AGFB composite frame and its junction mask.

    AGFB uses composites to combine disjoint component generators in one image.
    The function copies each `CompositeRect` into a zero-filled frame, rejects
    out-of-bounds or batched components, and returns the assembled `Frame`
    plus a boolean `(H, W)` mask around component boundaries.
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
    """Mark pixels near ownership changes in a composite partition.

    `composite` uses this private helper after filling the owner map. It first
    marks cells with a 4-connected neighbor from a different component, then
    dilates that set by a Chebyshev `radius` with max pooling.
    """
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
