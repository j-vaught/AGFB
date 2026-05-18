"""Truth masks for junction-local evaluation regions."""

from __future__ import annotations

import torch

from agfb_generators.base import Numeric, as_batch, coord_grid, infer_batch_size


def junction_mask(
    height: int,
    width: int,
    *,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    radius_px: Numeric = 8.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a boolean mask for pixels within a radius of a junction center."""
    device = device or torch.device("cpu")
    B = infer_batch_size(x0, y0, radius_px)
    if B != 1:
        raise ValueError("junction_mask expects scalar center and radius parameters")
    xx, yy = coord_grid(height, width, device, dtype)

    x0_b = as_batch(x0, B, device, dtype)
    y0_b = as_batch(y0, B, device, dtype)
    r = as_batch(radius_px, B, device, dtype)

    dx = xx - x0_b
    dy = yy - y0_b
    return (dx * dx + dy * dy <= r * r)[0]
