"""Truth masks for junction-local evaluation regions."""

from __future__ import annotations

import torch

from agfb_generators.base import Numeric, as_batch, coord_grid, infer_batch_size, infer_device


def junction_mask(
    height: int,
    width: int,
    *,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    radius_px: Numeric = 8.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return circular truth masks for junction-local scoring.

    Junction generators use this helper when a metric needs to restrict scoring
    to the neighborhood around the junction point rather than the full edge or
    arm support. The notebook also uses it to show the local evaluation region
    for L-, T-, Y-, and X-junction examples.

    `center_x` and `center_y` place the mask center in the shared centered
    coordinate system. `radius_px` controls the scoring radius in pixels.
    Scalar inputs return a single `(height, width)` boolean mask. One-dimensional
    tensor inputs return a batched `(B, height, width)` boolean mask. If `device`
    is omitted and a tensor parameter is passed, the mask stays on that tensor's
    device.
    """
    device = infer_device(device, center_x, center_y, radius_px)
    batch_size = infer_batch_size(center_x, center_y, radius_px)
    has_batched_input = _has_batched_input(center_x, center_y, radius_px)
    xx, yy = coord_grid(height, width, device, dtype)

    center_x_batch = as_batch(center_x, batch_size, device, dtype)
    center_y_batch = as_batch(center_y, batch_size, device, dtype)
    radius_batch = as_batch(radius_px, batch_size, device, dtype)

    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    mask = x_from_center * x_from_center + y_from_center * y_from_center
    mask = mask <= radius_batch * radius_batch
    if has_batched_input:
        return mask
    return mask[0]


def _has_batched_input(*params: Numeric) -> bool:
    """Return whether `junction_mask` should preserve a leading batch axis."""
    return any(isinstance(param, torch.Tensor) and param.ndim == 1 for param in params)
