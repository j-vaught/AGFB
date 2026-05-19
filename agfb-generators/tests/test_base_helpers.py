"""Tests for shared generator helpers."""

from __future__ import annotations

import torch

from agfb_generators.base import coord_grid


def test_coord_grid_reuses_cached_tensor_for_same_render_key() -> None:
    """Check that repeated renders can reuse the same coordinate grid tensors."""
    device = torch.device("cpu")
    first_x, first_y = coord_grid(12, 14, device, torch.float32)
    second_x, second_y = coord_grid(12, 14, device, torch.float32)

    assert first_x is second_x
    assert first_y is second_y
    assert first_x.shape == (12, 14)
    assert first_y.shape == (12, 14)


def test_coord_grid_cache_is_separated_by_dtype() -> None:
    """Check that dtype remains part of the coordinate-grid cache key."""
    device = torch.device("cpu")
    float_x, _ = coord_grid(8, 9, device, torch.float32)
    double_x, _ = coord_grid(8, 9, device, torch.float64)

    assert float_x.dtype == torch.float32
    assert double_x.dtype == torch.float64
