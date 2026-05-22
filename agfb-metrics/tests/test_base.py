"""Tests for agfb_metrics.metrics.base helpers."""

from __future__ import annotations

import torch

from agfb_metrics.metrics.base import (
    magnitude,
    masks,
    ridge_mask_from_truth,
    unit_normal_from_truth,
)


def test_magnitude_basic() -> None:
    gx = torch.tensor([[[3.0, 0.0], [0.0, 0.0]]])
    gy = torch.tensor([[[4.0, 0.0], [0.0, 1.0]]])
    m = magnitude(gx, gy)
    assert torch.allclose(m, torch.tensor([[[5.0, 0.0], [0.0, 1.0]]]))


def test_masks_signal_and_flat_partition() -> None:
    torch.manual_seed(0)
    H = W = 32
    gx = torch.zeros(2, H, W)
    gy = torch.zeros(2, H, W)
    gx[0, 16, 8:24] = 1.0
    gy[1, 8:24, 16] = 1.0
    out = masks(gx, gy)
    assert out["signal"].shape == (2, H, W)
    assert out["flat"].shape == (2, H, W)
    assert not (out["signal"] & out["flat"]).any()
    assert out["signal"][0, 16, 8:24].all()
    assert not out["flat"][0, 16, 8:24].any()


def test_masks_flat_eroded_inward() -> None:
    gx = torch.zeros(1, 32, 32)
    gy = torch.zeros(1, 32, 32)
    gx[0, 16, :] = 1.0
    out = masks(gx, gy, dilate_px=3)
    assert not out["flat"][0, 13:20, 5].any()
    assert out["flat"][0, 5, 5]
    assert out["flat"][0, 27, 27]


def test_ridge_mask_is_thin_subset_of_signal() -> None:
    """For a 1-D smoothed step along x, the ridge mask should be a single
    column (where |grad| peaks along the x-axis = normal direction) inside
    a wider signal band."""
    import math

    H, W = 32, 64
    sigma = 2.0
    x0 = 30.0
    xs = torch.arange(W, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    row = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    gx_t = row.unsqueeze(0).expand(H, W).unsqueeze(0)
    gy_t = torch.zeros_like(gx_t)
    signal = gx_t.abs() > 1e-3 * float(gx_t.abs().max())

    ridge = ridge_mask_from_truth(gx_t, gy_t, signal)
    # Ridge must be a (proper) subset of the signal band.
    assert (ridge & ~signal).sum() == 0
    assert int(ridge.sum()) < int(signal.sum())
    # The ridge column is exactly where the gaussian peaks.
    ridge_cols = ridge[0].nonzero(as_tuple=False)[:, 1].unique()
    assert ridge_cols.shape[0] == 1
    assert int(ridge_cols[0]) == int(round(x0))


def test_ridge_mask_empty_when_signal_empty() -> None:
    gx_t = torch.zeros(1, 8, 8)
    gy_t = torch.zeros(1, 8, 8)
    signal = torch.zeros(1, 8, 8, dtype=torch.bool)
    ridge = ridge_mask_from_truth(gx_t, gy_t, signal)
    assert ridge.sum() == 0


def test_unit_normal_unit_length_on_signal() -> None:
    gx = torch.tensor([[[3.0, 0.0], [0.0, 2.0]]])
    gy = torch.tensor([[[4.0, 0.0], [5.0, 0.0]]])
    nx, ny = unit_normal_from_truth(gx, gy)
    mag_n = torch.sqrt(nx * nx + ny * ny)
    assert torch.allclose(mag_n[0, 0, 0], torch.tensor(1.0))
    assert torch.allclose(mag_n[0, 1, 1], torch.tensor(1.0))
    assert mag_n[0, 0, 1] == 0.0
