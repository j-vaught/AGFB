"""Tests for cpgf_metrics.base helpers."""

from __future__ import annotations

import torch

from cpgf_metrics.base import magnitude, masks, unit_normal_from_truth


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


def test_unit_normal_unit_length_on_signal() -> None:
    gx = torch.tensor([[[3.0, 0.0], [0.0, 2.0]]])
    gy = torch.tensor([[[4.0, 0.0], [5.0, 0.0]]])
    nx, ny = unit_normal_from_truth(gx, gy)
    mag_n = torch.sqrt(nx * nx + ny * ny)
    assert torch.allclose(mag_n[0, 0, 0], torch.tensor(1.0))
    assert torch.allclose(mag_n[0, 1, 1], torch.tensor(1.0))
    assert mag_n[0, 0, 1] == 0.0
