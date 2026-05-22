"""Tests for A.2 angular MAE."""

from __future__ import annotations

import pytest
import torch

from agfb_metrics.a2_angular_mae import a2_angular_mae


def test_zero_degrees_when_aligned() -> None:
    gx_t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    gy_t = torch.tensor([[[0.5, 0.0], [-1.0, 2.0]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    out = a2_angular_mae(2.0 * gx_t, 2.0 * gy_t, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(0.0, abs=1e-3)


def test_ninety_degrees_when_perpendicular() -> None:
    gx_t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    gy_t = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    gx = torch.zeros_like(gx_t)
    gy = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    out = a2_angular_mae(gx, gy, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(90.0, abs=1e-3)


def test_one_eighty_degrees_when_antiparallel() -> None:
    gx_t = torch.tensor([[[1.0, 2.0]]])
    gy_t = torch.tensor([[[3.0, 4.0]]])
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    out = a2_angular_mae(-gx_t, -gy_t, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(180.0, abs=1e-2)


def test_skips_degenerate_pixels() -> None:
    gx_t = torch.tensor([[[1.0, 0.0]]])
    gy_t = torch.tensor([[[0.0, 0.0]]])
    gx = torch.tensor([[[0.0, 5.0]]])
    gy = torch.tensor([[[1.0, 5.0]]])
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    out = a2_angular_mae(gx, gy, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(90.0, abs=1e-3)


def test_empty_mask_returns_nan() -> None:
    gx = torch.zeros(1, 4, 4)
    gy = torch.zeros(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = a2_angular_mae(gx, gy, gx, gy, mask)
    assert torch.isnan(out[0])
