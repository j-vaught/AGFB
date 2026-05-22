"""Tests for 99th-percentile spurious gradient."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agfb_metrics.tail_spurious_grad import tail_spurious_grad


def test_matches_numpy_percentile() -> None:
    torch.manual_seed(0)
    H = W = 64
    mag = torch.rand(H, W) * 3.0
    gx = mag.unsqueeze(0)
    gy = torch.zeros_like(gx)
    mask = torch.ones(1, H, W, dtype=torch.bool)
    out = tail_spurious_grad(gx, gy, mask)
    expected = float(np.percentile(mag.numpy(), 99.0))
    assert out[0].item() == pytest.approx(expected, rel=1e-4)


def test_zero_when_filter_is_zero() -> None:
    gx = torch.zeros(1, 32, 32)
    gy = torch.zeros(1, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)
    out = tail_spurious_grad(gx, gy, mask)
    assert out[0].item() == 0.0


def test_empty_mask_returns_nan() -> None:
    gx = torch.ones(1, 4, 4)
    gy = torch.ones(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = tail_spurious_grad(gx, gy, mask)
    assert torch.isnan(out[0])
