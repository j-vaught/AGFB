"""Tests for 95th-percentile gradient-vector error."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agfb_metrics.metrics.tail_vector_error import tail_vector_error


def test_matches_numpy_percentile_on_uniform_errors() -> None:
    torch.manual_seed(0)
    H = W = 64
    g_t = torch.zeros(1, H, W)
    err = torch.rand(H, W) * 2.0
    gx = err.unsqueeze(0)
    gy = torch.zeros_like(gx)
    mask = torch.ones(1, H, W, dtype=torch.bool)
    out = tail_vector_error(gx, gy, g_t, g_t, mask)
    expected = float(np.percentile(err.numpy(), 95.0))
    assert out[0].item() == pytest.approx(expected, rel=1e-4)


def test_matches_torch_quantile_without_mask() -> None:
    torch.manual_seed(1)
    H = W = 32
    gx_t = torch.zeros(1, H, W)
    gy_t = torch.zeros_like(gx_t)
    gx = torch.rand(1, H, W)
    gy = torch.rand(1, H, W)

    out = tail_vector_error(gx, gy, gx_t, gy_t, signal_mask=None, q=0.95)
    err_mag = torch.sqrt(gx * gx + gy * gy)
    expected = torch.quantile(err_mag.reshape(1, -1), 0.95, dim=1)
    assert torch.allclose(out, expected)


def test_zero_error_returns_zero_p95() -> None:
    gx = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    gy = torch.tensor([[[0.5, 0.5], [-1.0, 2.0]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    out = tail_vector_error(gx, gy, gx, gy, mask)
    assert out[0].item() == pytest.approx(0.0, abs=1e-6)


def test_empty_mask_returns_nan() -> None:
    gx = torch.ones(1, 4, 4)
    gy = torch.ones(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = tail_vector_error(gx, gy, gx, gy, mask)
    assert torch.isnan(out[0])
