"""Tests for NRMSE on edge pixels."""

from __future__ import annotations

import math

import pytest
import torch

from agfb_metrics.nrmse import nrmse


def test_zero_error_when_filter_matches_truth() -> None:
    torch.manual_seed(0)
    gx_t = torch.randn(2, 16, 16)
    gy_t = torch.randn(2, 16, 16)
    mask = torch.ones(2, 16, 16, dtype=torch.bool)
    out = nrmse(gx_t, gy_t, gx_t, gy_t, mask)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_by_hand_value() -> None:
    gx = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    gy = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    gx_t = torch.tensor([[[2.0, 2.0], [3.0, 5.0]]])
    gy_t = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)

    expected_num = math.sqrt(((1 - 2) ** 2 + 0 + 0 + (4 - 5) ** 2) / 4)
    expected_den = (2 + 2 + 3 + 5) / 4
    expected = expected_num / expected_den

    out = nrmse(gx, gy, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(expected, rel=1e-6)


def test_empty_mask_returns_nan() -> None:
    gx = torch.ones(1, 4, 4)
    gy = torch.ones(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = nrmse(gx, gy, gx, gy, mask)
    assert torch.isnan(out[0])
