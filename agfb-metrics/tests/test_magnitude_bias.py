"""Tests for magnitude bias."""

from __future__ import annotations

import pytest
import torch

from agfb_metrics.metrics.magnitude_bias import magnitude_bias


def test_zero_bias_when_filter_matches_truth() -> None:
    torch.manual_seed(0)
    gx_t = torch.randn(2, 16, 16)
    gy_t = torch.randn(2, 16, 16)
    mask = torch.ones(2, 16, 16, dtype=torch.bool)
    out = magnitude_bias(gx_t, gy_t, gx_t, gy_t, mask)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_negative_when_under_reading() -> None:
    torch.manual_seed(1)
    gx_t = torch.randn(1, 8, 8).abs() + 0.1
    gy_t = torch.zeros_like(gx_t)
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    out = magnitude_bias(0.5 * gx_t, 0.5 * gy_t, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(-0.5, abs=1e-5)


def test_positive_when_over_reading() -> None:
    torch.manual_seed(2)
    gx_t = torch.randn(1, 8, 8).abs() + 0.1
    gy_t = torch.zeros_like(gx_t)
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    out = magnitude_bias(1.5 * gx_t, 1.5 * gy_t, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(0.5, abs=1e-5)


def test_empty_mask_returns_nan() -> None:
    gx = torch.ones(1, 4, 4)
    gy = torch.ones(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = magnitude_bias(gx, gy, gx, gy, mask)
    assert torch.isnan(out[0])
