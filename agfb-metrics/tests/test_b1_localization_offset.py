"""Tests for B.1 localization offset."""

from __future__ import annotations

import math

import pytest
import torch

from cpgf_metrics.b1_localization_offset import b1_localization_offset


def _gaussian_step_gx(H: int, W: int, sigma: float, x0: float) -> torch.Tensor:
    """gx field of a vertical smoothed step at x = x0 (gy = 0)."""
    xs = torch.arange(W, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    row = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    return row.unsqueeze(0).expand(H, W).unsqueeze(0)


def _signal_mask(gx: torch.Tensor) -> torch.Tensor:
    return gx.abs() > 1e-3 * float(gx.abs().max())


def test_zero_offset_when_filter_matches_truth() -> None:
    H = W = 64
    gx_t = _gaussian_step_gx(H, W, sigma=2.0, x0=32.0)
    gy_t = torch.zeros_like(gx_t)
    out = b1_localization_offset(gx_t, gy_t, gx_t, gy_t, _signal_mask(gx_t))
    assert out[0].item() == pytest.approx(0.0, abs=0.1)


def test_known_offset_when_filter_is_shifted_step() -> None:
    """If the filter output is a step shifted by 2 pixels, every cross-edge
    profile peaks at t=+2 (filter peak is 2 px to the right of the true
    edge), so the metric should report ≈ 2 pixels."""
    H = W = 96
    gx_t = _gaussian_step_gx(H, W, sigma=2.0, x0=48.0)
    gx_f = _gaussian_step_gx(H, W, sigma=2.0, x0=50.0)
    gy = torch.zeros_like(gx_t)
    out = b1_localization_offset(gx_f, gy, gx_t, gy, _signal_mask(gx_t))
    assert out[0].item() == pytest.approx(2.0, abs=0.15)
