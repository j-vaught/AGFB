"""Tests for localization offset."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from agfb_metrics.metrics.localization_offset import localization_offset


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
    out = localization_offset(gx_t, gy_t, gx_t, gy_t, _signal_mask(gx_t))
    assert out[0].item() == pytest.approx(0.0, abs=0.1)


def test_known_offset_when_filter_is_shifted_step() -> None:
    """If the filter output is a step shifted by 2 pixels, every cross-edge
    profile peaks at t=+2 (filter peak is 2 px to the right of the true
    edge), so the metric should report about 2 pixels."""
    H = W = 96
    gx_t = _gaussian_step_gx(H, W, sigma=2.0, x0=48.0)
    gx_f = _gaussian_step_gx(H, W, sigma=2.0, x0=50.0)
    gy = torch.zeros_like(gx_t)
    out = localization_offset(gx_f, gy, gx_t, gy, _signal_mask(gx_t))
    assert out[0].item() == pytest.approx(2.0, abs=0.15)


def test_ridge_mode_zero_when_filter_matches_truth() -> None:
    H = W = 64
    gx_t = _gaussian_step_gx(H, W, sigma=2.0, x0=32.0)
    gy_t = torch.zeros_like(gx_t)
    out = localization_offset(gx_t, gy_t, gx_t, gy_t, _signal_mask(gx_t), mode="ridge")
    assert out[0].item() == pytest.approx(0.0, abs=0.1)


def test_ridge_mode_matches_truth_anchored_on_shifted_step() -> None:
    """For a pure pixel shift, both modes report the same number - the
    shift is constant across the whole band, so the truth-anchoring just
    subtracts a different per-pixel offset (and ridge mode has the truth
    offset = 0 by construction). Both should land near 2.0 px."""
    H = W = 96
    gx_t = _gaussian_step_gx(H, W, sigma=2.0, x0=48.0)
    gx_f = _gaussian_step_gx(H, W, sigma=2.0, x0=50.0)
    gy = torch.zeros_like(gx_t)
    anchored = localization_offset(gx_f, gy, gx_t, gy, _signal_mask(gx_t))[0].item()
    ridge = localization_offset(gx_f, gy, gx_t, gy, _signal_mask(gx_t), mode="ridge")[0].item()
    assert anchored == pytest.approx(ridge, abs=0.1)


def test_ridge_mode_rejects_bad_mode_string() -> None:
    gx = torch.zeros(1, 4, 4)
    gy = torch.zeros(1, 4, 4)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    bogus_mode: Any = "bogus"
    with pytest.raises(ValueError):
        localization_offset(gx, gy, gx, gy, mask, mode=bogus_mode)
