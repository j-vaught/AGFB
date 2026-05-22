"""Tests for the shared cross-edge profile sampler."""

from __future__ import annotations

import math

import pytest
import torch

from agfb_metrics.metrics._cross_edge_profile import cross_edge_profile
from agfb_metrics.metrics.base import magnitude


def _synth_step_field(H: int = 64, W: int = 64, sigma: float = 2.0, x0: float = 32.0):
    """A 1-D smoothed step along x: gx = exp(-(x-x0)^2 / 2sigma^2) / (sigma*sqrt(2pi))
    so the gradient peaks at x = x0. Returns (gx, gy, signal_mask)."""
    xs = torch.arange(W, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    gx_row = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    gx = gx_row.unsqueeze(0).expand(H, W).unsqueeze(0)  # (1, H, W)
    gy = torch.zeros_like(gx)
    mag = gx.abs()
    signal = mag > 1e-3 * float(mag.max())
    return gx, gy, signal


def test_profile_shape_and_t_axis() -> None:
    gx, gy, signal = _synth_step_field()
    profiles, t, t0 = cross_edge_profile(magnitude(gx, gy), gx, gy, signal, r_max=8.0, step=0.5)
    assert len(profiles) == 1
    K = profiles[0].shape[1]
    assert t.shape[0] == K
    assert t[t0].item() == pytest.approx(0.0, abs=1e-6)
    assert t[0].item() == pytest.approx(-8.0, abs=1e-6)
    assert t[-1].item() == pytest.approx(8.0, abs=1e-6)


def test_filter_and_truth_argmax_agree_when_equal() -> None:
    """If the sampled field equals the truth field, every per-pixel profile
    peaks at the *same* sample position as the truth's profile at that
    pixel (the foot of the perpendicular from p to the edge crest)."""
    gx, gy, signal = _synth_step_field(H=32, W=64, sigma=2.0, x0=30.0)
    mag = magnitude(gx, gy)
    filt_profiles, _, _ = cross_edge_profile(mag, gx, gy, signal)
    true_profiles, _, _ = cross_edge_profile(mag, gx, gy, signal)
    assert torch.equal(filt_profiles[0], true_profiles[0])
    assert torch.equal(
        torch.argmax(filt_profiles[0], dim=1),
        torch.argmax(true_profiles[0], dim=1),
    )


def test_empty_signal_mask_returns_empty_profile() -> None:
    gx = torch.zeros(1, 8, 8)
    gy = torch.zeros(1, 8, 8)
    signal = torch.zeros(1, 8, 8, dtype=torch.bool)
    profiles, t, _ = cross_edge_profile(magnitude(gx, gy), gx, gy, signal)
    assert profiles[0].shape[0] == 0
    assert profiles[0].shape[1] == t.shape[0]
