"""Tests for background noise gain."""

from __future__ import annotations

import math

import pytest
import torch

from agfb_metrics.noise_gain import noise_gain


def test_half_normal_gain_matches_theory() -> None:
    """If gx is N(0, sigma^2) and gy is zero, |grad| = |gx| is half-normal with
    std = sigma * sqrt(1 - 2/pi), so noise gain = sqrt(1 - 2/pi) about 0.603."""
    torch.manual_seed(0)
    sigma_n = 0.7
    H = W = 256
    gx = sigma_n * torch.randn(1, H, W)
    gy = torch.zeros_like(gx)
    mask = torch.ones(1, H, W, dtype=torch.bool)
    out = noise_gain(gx, gy, mask, sigma_n=sigma_n)
    expected = math.sqrt(1.0 - 2.0 / math.pi)
    assert out[0].item() == pytest.approx(expected, abs=0.02)


def test_rayleigh_gain_matches_theory() -> None:
    """If both gx and gy are N(0, sigma^2), |grad| is Rayleigh-distributed with
    std = sigma * sqrt(2 - pi/2), so noise gain = sqrt(2 - pi/2) about 0.655."""
    torch.manual_seed(1)
    sigma_n = 1.3
    H = W = 256
    gx = sigma_n * torch.randn(1, H, W)
    gy = sigma_n * torch.randn(1, H, W)
    mask = torch.ones(1, H, W, dtype=torch.bool)
    out = noise_gain(gx, gy, mask, sigma_n=sigma_n)
    expected = math.sqrt(2.0 - math.pi / 2.0)
    assert out[0].item() == pytest.approx(expected, abs=0.02)


def test_zero_when_filter_is_zero() -> None:
    gx = torch.zeros(1, 32, 32)
    gy = torch.zeros(1, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)
    out = noise_gain(gx, gy, mask, sigma_n=1.0)
    assert out[0].item() == 0.0


def test_rejects_invalid_sigma() -> None:
    gx = torch.zeros(1, 4, 4)
    gy = torch.zeros(1, 4, 4)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        noise_gain(gx, gy, mask, sigma_n=0.0)
    with pytest.raises(ValueError):
        noise_gain(gx, gy, mask, sigma_n=-1.0)


def test_empty_mask_returns_nan() -> None:
    gx = torch.ones(1, 4, 4)
    gy = torch.ones(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = noise_gain(gx, gy, mask, sigma_n=1.0)
    assert torch.isnan(out[0])
