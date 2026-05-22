"""Tests for edge FWHM."""

from __future__ import annotations

import math

import pytest
import torch

from agfb_metrics.edge_fwhm import edge_fwhm


def _gaussian_step_gx(H: int, W: int, sigma: float, x0: float) -> torch.Tensor:
    xs = torch.arange(W, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    row = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    return row.unsqueeze(0).expand(H, W).unsqueeze(0)


def _signal_mask(gx: torch.Tensor) -> torch.Tensor:
    return gx.abs() > 1e-3 * float(gx.abs().max())


def test_fwhm_matches_gaussian_theory() -> None:
    """The FWHM of a unit Gaussian is sigma * 2*sqrt(2*ln 2)."""
    H = W = 96
    sigma = 2.0
    gx_t = _gaussian_step_gx(H, W, sigma=sigma, x0=48.0)
    gy_t = torch.zeros_like(gx_t)
    out = edge_fwhm(gx_t, gy_t, gx_t, gy_t, _signal_mask(gx_t))
    expected = sigma * 2.0 * math.sqrt(2.0 * math.log(2.0))
    assert out[0].item() == pytest.approx(expected, abs=0.05)


def test_wider_sigma_gives_wider_fwhm() -> None:
    H = W = 96
    narrow = _gaussian_step_gx(H, W, sigma=1.5, x0=48.0)
    wide = _gaussian_step_gx(H, W, sigma=3.0, x0=48.0)
    z = torch.zeros_like(narrow)
    out_narrow = edge_fwhm(narrow, z, narrow, z, _signal_mask(narrow))
    out_wide = edge_fwhm(wide, z, wide, z, _signal_mask(wide))
    assert out_wide[0].item() > out_narrow[0].item() + 1.0
