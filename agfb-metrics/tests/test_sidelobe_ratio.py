"""Tests for side-lobe ratio."""

from __future__ import annotations

import math

import torch

from agfb_metrics.sidelobe_ratio import sidelobe_ratio


def _gaussian_step_gx(H: int, W: int, sigma: float, x0: float) -> torch.Tensor:
    xs = torch.arange(W, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    row = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    return row.unsqueeze(0).expand(H, W).unsqueeze(0)


def _ringy_field(H: int, W: int, sigma: float, x0: float, ring_amp: float) -> torch.Tensor:
    """Pure Gaussian peak at x0 plus a satellite peak at x0+8 of relative
    amplitude `ring_amp`. The satellite is well outside the main lobe, so
    Side-lobe ratio should report ~20*log10(ring_amp) dB."""
    xs = torch.arange(W, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    main = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    side = ring_amp * coef * torch.exp(-0.5 * ((xs - (x0 + 8.0)) / sigma) ** 2)
    return (main + side).unsqueeze(0).expand(H, W).unsqueeze(0)


def _signal_mask(gx: torch.Tensor) -> torch.Tensor:
    return gx.abs() > 1e-3 * float(gx.abs().max())


def test_clean_gaussian_has_no_sidelobe() -> None:
    """A monotone-falling Gaussian profile has no local minimum before the
    window edge, so the main lobe fills the whole window and the metric returns
    NaN (no side-lobe to measure)."""
    H = W = 96
    gx_t = _gaussian_step_gx(H, W, sigma=2.0, x0=48.0)
    gy = torch.zeros_like(gx_t)
    out = sidelobe_ratio(gx_t, gy, gx_t, gy, _signal_mask(gx_t))
    assert torch.isnan(out[0])


def test_known_sidelobe_ratio() -> None:
    """Satellite peak at 10% of main -> 20*log10(0.1) = -20 dB."""
    H = W = 128
    gx_t = _gaussian_step_gx(H, W, sigma=2.0, x0=64.0)
    gy_t = torch.zeros_like(gx_t)
    gx_f = _ringy_field(H, W, sigma=2.0, x0=64.0, ring_amp=0.1)
    # Use the clean truth field for signal mask + normal direction; the side
    # lobe in the "filter" output is then visible on the cross-edge profile.
    out = sidelobe_ratio(gx_f, gy_t, gx_t, gy_t, _signal_mask(gx_t))
    assert out[0].item() < -15.0
    assert out[0].item() > -25.0
