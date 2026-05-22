"""Tests for metric runners."""

from __future__ import annotations

import math

import pytest
import torch

from agfb_metrics.metrics import noise_gain, nrmse
from agfb_metrics.runners import DEFAULT_METRICS, MetricSpec, run_all_metrics, run_metric_set


def _gaussian_step_gx(height: int = 64, width: int = 64) -> torch.Tensor:
    sigma = 2.0
    x0 = width // 2
    xs = torch.arange(width, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    row = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    return row.unsqueeze(0).expand(height, width).unsqueeze(0)


def test_run_all_metrics_returns_default_metric_names() -> None:
    gx_t = _gaussian_step_gx()
    gy_t = torch.zeros_like(gx_t)

    out = run_all_metrics(gx_t, gy_t, gx_t, gy_t, sigma_n=1.0)

    assert tuple(out) == tuple(spec.name for spec in DEFAULT_METRICS)
    assert all(value.shape == (1,) for value in out.values())
    assert out["nrmse"].item() == pytest.approx(0.0, abs=1e-6)


def test_run_metric_set_can_run_subset_without_noise_sigma() -> None:
    gx_t = _gaussian_step_gx()
    gy_t = torch.zeros_like(gx_t)
    specs = (MetricSpec("nrmse_only", nrmse, "edge"),)

    out = run_metric_set(gx_t, gy_t, gx_t, gy_t, metric_specs=specs)

    assert set(out) == {"nrmse_only"}
    assert out["nrmse_only"].item() == pytest.approx(0.0, abs=1e-6)


def test_noise_gain_requires_sigma_n() -> None:
    gx_t = _gaussian_step_gx()
    gy_t = torch.zeros_like(gx_t)
    specs = (MetricSpec("noise_gain", noise_gain, "flat_with_sigma"),)

    with pytest.raises(ValueError, match="sigma_n is required"):
        run_metric_set(gx_t, gy_t, gx_t, gy_t, metric_specs=specs)
