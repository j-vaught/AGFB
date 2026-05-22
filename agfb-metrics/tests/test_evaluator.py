"""Tests for shared-intermediate metric evaluation."""

from __future__ import annotations

import math

import pytest
import torch

from agfb_metrics.metrics import (
    angular_mae,
    edge_fwhm,
    evaluate_metrics,
    localization_offset,
    magnitude_bias,
    noise_gain,
    nrmse,
    sidelobe_ratio,
    tail_spurious_grad,
    tail_vector_error,
    tangential_normal_leak,
)


def _gaussian_step_gx(height: int = 48, width: int = 64, sigma: float = 2.0, x0: float = 30.0):
    xs = torch.arange(width, dtype=torch.float32)
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    row = coef * torch.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    return row.unsqueeze(0).expand(height, width).unsqueeze(0)


def _case():
    torch.manual_seed(0)
    gx_t = _gaussian_step_gx()
    gy_t = torch.zeros_like(gx_t)
    gx = gx_t + 0.01 * torch.randn_like(gx_t)
    gy = gy_t + 0.01 * torch.randn_like(gy_t)
    signal = gx_t.abs() > 1e-3 * float(gx_t.abs().max())
    flat = ~signal
    return gx, gy, gx_t, gy_t, signal, flat


def test_evaluate_metrics_matches_individual_metrics() -> None:
    gx, gy, gx_t, gy_t, signal, flat = _case()
    out = evaluate_metrics(
        gx,
        gy,
        gx_t,
        gy_t,
        metrics=(
            "nrmse",
            "angular_mae",
            "tail_vector_error",
            "localization_offset",
            "tangential_normal_leak",
            "magnitude_bias",
            "edge_fwhm",
            "sidelobe_ratio",
            "noise_gain",
            "tail_spurious_grad",
        ),
        signal_mask=signal,
        flat_mask=flat,
        sigma_n=0.01,
    )

    expected = {
        "nrmse": nrmse(gx, gy, gx_t, gy_t, signal),
        "angular_mae": angular_mae(gx, gy, gx_t, gy_t, signal),
        "tail_vector_error": tail_vector_error(gx, gy, gx_t, gy_t, signal),
        "localization_offset": localization_offset(gx, gy, gx_t, gy_t, signal),
        "tangential_normal_leak": tangential_normal_leak(gx, gy, gx_t, gy_t, signal),
        "magnitude_bias": magnitude_bias(gx, gy, gx_t, gy_t, signal),
        "edge_fwhm": edge_fwhm(gx, gy, gx_t, gy_t, signal),
        "sidelobe_ratio": sidelobe_ratio(gx, gy, gx_t, gy_t, signal),
        "noise_gain": noise_gain(gx, gy, flat, sigma_n=0.01),
        "tail_spurious_grad": tail_spurious_grad(gx, gy, flat),
    }

    assert tuple(out) == tuple(expected)
    for name, value in out.items():
        assert torch.allclose(value, expected[name], equal_nan=True, atol=1e-5, rtol=1e-5)


def test_evaluate_metrics_can_use_all_pixels_without_masks() -> None:
    gx, gy, gx_t, gy_t, _, _ = _case()
    out = evaluate_metrics(
        gx,
        gy,
        gx_t,
        gy_t,
        metrics=("nrmse", "magnitude_bias", "noise_gain"),
        signal_mask=None,
        flat_mask=None,
        sigma_n=0.01,
    )

    full = torch.ones_like(gx, dtype=torch.bool)
    assert torch.allclose(out["nrmse"], nrmse(gx, gy, gx_t, gy_t, full))
    assert torch.allclose(out["magnitude_bias"], magnitude_bias(gx, gy, gx_t, gy_t, full))
    assert torch.allclose(out["noise_gain"], noise_gain(gx, gy, full, sigma_n=0.01))


def test_evaluate_metrics_requires_sigma_for_noise_gain() -> None:
    gx, gy, gx_t, gy_t, signal, flat = _case()
    with pytest.raises(ValueError, match="sigma_n is required"):
        evaluate_metrics(
            gx,
            gy,
            gx_t,
            gy_t,
            metrics=("noise_gain",),
            signal_mask=signal,
            flat_mask=flat,
        )


def test_evaluate_metrics_requires_mask_for_profile_metrics() -> None:
    gx, gy, gx_t, gy_t, _, flat = _case()
    with pytest.raises(ValueError, match="profile metrics require"):
        evaluate_metrics(
            gx,
            gy,
            gx_t,
            gy_t,
            metrics=("edge_fwhm",),
            signal_mask=None,
            flat_mask=flat,
        )
