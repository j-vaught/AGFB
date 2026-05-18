"""Tests for B.2 tangential-to-normal leak."""

from __future__ import annotations

import math

import pytest
import torch

from cpgf_metrics.b2_tangential_normal_leak import b2_tangential_normal_leak


def test_perfectly_normal_filter_is_very_negative() -> None:
    """If grad_filter == grad_true, the entire signal projects on the normal.
    Float32 round-off keeps E_t > 0 (about 1e-15 of E_n), so we expect a very
    negative dB value rather than exact -inf — the -inf branch only triggers
    when filter output is literally zero."""
    torch.manual_seed(0)
    gx_t = torch.randn(1, 16, 16)
    gy_t = torch.randn(1, 16, 16)
    mask = torch.ones(1, 16, 16, dtype=torch.bool)
    out = b2_tangential_normal_leak(gx_t, gy_t, gx_t, gy_t, mask)
    assert out[0].item() < -100.0


def test_perfectly_tangential_filter_is_very_positive() -> None:
    """If grad_filter is rotated 90 deg from grad_true, all energy is on the
    tangent — float32 round-off leaves a tiny normal component, so we expect
    a very positive dB value rather than exact +inf."""
    torch.manual_seed(1)
    gx_t = torch.randn(1, 16, 16)
    gy_t = torch.randn(1, 16, 16)
    mask = torch.ones(1, 16, 16, dtype=torch.bool)
    out = b2_tangential_normal_leak(-gy_t, gx_t, gx_t, gy_t, mask)
    assert out[0].item() > 100.0


def test_zero_filter_output_returns_minus_inf() -> None:
    """The literal-zero branch: when both E_n and E_t are zero, the metric
    falls into the -inf clamp."""
    gx_t = torch.tensor([[[1.0, 1.0]]])
    gy_t = torch.tensor([[[0.0, 0.0]]])
    gx = torch.zeros_like(gx_t)
    gy = torch.zeros_like(gy_t)
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    out = b2_tangential_normal_leak(gx, gy, gx_t, gy_t, mask)
    assert math.isinf(out[0].item()) and out[0].item() < 0


def test_equal_normal_and_tangential_energy_is_zero_db() -> None:
    """50/50 split of energy between normal and tangent → 10*log10(1) = 0."""
    gx_t = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    gy_t = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    gx = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    gy = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    out = b2_tangential_normal_leak(gx, gy, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(0.0, abs=1e-5)


def test_known_db_value() -> None:
    """E_t / E_n = 0.01 → -20 dB."""
    gx_t = torch.tensor([[[1.0]]])
    gy_t = torch.tensor([[[0.0]]])
    gx = torch.tensor([[[1.0]]])
    gy = torch.tensor([[[0.1]]])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    out = b2_tangential_normal_leak(gx, gy, gx_t, gy_t, mask)
    assert out[0].item() == pytest.approx(-20.0, abs=1e-4)


def test_empty_mask_returns_nan() -> None:
    gx = torch.zeros(1, 4, 4)
    gy = torch.zeros(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    out = b2_tangential_normal_leak(gx, gy, gx, gy, mask)
    assert torch.isnan(out[0])
