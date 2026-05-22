"""Tests for the Triton full-image pixel evaluator."""

from __future__ import annotations

import pytest
import torch

from agfb_metrics.metrics import (
    PIXEL_METRICS,
    TritonPixelEvaluator,
    evaluate_metrics,
    is_triton_pixel_available,
)


def _case(device: torch.device):
    torch.manual_seed(4)
    B, H, W = 2, 48, 64
    gx_t = torch.randn(B, H, W, device=device)
    gy_t = torch.randn(B, H, W, device=device)
    mag_t = torch.sqrt(gx_t * gx_t + gy_t * gy_t).clamp_min(1e-6)
    gx_t = gx_t / mag_t
    gy_t = gy_t / mag_t
    gx = gx_t + 0.01 * torch.randn(B, H, W, device=device)
    gy = gy_t + 0.01 * torch.randn(B, H, W, device=device)
    return gx, gy, gx_t, gy_t


def test_triton_pixel_rejects_profile_metrics() -> None:
    with pytest.raises(ValueError, match="pixel metrics"):
        TritonPixelEvaluator(metrics=("edge_fwhm",), sigma_n=0.01)


def test_triton_pixel_requires_cuda_tensors() -> None:
    gx, gy, gx_t, gy_t = _case(torch.device("cpu"))
    evaluator = TritonPixelEvaluator(metrics=("nrmse",), sigma_n=0.01)

    with pytest.raises(RuntimeError, match="TritonPixelEvaluator requires"):
        evaluator(gx, gy, gx_t, gy_t, signal_mask=None, flat_mask=None)


def test_triton_pixel_validates_block_size() -> None:
    with pytest.raises(ValueError, match="block_size"):
        TritonPixelEvaluator(metrics=("nrmse",), block_size=256)


@pytest.mark.skipif(not is_triton_pixel_available(), reason="Triton CUDA is unavailable")
def test_triton_pixel_matches_torch_full_image_metrics() -> None:
    gx, gy, gx_t, gy_t = _case(torch.device("cuda"))
    evaluator = TritonPixelEvaluator(metrics=PIXEL_METRICS, sigma_n=0.01)

    out = evaluator(gx, gy, gx_t, gy_t, signal_mask=None, flat_mask=None)
    expected = evaluate_metrics(
        gx,
        gy,
        gx_t,
        gy_t,
        metrics=PIXEL_METRICS,
        signal_mask=None,
        flat_mask=None,
        sigma_n=0.01,
    )

    assert tuple(out) == PIXEL_METRICS
    for name, value in out.items():
        assert torch.allclose(value, expected[name], equal_nan=True, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not is_triton_pixel_available(), reason="Triton CUDA is unavailable")
def test_triton_pixel_matches_each_metric_individually() -> None:
    gx, gy, gx_t, gy_t = _case(torch.device("cuda"))

    for name in PIXEL_METRICS:
        evaluator = TritonPixelEvaluator(metrics=(name,), sigma_n=0.01, tail_mode="histogram")
        out = evaluator(gx, gy, gx_t, gy_t, signal_mask=None, flat_mask=None)
        expected = evaluate_metrics(
            gx,
            gy,
            gx_t,
            gy_t,
            metrics=(name,),
            signal_mask=None,
            flat_mask=None,
            sigma_n=0.01,
        )

        assert torch.allclose(out[name], expected[name], equal_nan=True, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not is_triton_pixel_available(), reason="Triton CUDA is unavailable")
def test_triton_pixel_histogram_tails_match_torch_quantiles_closely() -> None:
    gx, gy, gx_t, gy_t = _case(torch.device("cuda"))
    evaluator = TritonPixelEvaluator(
        metrics=("tail_vector_error", "tail_spurious_grad"),
        sigma_n=0.01,
        tail_mode="histogram",
        tail_bins=4096,
    )

    out = evaluator(gx, gy, gx_t, gy_t, signal_mask=None, flat_mask=None)
    expected = evaluate_metrics(
        gx,
        gy,
        gx_t,
        gy_t,
        metrics=("tail_vector_error", "tail_spurious_grad"),
        signal_mask=None,
        flat_mask=None,
        sigma_n=0.01,
    )

    assert torch.allclose(
        out["tail_vector_error"], expected["tail_vector_error"], atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        out["tail_spurious_grad"], expected["tail_spurious_grad"], atol=1e-3, rtol=1e-3
    )
