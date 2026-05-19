"""Tests for asymmetric ridge, curved ridge, and vessel junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators.asymmetric_ridge import asymmetric_ridge
from agfb_generators.base import Frame
from agfb_generators.curved_ridge import curved_ridge
from agfb_generators.vessel_junction import vessel_bifurcation, vessel_crossing
from agfb_generators.vessel_truth import vessel_bifurcation_truth, vessel_crossing_truth


def _fd4(I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(I)
    gy = torch.zeros_like(I)
    gx[:, 2:-2] = (-I[:, 4:] + 8 * I[:, 3:-1] - 8 * I[:, 1:-3] + I[:, :-4]) / 12.0
    gy[2:-2, :] = (-I[4:, :] + 8 * I[3:-1, :] - 8 * I[1:-3, :] + I[:-4, :]) / 12.0
    return gx, gy


def _check_signal_mask(
    frame: Frame,
    *,
    rel_tol: float,
    name: str,
    exclude: torch.Tensor | None = None,
) -> None:
    I = frame.I[0]
    fd_gx, fd_gy = _fd4(I)
    inner = torch.zeros_like(I, dtype=torch.bool)
    inner[3:-3, 3:-3] = True

    mag = torch.sqrt(frame.gx[0] ** 2 + frame.gy[0] ** 2)
    signal = (mag > 1e-2 * float(mag.max())) & inner
    if exclude is not None:
        signal &= ~exclude
    n = int(signal.sum())
    assert n > 50, f"{name}: signal mask too small ({n})"

    diff_x = (fd_gx - frame.gx[0])[signal]
    diff_y = (fd_gy - frame.gy[0])[signal]
    num = torch.mean(diff_x * diff_x + diff_y * diff_y)
    den = torch.mean(frame.gx[0][signal] ** 2 + frame.gy[0][signal] ** 2)
    nrmse = float(torch.sqrt(num / den))
    assert nrmse < rel_tol, f"{name}: NRMSE={nrmse:.2e} >= {rel_tol:.2e}"


def test_asymmetric_ridge_gradient_matches_fd() -> None:
    f = asymmetric_ridge(
        256,
        256,
        negative_sigma=8.0,
        positive_sigma=9.0,
        angle_rad=math.radians(25.0),
        center_offset=1.25,
    )
    _check_signal_mask(f, rel_tol=1e-3, name="asymmetric_ridge")


def test_curved_ridge_gradient_matches_fd() -> None:
    f = curved_ridge(
        256,
        256,
        width_sigma=6.0,
        angle_rad=math.radians(35.0),
        curvature=0.002,
        normal_offset=-2.0,
        tangent_offset=3.0,
    )
    _check_signal_mask(f, rel_tol=1e-3, name="curved_ridge")


def test_curved_ridge_batched_consistent_with_scalar() -> None:
    width_sigma = torch.tensor([4.0, 5.0, 6.0])
    angle = torch.tensor([0.0, math.radians(20.0), math.radians(40.0)])
    curvature = torch.tensor([0.002, 0.003, 0.004])
    normal_offset = torch.tensor([-2.0, 0.0, 2.0])
    tangent_offset = torch.tensor([4.0, 0.0, -4.0])
    amplitude = torch.tensor([0.75, 1.0, 1.25])

    out = curved_ridge(
        96,
        112,
        width_sigma=width_sigma,
        angle_rad=angle,
        curvature=curvature,
        normal_offset=normal_offset,
        tangent_offset=tangent_offset,
        amplitude=amplitude,
    )

    assert out.I.shape == (3, 96, 112)
    assert out.g.shape == (3, 2, 96, 112)
    for i in range(3):
        single = curved_ridge(
            96,
            112,
            width_sigma=float(width_sigma[i]),
            angle_rad=float(angle[i]),
            curvature=float(curvature[i]),
            normal_offset=float(normal_offset[i]),
            tangent_offset=float(tangent_offset[i]),
            amplitude=float(amplitude[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_curved_ridge_honors_requested_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = curved_ridge(
        20,
        22,
        width_sigma=4.0,
        angle_rad=math.radians(20.0),
        curvature=0.003,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_curved_ridge_infers_tensor_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = curved_ridge(
        20,
        22,
        width_sigma=torch.tensor([4.0, 5.0], device=device),
        angle_rad=torch.tensor([0.0, math.radians(25.0)], device=device),
        curvature=0.003,
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_vessel_crossing_gradient_matches_fd() -> None:
    f = vessel_crossing(
        256,
        256,
        sigma_a=5.0,
        sigma_b=7.0,
        theta_a_rad=math.radians(20.0),
        theta_b_rad=math.radians(115.0),
        contrast_a=0.8,
        contrast_b=1.2,
        u0_a=-1.0,
        u0_b=2.0,
    )
    _check_signal_mask(f, rel_tol=1e-3, name="vessel_crossing")


def test_vessel_bifurcation_gradient_matches_fd() -> None:
    f = vessel_bifurcation(
        256,
        256,
        sigma_trunk=5.0,
        sigma_left=4.5,
        sigma_right=5.5,
        theta_trunk_rad=math.radians(90.0),
        theta_left_rad=math.radians(40.0),
        theta_right_rad=math.radians(140.0),
        contrast=1.0,
        gate_sigma=10.0,
    )
    _check_signal_mask(f, rel_tol=2e-3, name="vessel_bifurcation")


def test_asymmetric_ridge_batched_consistent_with_scalar() -> None:
    angle = torch.tensor([0.0, math.radians(30.0), math.radians(70.0)])
    negative_sigma = torch.tensor([3.0, 4.0, 5.0])
    positive_sigma = torch.tensor([6.0, 7.0, 8.0])
    center_offset = torch.tensor([-2.0, 0.0, 2.0])
    amplitude = torch.tensor([0.75, 1.0, 1.25])
    out = asymmetric_ridge(
        96,
        112,
        negative_sigma=negative_sigma,
        positive_sigma=positive_sigma,
        angle_rad=angle,
        center_offset=center_offset,
        amplitude=amplitude,
    )
    assert out.I.shape == (3, 96, 112)
    assert out.g.shape == (3, 2, 96, 112)
    for i in range(3):
        single = asymmetric_ridge(
            96,
            112,
            negative_sigma=float(negative_sigma[i]),
            positive_sigma=float(positive_sigma[i]),
            angle_rad=float(angle[i]),
            center_offset=float(center_offset[i]),
            amplitude=float(amplitude[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_asymmetric_ridge_infers_tensor_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = asymmetric_ridge(
        20,
        22,
        negative_sigma=torch.tensor([3.0, 4.0], device=device),
        positive_sigma=7.0,
        angle_rad=torch.tensor([0.0, math.radians(25.0)], device=device),
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_vessel_crossing_truth_shapes_and_dtypes() -> None:
    truth = vessel_crossing_truth(
        64,
        80,
        sigma_a=3.0,
        sigma_b=4.0,
        theta_a_rad=math.radians(20.0),
        theta_b_rad=math.radians(110.0),
        dtype=torch.float64,
    )
    assert set(truth) == {"centerline_mask", "branch_label", "junction_mask", "radius_map"}
    assert truth["centerline_mask"].shape == (64, 80)
    assert truth["centerline_mask"].dtype == torch.bool
    assert truth["junction_mask"].shape == (64, 80)
    assert truth["junction_mask"].dtype == torch.bool
    assert truth["branch_label"].shape == (64, 80)
    assert truth["branch_label"].dtype == torch.long
    assert truth["radius_map"].shape == (64, 80)
    assert truth["radius_map"].dtype == torch.float64


def test_vessel_bifurcation_truth_shapes_and_dtypes() -> None:
    truth = vessel_bifurcation_truth(
        72,
        88,
        sigma_trunk=3.0,
        sigma_left=4.0,
        sigma_right=5.0,
        theta_trunk_rad=math.radians(90.0),
        theta_left_rad=math.radians(35.0),
        theta_right_rad=math.radians(145.0),
        gate_sigma=6.0,
        dtype=torch.float64,
    )
    assert set(truth) == {"centerline_mask", "branch_label", "junction_mask", "radius_map"}
    assert truth["centerline_mask"].shape == (72, 88)
    assert truth["centerline_mask"].dtype == torch.bool
    assert truth["junction_mask"].shape == (72, 88)
    assert truth["junction_mask"].dtype == torch.bool
    assert truth["branch_label"].shape == (72, 88)
    assert truth["branch_label"].dtype == torch.long
    assert truth["radius_map"].shape == (72, 88)
    assert truth["radius_map"].dtype == torch.float64
