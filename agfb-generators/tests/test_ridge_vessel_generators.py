"""Tests for asymmetric ridge, curved ridge, and vessel junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators.asymmetric_ridge import asymmetric_ridge
from agfb_generators.base import Frame
from agfb_generators.curved_ridge import curved_ridge
from agfb_generators.smoothed_bar import smoothed_bar
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


def test_smoothed_bar_default_call_renders_frame() -> None:
    frame = smoothed_bar(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_smoothed_bar_batched_consistent_with_scalar() -> None:
    bar_width = torch.tensor([18.0, 24.0, 30.0])
    angle = torch.tensor([0.0, math.radians(20.0), math.radians(40.0)])
    center_offset = torch.tensor([-3.0, 0.0, 3.0])
    amplitude = torch.tensor([0.75, 1.0, 1.25])
    edge_sigma = torch.tensor([2.0, 3.0, 4.0])

    out = smoothed_bar(
        80,
        88,
        bar_width=bar_width,
        angle_rad=angle,
        center_offset=center_offset,
        amplitude=amplitude,
        edge_sigma=edge_sigma,
    )

    assert out.I.shape == (3, 80, 88)
    assert out.g.shape == (3, 2, 80, 88)
    for i in range(3):
        single = smoothed_bar(
            80,
            88,
            bar_width=float(bar_width[i]),
            angle_rad=float(angle[i]),
            center_offset=float(center_offset[i]),
            amplitude=float(amplitude[i]),
            edge_sigma=float(edge_sigma[i]),
        )
        assert torch.allclose(out.I[i], single.I[0])
        assert torch.allclose(out.gx[i], single.gx[0])
        assert torch.allclose(out.gy[i], single.gy[0])


def test_smoothed_bar_honors_requested_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_bar(
        20,
        22,
        bar_width=14.0,
        angle_rad=math.radians(20.0),
        edge_sigma=2.0,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_bar_infers_tensor_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_bar(
        20,
        22,
        bar_width=torch.tensor([12.0, 16.0], device=device),
        angle_rad=torch.tensor([0.0, math.radians(25.0)], device=device),
        edge_sigma=2.0,
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device


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
        branch_a_width_sigma=5.0,
        branch_b_width_sigma=7.0,
        branch_a_normal_angle_rad=math.radians(20.0),
        branch_b_normal_angle_rad=math.radians(115.0),
        branch_a_amplitude=0.8,
        branch_b_amplitude=1.2,
        branch_a_center_offset=-1.0,
        branch_b_center_offset=2.0,
    )
    _check_signal_mask(f, rel_tol=1e-3, name="vessel_crossing")


def test_vessel_bifurcation_gradient_matches_fd() -> None:
    f = vessel_bifurcation(
        256,
        256,
        trunk_width_sigma=5.0,
        left_width_sigma=4.5,
        right_width_sigma=5.5,
        trunk_tangent_angle_rad=math.radians(90.0),
        left_tangent_angle_rad=math.radians(40.0),
        right_tangent_angle_rad=math.radians(140.0),
        amplitude=1.0,
        branch_gate_sigma=10.0,
    )
    _check_signal_mask(f, rel_tol=2e-3, name="vessel_bifurcation")


def test_vessel_crossing_default_call_renders_frame() -> None:
    frame = vessel_crossing(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_vessel_bifurcation_default_call_renders_frame() -> None:
    frame = vessel_bifurcation(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_vessel_crossing_batched_consistent_with_scalar() -> None:
    branch_a_width = torch.tensor([4.0, 5.0, 6.0])
    branch_b_width = torch.tensor([3.0, 4.0, 5.0])
    branch_a_angle = torch.tensor([math.radians(15.0), math.radians(25.0), math.radians(35.0)])
    branch_b_angle = torch.tensor([math.radians(100.0), math.radians(115.0), math.radians(130.0)])
    branch_a_amplitude = torch.tensor([0.8, 1.0, 1.2])
    branch_b_amplitude = torch.tensor([1.2, 1.0, 0.8])
    branch_a_offset = torch.tensor([-2.0, 0.0, 2.0])
    branch_b_offset = torch.tensor([2.0, 0.0, -2.0])

    out = vessel_crossing(
        80,
        84,
        branch_a_width_sigma=branch_a_width,
        branch_b_width_sigma=branch_b_width,
        branch_a_normal_angle_rad=branch_a_angle,
        branch_b_normal_angle_rad=branch_b_angle,
        branch_a_amplitude=branch_a_amplitude,
        branch_b_amplitude=branch_b_amplitude,
        branch_a_center_offset=branch_a_offset,
        branch_b_center_offset=branch_b_offset,
    )

    assert out.I.shape == (3, 80, 84)
    assert out.g.shape == (3, 2, 80, 84)
    for i in range(3):
        single = vessel_crossing(
            80,
            84,
            branch_a_width_sigma=float(branch_a_width[i]),
            branch_b_width_sigma=float(branch_b_width[i]),
            branch_a_normal_angle_rad=float(branch_a_angle[i]),
            branch_b_normal_angle_rad=float(branch_b_angle[i]),
            branch_a_amplitude=float(branch_a_amplitude[i]),
            branch_b_amplitude=float(branch_b_amplitude[i]),
            branch_a_center_offset=float(branch_a_offset[i]),
            branch_b_center_offset=float(branch_b_offset[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_vessel_bifurcation_batched_consistent_with_scalar() -> None:
    trunk_width = torch.tensor([4.0, 5.0, 6.0])
    left_width = torch.tensor([3.5, 4.0, 4.5])
    right_width = torch.tensor([4.5, 4.0, 3.5])
    trunk_angle = torch.tensor([math.radians(-90.0), math.radians(-80.0), math.radians(-100.0)])
    left_angle = torch.tensor([math.radians(35.0), math.radians(45.0), math.radians(55.0)])
    right_angle = torch.tensor([math.radians(145.0), math.radians(135.0), math.radians(125.0)])
    center_x = torch.tensor([-2.0, 0.0, 2.0])
    center_y = torch.tensor([1.0, 0.0, -1.0])
    amplitude = torch.tensor([0.8, 1.0, 1.2])
    branch_gate_sigma = torch.tensor([3.0, 4.0, 5.0])

    out = vessel_bifurcation(
        80,
        84,
        trunk_width_sigma=trunk_width,
        left_width_sigma=left_width,
        right_width_sigma=right_width,
        trunk_tangent_angle_rad=trunk_angle,
        left_tangent_angle_rad=left_angle,
        right_tangent_angle_rad=right_angle,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        branch_gate_sigma=branch_gate_sigma,
    )

    assert out.I.shape == (3, 80, 84)
    assert out.g.shape == (3, 2, 80, 84)
    for i in range(3):
        single = vessel_bifurcation(
            80,
            84,
            trunk_width_sigma=float(trunk_width[i]),
            left_width_sigma=float(left_width[i]),
            right_width_sigma=float(right_width[i]),
            trunk_tangent_angle_rad=float(trunk_angle[i]),
            left_tangent_angle_rad=float(left_angle[i]),
            right_tangent_angle_rad=float(right_angle[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
            branch_gate_sigma=float(branch_gate_sigma[i]),
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_vessel_crossing_honors_requested_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = vessel_crossing(
        20,
        24,
        branch_a_width_sigma=5.0,
        branch_b_width_sigma=4.0,
        branch_a_amplitude=0.8,
        branch_b_amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_vessel_crossing_infers_tensor_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = vessel_crossing(
        20,
        24,
        branch_a_width_sigma=torch.tensor([5.0, 6.0], device=device),
        branch_b_width_sigma=4.0,
        branch_a_normal_angle_rad=torch.tensor([0.0, math.radians(20.0)], device=device),
        branch_b_normal_angle_rad=math.radians(115.0),
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_vessel_bifurcation_honors_requested_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = vessel_bifurcation(
        20,
        24,
        trunk_width_sigma=5.0,
        left_width_sigma=4.0,
        right_width_sigma=4.0,
        amplitude=1.2,
        branch_gate_sigma=4.0,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_vessel_bifurcation_infers_tensor_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = vessel_bifurcation(
        20,
        24,
        trunk_width_sigma=torch.tensor([5.0, 6.0], device=device),
        left_width_sigma=4.0,
        right_width_sigma=4.0,
        trunk_tangent_angle_rad=torch.tensor(
            [math.radians(-90.0), math.radians(-80.0)], device=device
        ),
        branch_gate_sigma=4.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


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
        branch_a_width_sigma=3.0,
        branch_b_width_sigma=4.0,
        branch_a_normal_angle_rad=math.radians(20.0),
        branch_b_normal_angle_rad=math.radians(110.0),
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
        trunk_width_sigma=3.0,
        left_width_sigma=4.0,
        right_width_sigma=5.0,
        trunk_tangent_angle_rad=math.radians(90.0),
        left_tangent_angle_rad=math.radians(35.0),
        right_tangent_angle_rad=math.radians(145.0),
        branch_gate_sigma=6.0,
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


def test_vessel_crossing_truth_default_call_renders_maps() -> None:
    truth = vessel_crossing_truth(24, 28)

    assert truth["centerline_mask"].shape == (24, 28)
    assert truth["branch_label"].shape == (24, 28)
    assert truth["junction_mask"].shape == (24, 28)
    assert truth["radius_map"].shape == (24, 28)


def test_vessel_bifurcation_truth_default_call_renders_maps() -> None:
    truth = vessel_bifurcation_truth(24, 28)

    assert truth["centerline_mask"].shape == (24, 28)
    assert truth["branch_label"].shape == (24, 28)
    assert truth["junction_mask"].shape == (24, 28)
    assert truth["radius_map"].shape == (24, 28)


def test_vessel_crossing_truth_batched_consistent_with_scalar() -> None:
    branch_a_width = torch.tensor([3.0, 4.0])
    branch_b_width = torch.tensor([5.0, 6.0])
    branch_a_angle = torch.tensor([math.radians(20.0), math.radians(30.0)])
    branch_b_angle = torch.tensor([math.radians(110.0), math.radians(120.0)])
    branch_a_offset = torch.tensor([-1.0, 1.0])
    branch_b_offset = torch.tensor([2.0, -2.0])

    truth = vessel_crossing_truth(
        40,
        44,
        branch_a_width_sigma=branch_a_width,
        branch_b_width_sigma=branch_b_width,
        branch_a_normal_angle_rad=branch_a_angle,
        branch_b_normal_angle_rad=branch_b_angle,
        branch_a_center_offset=branch_a_offset,
        branch_b_center_offset=branch_b_offset,
    )

    assert truth["centerline_mask"].shape == (2, 40, 44)
    assert truth["branch_label"].shape == (2, 40, 44)
    assert truth["junction_mask"].shape == (2, 40, 44)
    assert truth["radius_map"].shape == (2, 40, 44)
    for i in range(2):
        scalar_truth = vessel_crossing_truth(
            40,
            44,
            branch_a_width_sigma=float(branch_a_width[i]),
            branch_b_width_sigma=float(branch_b_width[i]),
            branch_a_normal_angle_rad=float(branch_a_angle[i]),
            branch_b_normal_angle_rad=float(branch_b_angle[i]),
            branch_a_center_offset=float(branch_a_offset[i]),
            branch_b_center_offset=float(branch_b_offset[i]),
        )
        for key, value in scalar_truth.items():
            assert torch.equal(truth[key][i], value)


def test_vessel_bifurcation_truth_batched_consistent_with_scalar() -> None:
    trunk_width = torch.tensor([3.0, 4.0])
    left_width = torch.tensor([4.0, 5.0])
    right_width = torch.tensor([5.0, 6.0])
    trunk_angle = torch.tensor([math.radians(-90.0), math.radians(-80.0)])
    left_angle = torch.tensor([math.radians(35.0), math.radians(45.0)])
    right_angle = torch.tensor([math.radians(145.0), math.radians(135.0)])
    center_x = torch.tensor([-1.0, 1.0])
    center_y = torch.tensor([2.0, -2.0])
    branch_gate_sigma = torch.tensor([4.0, 6.0])

    truth = vessel_bifurcation_truth(
        40,
        44,
        trunk_width_sigma=trunk_width,
        left_width_sigma=left_width,
        right_width_sigma=right_width,
        trunk_tangent_angle_rad=trunk_angle,
        left_tangent_angle_rad=left_angle,
        right_tangent_angle_rad=right_angle,
        center_x=center_x,
        center_y=center_y,
        branch_gate_sigma=branch_gate_sigma,
    )

    assert truth["centerline_mask"].shape == (2, 40, 44)
    assert truth["branch_label"].shape == (2, 40, 44)
    assert truth["junction_mask"].shape == (2, 40, 44)
    assert truth["radius_map"].shape == (2, 40, 44)
    for i in range(2):
        scalar_truth = vessel_bifurcation_truth(
            40,
            44,
            trunk_width_sigma=float(trunk_width[i]),
            left_width_sigma=float(left_width[i]),
            right_width_sigma=float(right_width[i]),
            trunk_tangent_angle_rad=float(trunk_angle[i]),
            left_tangent_angle_rad=float(left_angle[i]),
            right_tangent_angle_rad=float(right_angle[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            branch_gate_sigma=float(branch_gate_sigma[i]),
        )
        for key, value in scalar_truth.items():
            assert torch.equal(truth[key][i], value)


def test_vessel_truth_helpers_infer_tensor_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crossing_truth = vessel_crossing_truth(
        20,
        24,
        branch_a_width_sigma=torch.tensor([3.0, 4.0], device=device),
        branch_b_width_sigma=5.0,
    )
    bifurcation_truth = vessel_bifurcation_truth(
        20,
        24,
        trunk_width_sigma=torch.tensor([3.0, 4.0], device=device),
        left_width_sigma=4.0,
        right_width_sigma=5.0,
    )

    for truth in (crossing_truth, bifurcation_truth):
        for value in truth.values():
            assert value.device == device
