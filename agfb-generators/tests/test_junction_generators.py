"""Tests for softened junction generators."""

from __future__ import annotations

import math

import torch

from agfb_generators.junction_truth import junction_mask
from agfb_generators.l_junction import hard_l_junction, smoothed_l_junction
from agfb_generators.t_junction import hard_t_junction, smoothed_t_junction
from agfb_generators.x_junction import hard_x_junction, smoothed_x_junction
from agfb_generators.y_junction import hard_y_junction, smoothed_y_junction
from tests.test_analytic_gradients import _check_signal_mask


def test_smoothed_junction_gradients_match_fd() -> None:
    """Check smoothed junction gradients against finite differences."""
    cases = [
        (
            "l",
            lambda: smoothed_l_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(15.0),
                center_x=0.25,
                center_y=-0.25,
                edge_sigma=4.0,
            ),
        ),
        (
            "t",
            lambda: smoothed_t_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(20.0),
                center_x=0.25,
                center_y=-0.25,
                edge_sigma=4.0,
            ),
        ),
        (
            "y",
            lambda: smoothed_y_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(10.0),
                center_x=0.25,
                center_y=-0.25,
                edge_sigma=4.0,
            ),
        ),
        (
            "x",
            lambda: smoothed_x_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(12.0),
                center_x=0.25,
                center_y=-0.25,
                edge_sigma=4.0,
            ),
        ),
    ]
    for name, render in cases:
        frame = render()
        _check_signal_mask(frame, rel_tol=5e-2, name=f"smoothed_{name}_junction")


def test_hard_junction_gradients_match_fd_relaxed() -> None:
    """Check hard junction gradients at the relaxed tolerance their width requires."""
    cases = [
        (
            "l",
            lambda: hard_l_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(15.0),
                center_x=0.25,
                center_y=-0.25,
            ),
        ),
        (
            "t",
            lambda: hard_t_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(20.0),
                center_x=0.25,
                center_y=-0.25,
            ),
        ),
        (
            "y",
            lambda: hard_y_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(10.0),
                center_x=0.25,
                center_y=-0.25,
            ),
        ),
        (
            "x",
            lambda: hard_x_junction(
                256,
                256,
                arm_width=28.0,
                angle_rad=math.radians(12.0),
                center_x=0.25,
                center_y=-0.25,
            ),
        ),
    ]
    for name, render in cases:
        frame = render()
        _check_signal_mask(frame, rel_tol=3e-1, name=f"hard_{name}_junction")


def test_junction_intensity_is_bounded() -> None:
    """Verify smooth union output stays within the requested contrast range."""
    frame = smoothed_x_junction(
        96,
        112,
        arm_width=18.0,
        angle_rad=math.radians(18.0),
        amplitude=2.5,
        edge_sigma=3.0,
    )
    assert frame.I.shape == (1, 96, 112)
    assert frame.g.shape == (1, 2, 96, 112)
    assert float(frame.I.min()) >= 0.0
    assert float(frame.I.max()) <= 2.5


def test_smoothed_l_junction_center_is_filled() -> None:
    """Verify the junction endpoint fills without a behind-corner bulge."""
    frame = smoothed_l_junction(
        65,
        65,
        arm_width=18.0,
        angle_rad=0.0,
        edge_sigma=3.0,
    )

    assert float(frame.I[0, 32, 32]) > 0.98
    assert float(frame.I[0, 23, 23]) < 0.15


def test_junction_mask_shape_dtype_and_center() -> None:
    """Check the scalar junction truth mask contract."""
    mask = junction_mask(33, 35, center_x=0.0, center_y=0.0, radius_px=4.0)
    assert mask.shape == (33, 35)
    assert mask.dtype == torch.bool
    assert bool(mask[16, 17])
    assert not bool(mask[0, 0])


def test_junction_mask_batched_consistent_with_scalar() -> None:
    """Verify batched junction truth masks match repeated scalar masks."""
    height = 33
    width = 35
    center_x = torch.tensor([-2.0, 0.0, 3.0])
    center_y = torch.tensor([1.0, 0.0, -2.0])
    radius = torch.tensor([3.0, 4.0, 5.0])

    mask = junction_mask(
        height,
        width,
        center_x=center_x,
        center_y=center_y,
        radius_px=radius,
    )

    assert mask.shape == (3, height, width)
    assert mask.dtype == torch.bool
    for i in range(3):
        single = junction_mask(
            height,
            width,
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            radius_px=float(radius[i]),
        )
        assert torch.equal(mask[i], single)


def test_junction_mask_honors_requested_device() -> None:
    """Verify scalar junction truth masks render on the requested device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = junction_mask(
        20,
        24,
        center_x=-1.0,
        center_y=2.0,
        radius_px=5.0,
        device=device,
    )

    assert mask.device == device


def test_junction_mask_infers_tensor_device() -> None:
    """Verify tensor inputs keep junction truth masks on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = junction_mask(
        20,
        24,
        center_x=torch.tensor([-1.0, 1.0], device=device),
        center_y=2.0,
        radius_px=5.0,
    )

    assert mask.device == device


def test_smoothed_l_junction_batched_consistent_with_scalar() -> None:
    """Verify batched L-junction rendering matches repeated scalar renders."""
    height = 80
    width = 84
    arm_width = torch.tensor([10.0, 14.0, 18.0])
    angle = torch.tensor([0.0, math.radians(22.5), math.radians(45.0)])
    center_x = torch.tensor([0.0, 1.0, -1.5])
    center_y = torch.tensor([0.0, -0.75, 1.25])
    amplitude = torch.tensor([1.0, 0.75, 1.25])
    edge_sigma = torch.tensor([2.0, 3.0, 4.0])

    out = smoothed_l_junction(
        height,
        width,
        arm_width=arm_width,
        angle_rad=angle,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        edge_sigma=edge_sigma,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = smoothed_l_junction(
            height,
            width,
            arm_width=float(arm_width[i]),
            angle_rad=float(angle[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
            edge_sigma=float(edge_sigma[i]),
        )
        assert torch.allclose(out.I[i], single.I[0], rtol=1e-6, atol=1e-6)
        assert torch.allclose(out.gx[i], single.gx[0], rtol=1e-6, atol=1e-6)
        assert torch.allclose(out.gy[i], single.gy[0], rtol=1e-6, atol=1e-6)


def test_smoothed_l_junction_honors_requested_device() -> None:
    """Verify scalar L-junction inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_l_junction(
        20,
        24,
        arm_width=8.0,
        angle_rad=math.radians(20.0),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_l_junction_infers_tensor_device() -> None:
    """Verify tensor inputs keep L-junction output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_l_junction(
        20,
        24,
        arm_width=torch.tensor([8.0, 10.0], device=device),
        angle_rad=torch.tensor([0.0, math.radians(20.0)], device=device),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_t_junction_default_call_renders_frame() -> None:
    """Verify T-junction defaults render a usable analytic frame."""
    frame = smoothed_t_junction(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_smoothed_t_junction_honors_requested_device() -> None:
    """Verify scalar T-junction inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_t_junction(
        20,
        24,
        arm_width=8.0,
        angle_rad=math.radians(20.0),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_t_junction_infers_tensor_device() -> None:
    """Verify tensor inputs keep T-junction output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_t_junction(
        20,
        24,
        arm_width=torch.tensor([8.0, 10.0], device=device),
        angle_rad=torch.tensor([0.0, math.radians(20.0)], device=device),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_t_junction_batched_consistent_with_scalar() -> None:
    """Verify batched T-junction rendering matches repeated scalar renders."""
    H = 80
    W = 84
    angles = torch.tensor([0.0, math.radians(22.5), math.radians(45.0)])
    arm_width = torch.tensor([10.0, 14.0, 18.0])
    center_x = torch.tensor([0.0, 1.0, -1.5])
    center_y = torch.tensor([0.0, -0.75, 1.25])
    amplitude = torch.tensor([1.0, 0.75, 1.25])
    edge_sigma = torch.tensor([2.0, 3.0, 4.0])

    out = smoothed_t_junction(
        H,
        W,
        arm_width=arm_width,
        angle_rad=angles,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        edge_sigma=edge_sigma,
    )
    assert out.I.shape == (3, H, W)

    for i in range(3):
        single = smoothed_t_junction(
            H,
            W,
            arm_width=float(arm_width[i]),
            angle_rad=float(angles[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
            edge_sigma=float(edge_sigma[i]),
        )
        assert torch.allclose(out.I[i], single.I[0], rtol=1e-6, atol=1e-6)
        assert torch.allclose(out.gx[i], single.gx[0], rtol=1e-6, atol=1e-6)
        assert torch.allclose(out.gy[i], single.gy[0], rtol=1e-6, atol=1e-6)


def test_smoothed_y_junction_default_call_renders_frame() -> None:
    """Verify Y-junction defaults render a usable analytic frame."""
    frame = smoothed_y_junction(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_smoothed_y_junction_honors_requested_device() -> None:
    """Verify scalar Y-junction inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_y_junction(
        20,
        24,
        arm_width=8.0,
        angle_rad=math.radians(-90.0),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_y_junction_infers_tensor_device() -> None:
    """Verify tensor inputs keep Y-junction output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_y_junction(
        20,
        24,
        arm_width=torch.tensor([8.0, 10.0], device=device),
        angle_rad=torch.tensor([math.radians(-90.0), math.radians(-70.0)], device=device),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_y_junction_batched_consistent_with_scalar() -> None:
    """Verify batched Y-junction rendering matches repeated scalar renders."""
    height = 80
    width = 84
    arm_width = torch.tensor([10.0, 14.0, 18.0])
    angle = torch.tensor([math.radians(-90.0), math.radians(-70.0), math.radians(-50.0)])
    center_x = torch.tensor([0.0, 1.0, -1.5])
    center_y = torch.tensor([0.0, -0.75, 1.25])
    amplitude = torch.tensor([1.0, 0.75, 1.25])
    edge_sigma = torch.tensor([2.0, 3.0, 4.0])

    out = smoothed_y_junction(
        height,
        width,
        arm_width=arm_width,
        angle_rad=angle,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        edge_sigma=edge_sigma,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = smoothed_y_junction(
            height,
            width,
            arm_width=float(arm_width[i]),
            angle_rad=float(angle[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
            edge_sigma=float(edge_sigma[i]),
        )
        assert torch.allclose(out.I[i], single.I[0], rtol=1e-5, atol=1e-5)
        assert torch.allclose(out.gx[i], single.gx[0], rtol=1e-5, atol=1e-5)
        assert torch.allclose(out.gy[i], single.gy[0], rtol=1e-5, atol=1e-5)


def test_smoothed_x_junction_default_call_renders_frame() -> None:
    """Verify X-junction defaults render a usable analytic frame."""
    frame = smoothed_x_junction(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_smoothed_x_junction_honors_requested_device() -> None:
    """Verify scalar X-junction inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_x_junction(
        20,
        24,
        arm_width=8.0,
        angle_rad=math.radians(20.0),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_x_junction_infers_tensor_device() -> None:
    """Verify tensor inputs keep X-junction output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_x_junction(
        20,
        24,
        arm_width=torch.tensor([8.0, 10.0], device=device),
        angle_rad=torch.tensor([0.0, math.radians(20.0)], device=device),
        center_x=1.0,
        center_y=-1.0,
        amplitude=1.2,
        edge_sigma=2.0,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_x_junction_batched_consistent_with_scalar() -> None:
    """Verify batched X-junction rendering matches repeated scalar renders."""
    height = 80
    width = 84
    arm_width = torch.tensor([10.0, 14.0, 18.0])
    angle = torch.tensor([math.radians(22.5), math.radians(45.0), math.radians(67.5)])
    center_x = torch.tensor([0.0, 1.0, -1.5])
    center_y = torch.tensor([0.0, -0.75, 1.25])
    amplitude = torch.tensor([1.0, 0.75, 1.25])
    edge_sigma = torch.tensor([2.0, 3.0, 4.0])

    out = smoothed_x_junction(
        height,
        width,
        arm_width=arm_width,
        angle_rad=angle,
        center_x=center_x,
        center_y=center_y,
        amplitude=amplitude,
        edge_sigma=edge_sigma,
    )

    assert out.I.shape == (3, height, width)
    assert out.g.shape == (3, 2, height, width)
    for i in range(3):
        single = smoothed_x_junction(
            height,
            width,
            arm_width=float(arm_width[i]),
            angle_rad=float(angle[i]),
            center_x=float(center_x[i]),
            center_y=float(center_y[i]),
            amplitude=float(amplitude[i]),
            edge_sigma=float(edge_sigma[i]),
        )
        assert torch.allclose(out.I[i], single.I[0], rtol=1e-5, atol=1e-5)
        assert torch.allclose(out.gx[i], single.gx[0], rtol=1e-5, atol=1e-5)
        assert torch.allclose(out.gy[i], single.gy[0], rtol=1e-5, atol=1e-5)
