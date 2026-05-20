"""Bit-identical regression check for the smoothed-step reference formula.

Re-implements the closed-form smoothed-step render inline and asserts the
package output matches for batch size one.
"""

from __future__ import annotations

import math

import torch

from agfb_generators import smoothed_step


def _reference_render(
    height: int,
    width: int,
    angle_rad: float,
    center_offset: float,
    amplitude: float,
    edge_sigma: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render the reference smoothed-step formula used by regression tests.

    The tests use this local copy of the closed-form math to prove the package
    `smoothed_step` output remains bit-identical for scalar rendering.
    """
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    ys = torch.arange(height, device=device, dtype=torch.float32) - cy
    xs = torch.arange(width, device=device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    normalized_distance = (xx * cos_angle + yy * sin_angle - center_offset) / edge_sigma
    norm = 1.0 / math.sqrt(2.0 * math.pi)
    phi = norm * torch.exp(-0.5 * normalized_distance * normalized_distance)
    Phi = 0.5 * (1.0 + torch.erf(normalized_distance / math.sqrt(2.0)))
    I = amplitude * Phi
    normal_gradient = (amplitude / edge_sigma) * phi
    return {"I": I, "gx": normal_gradient * cos_angle, "gy": normal_gradient * sin_angle}


def test_smoothed_step_matches_reference_bitwise_at_b1() -> None:
    """Verify the scalar AGFB smoothed-step frame matches the reference exactly."""
    device = torch.device("cpu")
    H = W = 256
    angle = math.radians(30.0)
    ref = _reference_render(
        H,
        W,
        angle,
        center_offset=0.0,
        amplitude=1.0,
        edge_sigma=2.0,
        device=device,
    )
    out = smoothed_step(
        H,
        W,
        angle_rad=angle,
        center_offset=0.0,
        amplitude=1.0,
        edge_sigma=2.0,
        device=device,
    )
    assert out.I.shape == (1, H, W)
    assert out.g.shape == (1, 2, H, W)
    assert torch.equal(out.I[0], ref["I"])
    assert torch.equal(out.gx[0], ref["gx"])
    assert torch.equal(out.gy[0], ref["gy"])


def test_smoothed_step_batched_consistent_with_scalar() -> None:
    """Verify batched smoothed-step rendering matches repeated scalar renders."""
    device = torch.device("cpu")
    H = W = 128
    angles = torch.tensor([0.0, math.radians(22.5), math.radians(45.0), math.radians(90.0)])
    out = smoothed_step(
        H,
        W,
        angle_rad=angles,
        center_offset=0.0,
        amplitude=1.0,
        edge_sigma=2.0,
        device=device,
    )
    assert out.I.shape == (4, H, W)
    for i, angle in enumerate(angles.tolist()):
        single = smoothed_step(
            H,
            W,
            angle_rad=angle,
            center_offset=0.0,
            amplitude=1.0,
            edge_sigma=2.0,
            device=device,
        )
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])


def test_smoothed_step_default_call_renders_frame() -> None:
    """Verify smoothed-step defaults render a usable analytic frame."""
    frame = smoothed_step(32, 36)

    assert frame.I.shape == (1, 32, 36)
    assert frame.g.shape == (1, 2, 32, 36)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()


def test_smoothed_step_honors_requested_device() -> None:
    """Verify scalar smoothed-step inputs render on the requested compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_step(
        20,
        24,
        angle_rad=math.radians(20.0),
        edge_sigma=2.0,
        amplitude=1.2,
        device=device,
    )

    assert frame.I.device == device
    assert frame.g.device == device


def test_smoothed_step_infers_tensor_device() -> None:
    """Verify tensor inputs keep smoothed-step output on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = smoothed_step(
        20,
        24,
        angle_rad=torch.tensor([0.0, math.radians(25.0)], device=device),
        edge_sigma=torch.tensor([2.0, 3.0], device=device),
        amplitude=1.2,
    )

    assert frame.I.device == device
    assert frame.g.device == device
