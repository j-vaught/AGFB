"""Bit-identical regression check against the PGF_paper prototype.

Re-implements the prototype's `SmoothedStep.render` inline (no import path
back into the sibling repo) and asserts batch-size-1 output matches.
"""

from __future__ import annotations

import math

import torch

from cpgf_generators import smoothed_step


def _prototype_render(
    height: int,
    width: int,
    theta_rad: float,
    x0: float,
    contrast: float,
    sigma_e: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render the reference smoothed step formula used by regression tests.

    The tests use this local copy of the prototype math to prove the package
    `smoothed_step` output remains bit-identical without importing sibling
    benchmark modules.
    """
    c_t, s_t = math.cos(theta_rad), math.sin(theta_rad)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    ys = torch.arange(height, device=device, dtype=torch.float32) - cy
    xs = torch.arange(width, device=device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    u = (xx * c_t + yy * s_t - x0) / sigma_e
    norm = 1.0 / math.sqrt(2.0 * math.pi)
    phi = norm * torch.exp(-0.5 * u * u)
    Phi = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))
    I = contrast * Phi
    gmag = (contrast / sigma_e) * phi
    return {"I": I, "gx": gmag * c_t, "gy": gmag * s_t}


def test_smoothed_step_matches_prototype_bitwise_at_b1() -> None:
    """Verify the scalar CPGF smoothed-step frame matches the prototype exactly."""
    device = torch.device("cpu")
    H = W = 256
    theta = math.radians(30.0)
    ref = _prototype_render(H, W, theta, x0=0.0, contrast=1.0, sigma_e=2.0, device=device)
    out = smoothed_step(H, W, theta_rad=theta, x0=0.0, contrast=1.0, sigma_e=2.0, device=device)
    assert out.I.shape == (1, H, W)
    assert out.g.shape == (1, 2, H, W)
    assert torch.equal(out.I[0], ref["I"])
    assert torch.equal(out.gx[0], ref["gx"])
    assert torch.equal(out.gy[0], ref["gy"])


def test_smoothed_step_batched_consistent_with_scalar() -> None:
    """Verify batched smoothed-step rendering matches repeated scalar renders."""
    device = torch.device("cpu")
    H = W = 128
    thetas = torch.tensor([0.0, math.radians(22.5), math.radians(45.0), math.radians(90.0)])
    out = smoothed_step(H, W, theta_rad=thetas, x0=0.0, contrast=1.0, sigma_e=2.0, device=device)
    assert out.I.shape == (4, H, W)
    for i, t in enumerate(thetas.tolist()):
        single = smoothed_step(H, W, theta_rad=t, x0=0.0, contrast=1.0, sigma_e=2.0, device=device)
        assert torch.equal(out.I[i], single.I[0])
        assert torch.equal(out.gx[i], single.gx[0])
        assert torch.equal(out.gy[i], single.gy[0])
