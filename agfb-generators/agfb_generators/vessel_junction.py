"""Vessel crossing and bifurcation generators."""

from __future__ import annotations

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    gauss_Phi,
    gauss_phi,
    infer_batch_size,
    pack,
)


def vessel_crossing(
    height: int,
    width: int,
    *,
    sigma_a: Numeric,
    sigma_b: Numeric,
    theta_a_rad: Numeric,
    theta_b_rad: Numeric,
    contrast_a: Numeric = 1.0,
    contrast_b: Numeric = 1.0,
    u0_a: Numeric = 0.0,
    u0_b: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render two independent straight Gaussian ridges and sum their fields."""
    device = device or torch.device("cpu")
    B = infer_batch_size(
        sigma_a,
        sigma_b,
        theta_a_rad,
        theta_b_rad,
        contrast_a,
        contrast_b,
        u0_a,
        u0_b,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    s_a = as_batch(sigma_a, B, device, dtype)
    s_b = as_batch(sigma_b, B, device, dtype)
    theta_a = as_batch(theta_a_rad, B, device, dtype)
    theta_b = as_batch(theta_b_rad, B, device, dtype)
    c_a = as_batch(contrast_a, B, device, dtype)
    c_b = as_batch(contrast_b, B, device, dtype)
    u0_a_b = as_batch(u0_a, B, device, dtype)
    u0_b_b = as_batch(u0_b, B, device, dtype)

    I_a, gx_a, gy_a = _straight_ridge(xx, yy, s_a, theta_a, c_a, u0_a_b)
    I_b, gx_b, gy_b = _straight_ridge(xx, yy, s_b, theta_b, c_b, u0_b_b)
    return pack(I_a + I_b, gx_a + gx_b, gy_a + gy_b)


def vessel_bifurcation(
    height: int,
    width: int,
    *,
    sigma_trunk: Numeric,
    sigma_left: Numeric,
    sigma_right: Numeric,
    theta_trunk_rad: Numeric,
    theta_left_rad: Numeric,
    theta_right_rad: Numeric,
    junction_x: Numeric = 0.0,
    junction_y: Numeric = 0.0,
    contrast: Numeric = 1.0,
    gate_sigma: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a smooth three-branch vessel bifurcation.

    In this generator, each `theta_*_rad` is a branch tangent direction. The
    Gaussian ridge is evaluated across the branch normal, then multiplied by a
    smooth one-sided gate along the tangent coordinate. The trunk gate keeps
    the negative tangent side so the trunk points into the junction. The left
    and right gates keep the positive tangent side so branches point outward.
    Product-rule terms from both the ridge and gate are included in `g`.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(
        sigma_trunk,
        sigma_left,
        sigma_right,
        theta_trunk_rad,
        theta_left_rad,
        theta_right_rad,
        junction_x,
        junction_y,
        contrast,
        gate_sigma,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    s_trunk = as_batch(sigma_trunk, B, device, dtype)
    s_left = as_batch(sigma_left, B, device, dtype)
    s_right = as_batch(sigma_right, B, device, dtype)
    theta_trunk = as_batch(theta_trunk_rad, B, device, dtype)
    theta_left = as_batch(theta_left_rad, B, device, dtype)
    theta_right = as_batch(theta_right_rad, B, device, dtype)
    jx = as_batch(junction_x, B, device, dtype)
    jy = as_batch(junction_y, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    gate_s = as_batch(gate_sigma, B, device, dtype)

    rel_x = xx - jx
    rel_y = yy - jy
    trunk = _gated_branch(rel_x, rel_y, s_trunk, theta_trunk, c, gate_s, side=-1.0)
    left = _gated_branch(rel_x, rel_y, s_left, theta_left, c, gate_s, side=1.0)
    right = _gated_branch(rel_x, rel_y, s_right, theta_right, c, gate_s, side=1.0)

    I = trunk[0] + left[0] + right[0]
    gx = trunk[1] + left[1] + right[1]
    gy = trunk[2] + left[2] + right[2]
    return pack(I, gx, gy)


def _straight_ridge(
    xx: torch.Tensor,
    yy: torch.Tensor,
    sigma: torch.Tensor,
    theta: torch.Tensor,
    contrast: torch.Tensor,
    u0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    u = xx * cos_t + yy * sin_t - u0
    s2 = sigma * sigma
    I = contrast * torch.exp(-(u * u) / (2.0 * s2))
    gmag = -I * u / s2
    return I, gmag * cos_t, gmag * sin_t


def _gated_branch(
    rel_x: torch.Tensor,
    rel_y: torch.Tensor,
    sigma: torch.Tensor,
    theta: torch.Tensor,
    contrast: torch.Tensor,
    gate_sigma: torch.Tensor,
    *,
    side: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_x = torch.cos(theta)
    t_y = torch.sin(theta)
    n_x = -t_y
    n_y = t_x

    ell = rel_x * t_x + rel_y * t_y
    u = rel_x * n_x + rel_y * n_y
    s2 = sigma * sigma
    ridge = torch.exp(-(u * u) / (2.0 * s2))

    z = side * ell / gate_sigma
    gate = gauss_Phi(z)
    dgate_dell = side * gauss_phi(z) / gate_sigma

    dridge_du = -ridge * u / s2
    I = contrast * ridge * gate
    gx = contrast * (gate * dridge_du * n_x + ridge * dgate_dell * t_x)
    gy = contrast * (gate * dridge_du * n_y + ridge * dgate_dell * t_y)
    return I, gx, gy
