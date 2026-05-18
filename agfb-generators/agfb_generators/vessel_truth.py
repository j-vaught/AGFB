"""Scalar truth masks for vessel crossing and bifurcation generators."""

from __future__ import annotations

import torch

from agfb_generators.base import coord_grid


def vessel_crossing_truth(
    height: int,
    width: int,
    *,
    sigma_a: float,
    sigma_b: float,
    theta_a_rad: float,
    theta_b_rad: float,
    contrast_a: float = 1.0,
    contrast_b: float = 1.0,
    u0_a: float = 0.0,
    u0_b: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return scalar truth maps for a two-branch vessel crossing.

    The contrast parameters are accepted for signature symmetry with the
    generator but do not affect geometric truth. `centerline_mask` is the union
    of one-sigma branch tubes. `branch_label` is 0 for background, 1 for branch
    A, and 2 for branch B, using nearest-centerline assignment inside the
    union. `junction_mask` marks the overlap of the two branch tubes.
    `radius_map` stores the nearest absolute centerline distance in pixels.
    """
    del contrast_a, contrast_b
    device = device or torch.device("cpu")
    xx, yy = coord_grid(height, width, device, dtype)

    u_a = xx * torch.cos(_scalar(theta_a_rad, device, dtype))
    u_a = u_a + yy * torch.sin(_scalar(theta_a_rad, device, dtype)) - float(u0_a)
    u_b = xx * torch.cos(_scalar(theta_b_rad, device, dtype))
    u_b = u_b + yy * torch.sin(_scalar(theta_b_rad, device, dtype)) - float(u0_b)

    r_a = torch.abs(u_a)
    r_b = torch.abs(u_b)
    mask_a = r_a <= float(sigma_a)
    mask_b = r_b <= float(sigma_b)
    centerline_mask = mask_a | mask_b
    junction_mask = mask_a & mask_b
    radius_map = torch.minimum(r_a, r_b).to(dtype=dtype)

    branch_label = torch.zeros((height, width), device=device, dtype=torch.long)
    branch_label[centerline_mask & (r_a <= r_b)] = 1
    branch_label[centerline_mask & (r_b < r_a)] = 2
    return {
        "centerline_mask": centerline_mask,
        "branch_label": branch_label,
        "junction_mask": junction_mask,
        "radius_map": radius_map,
    }


def vessel_bifurcation_truth(
    height: int,
    width: int,
    *,
    sigma_trunk: float,
    sigma_left: float,
    sigma_right: float,
    theta_trunk_rad: float,
    theta_left_rad: float,
    theta_right_rad: float,
    junction_x: float = 0.0,
    junction_y: float = 0.0,
    contrast: float = 1.0,
    gate_sigma: float = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return scalar truth maps for a gated three-branch bifurcation.

    The contrast parameter is accepted for signature symmetry with the
    generator but does not affect geometric truth. The theta parameters are
    tangent directions, matching `vessel_bifurcation`. Labels are 0 for
    background, 1 for trunk, 2 for left branch, and 3 for right branch.
    Branch membership uses a one-sigma tube and the half-plane selected by the
    branch gate. `junction_mask` marks pixels within one `gate_sigma` of the
    junction point.
    """
    del contrast
    device = device or torch.device("cpu")
    xx, yy = coord_grid(height, width, device, dtype)
    rel_x = xx - float(junction_x)
    rel_y = yy - float(junction_y)

    trunk_r, trunk_mask = _branch_truth(
        rel_x, rel_y, theta_trunk_rad, sigma_trunk, side=-1.0, device=device, dtype=dtype
    )
    left_r, left_mask = _branch_truth(
        rel_x, rel_y, theta_left_rad, sigma_left, side=1.0, device=device, dtype=dtype
    )
    right_r, right_mask = _branch_truth(
        rel_x, rel_y, theta_right_rad, sigma_right, side=1.0, device=device, dtype=dtype
    )

    centerline_mask = trunk_mask | left_mask | right_mask
    stacked_r = torch.stack((trunk_r, left_r, right_r), dim=0)
    stacked_mask = torch.stack((trunk_mask, left_mask, right_mask), dim=0)
    inf = torch.full_like(stacked_r, torch.inf)
    masked_r = torch.where(stacked_mask, stacked_r, inf)
    radius_map = torch.min(masked_r, dim=0).values
    radius_map = torch.where(centerline_mask, radius_map, torch.min(stacked_r, dim=0).values)

    labels = torch.argmin(masked_r, dim=0).to(torch.long) + 1
    branch_label = torch.where(centerline_mask, labels, torch.zeros_like(labels))
    junction_radius = max(
        float(gate_sigma), float(sigma_trunk), float(sigma_left), float(sigma_right)
    )
    junction_mask = torch.sqrt(rel_x * rel_x + rel_y * rel_y) <= junction_radius
    return {
        "centerline_mask": centerline_mask,
        "branch_label": branch_label,
        "junction_mask": junction_mask,
        "radius_map": radius_map.to(dtype=dtype),
    }


def _scalar(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(float(value), device=device, dtype=dtype)


def _branch_truth(
    rel_x: torch.Tensor,
    rel_y: torch.Tensor,
    theta_rad: float,
    sigma: float,
    *,
    side: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = _scalar(theta_rad, device, dtype)
    t_x = torch.cos(theta)
    t_y = torch.sin(theta)
    n_x = -t_y
    n_y = t_x
    ell = rel_x * t_x + rel_y * t_y
    u = rel_x * n_x + rel_y * n_y
    radius = torch.abs(u)
    mask = (radius <= float(sigma)) & (side * ell >= 0.0)
    return radius, mask
