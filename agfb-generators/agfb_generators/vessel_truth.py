"""Truth masks for vessel crossing and bifurcation generators."""

from __future__ import annotations

import math

import torch

from agfb_generators.base import (
    Numeric,
    as_batch,
    coord_grid,
    infer_batch_size,
    infer_device,
    validate_positive,
)

_DEFAULT_CROSSING_BRANCH_A_NORMAL_ANGLE_RAD = math.radians(25.0)
_DEFAULT_CROSSING_BRANCH_B_NORMAL_ANGLE_RAD = math.radians(115.0)
_DEFAULT_BIFURCATION_TRUNK_TANGENT_ANGLE_RAD = -math.pi / 2.0
_DEFAULT_BIFURCATION_LEFT_TANGENT_ANGLE_RAD = math.radians(35.0)
_DEFAULT_BIFURCATION_RIGHT_TANGENT_ANGLE_RAD = math.radians(145.0)


def vessel_crossing_truth(
    height: int,
    width: int,
    *,
    branch_a_width_sigma: Numeric = 5.0,
    branch_b_width_sigma: Numeric = 4.0,
    branch_a_normal_angle_rad: Numeric = _DEFAULT_CROSSING_BRANCH_A_NORMAL_ANGLE_RAD,
    branch_b_normal_angle_rad: Numeric = _DEFAULT_CROSSING_BRANCH_B_NORMAL_ANGLE_RAD,
    branch_a_center_offset: Numeric = 0.0,
    branch_b_center_offset: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return geometric truth maps for a two-branch vessel crossing.

    Vessel crossing metrics use this helper when they need branch support,
    branch identity, a crossing-local mask, or a per-pixel distance-to-branch
    reference instead of image intensity. The visual notebook uses it to show
    the centerline-support mask for the crossing example.

    `branch_a_width_sigma` and `branch_b_width_sigma` define one-sigma
    evaluation tubes around each branch centerline. The two
    `*_normal_angle_rad` parameters match `vessel_crossing` and are measured
    from image `+x`. `branch_a_center_offset` and `branch_b_center_offset`
    shift each branch across its normal coordinate in the shared centered
    coordinate system.

    The returned dictionary contains `centerline_mask`, `branch_label`,
    `junction_mask`, and `radius_map`. `centerline_mask` is the union of both
    one-sigma branch tubes. `branch_label` is 0 for background, 1 for branch A,
    and 2 for branch B, using nearest-centerline assignment inside the union.
    `junction_mask` marks the overlap of the branch tubes. `radius_map` stores
    the nearest absolute centerline distance in pixels. Scalar inputs return
    `(height, width)` maps; one-dimensional tensor inputs return
    `(B, height, width)` maps. If `device` is omitted and a tensor parameter is
    passed, the maps stay on that tensor's device.
    """
    validate_positive("branch_a_width_sigma", branch_a_width_sigma)
    validate_positive("branch_b_width_sigma", branch_b_width_sigma)
    device = infer_device(
        device,
        branch_a_width_sigma,
        branch_b_width_sigma,
        branch_a_normal_angle_rad,
        branch_b_normal_angle_rad,
        branch_a_center_offset,
        branch_b_center_offset,
    )
    batch_size = infer_batch_size(
        branch_a_width_sigma,
        branch_b_width_sigma,
        branch_a_normal_angle_rad,
        branch_b_normal_angle_rad,
        branch_a_center_offset,
        branch_b_center_offset,
    )
    has_batched_input = _has_batched_input(
        branch_a_width_sigma,
        branch_b_width_sigma,
        branch_a_normal_angle_rad,
        branch_b_normal_angle_rad,
        branch_a_center_offset,
        branch_b_center_offset,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    branch_a_width_batch = as_batch(branch_a_width_sigma, batch_size, device, dtype)
    branch_b_width_batch = as_batch(branch_b_width_sigma, batch_size, device, dtype)
    branch_a_angle_batch = as_batch(branch_a_normal_angle_rad, batch_size, device, dtype)
    branch_b_angle_batch = as_batch(branch_b_normal_angle_rad, batch_size, device, dtype)
    branch_a_offset_batch = as_batch(branch_a_center_offset, batch_size, device, dtype)
    branch_b_offset_batch = as_batch(branch_b_center_offset, batch_size, device, dtype)

    branch_a_distance = torch.abs(
        _normal_coordinate(xx, yy, branch_a_angle_batch, branch_a_offset_batch)
    )
    branch_b_distance = torch.abs(
        _normal_coordinate(xx, yy, branch_b_angle_batch, branch_b_offset_batch)
    )
    branch_a_mask = branch_a_distance <= branch_a_width_batch
    branch_b_mask = branch_b_distance <= branch_b_width_batch
    centerline_mask = branch_a_mask | branch_b_mask
    junction_mask = branch_a_mask & branch_b_mask
    radius_map = torch.minimum(branch_a_distance, branch_b_distance)

    nearest_branch_label = torch.where(
        branch_a_distance <= branch_b_distance,
        torch.ones((), device=device, dtype=torch.long),
        torch.full((), 2, device=device, dtype=torch.long),
    )
    branch_label = torch.where(
        centerline_mask,
        nearest_branch_label,
        torch.zeros((), device=device, dtype=torch.long),
    )
    return {
        "centerline_mask": _maybe_drop_batch(centerline_mask, has_batched_input),
        "branch_label": _maybe_drop_batch(branch_label, has_batched_input),
        "junction_mask": _maybe_drop_batch(junction_mask, has_batched_input),
        "radius_map": _maybe_drop_batch(radius_map.to(dtype=dtype), has_batched_input),
    }


def vessel_bifurcation_truth(
    height: int,
    width: int,
    *,
    trunk_width_sigma: Numeric = 5.0,
    left_width_sigma: Numeric = 4.0,
    right_width_sigma: Numeric = 4.0,
    trunk_tangent_angle_rad: Numeric = _DEFAULT_BIFURCATION_TRUNK_TANGENT_ANGLE_RAD,
    left_tangent_angle_rad: Numeric = _DEFAULT_BIFURCATION_LEFT_TANGENT_ANGLE_RAD,
    right_tangent_angle_rad: Numeric = _DEFAULT_BIFURCATION_RIGHT_TANGENT_ANGLE_RAD,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    branch_gate_sigma: Numeric = 4.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return geometric truth maps for a three-branch vessel bifurcation.

    Bifurcation metrics use this helper when they need to score trunk and
    branch support separately from image intensity. The visual notebook uses it
    to show the branch-support mask for the bifurcation example.

    `trunk_width_sigma`, `left_width_sigma`, and `right_width_sigma` define
    one-sigma evaluation tubes around each branch centerline. The
    `*_tangent_angle_rad` parameters match `vessel_bifurcation` and are
    measured from image `+x`. `center_x` and `center_y` place the bifurcation
    point in the shared centered coordinate system. `branch_gate_sigma` sets
    the local junction radius floor used by `junction_mask`.

    The returned dictionary contains `centerline_mask`, `branch_label`,
    `junction_mask`, and `radius_map`. `branch_label` is 0 for background, 1
    for trunk, 2 for the left branch, and 3 for the right branch. Branch
    membership uses a one-sigma tube and a hard half-plane matching each
    branch gate side. `junction_mask` marks pixels within the larger of
    `branch_gate_sigma` and the three branch widths from the bifurcation point.
    Scalar inputs return `(height, width)` maps; one-dimensional tensor inputs
    return `(B, height, width)` maps. If `device` is omitted and a tensor
    parameter is passed, the maps stay on that tensor's device.
    """
    validate_positive("trunk_width_sigma", trunk_width_sigma)
    validate_positive("left_width_sigma", left_width_sigma)
    validate_positive("right_width_sigma", right_width_sigma)
    validate_positive("branch_gate_sigma", branch_gate_sigma)
    device = infer_device(
        device,
        trunk_width_sigma,
        left_width_sigma,
        right_width_sigma,
        trunk_tangent_angle_rad,
        left_tangent_angle_rad,
        right_tangent_angle_rad,
        center_x,
        center_y,
        branch_gate_sigma,
    )
    batch_size = infer_batch_size(
        trunk_width_sigma,
        left_width_sigma,
        right_width_sigma,
        trunk_tangent_angle_rad,
        left_tangent_angle_rad,
        right_tangent_angle_rad,
        center_x,
        center_y,
        branch_gate_sigma,
    )
    has_batched_input = _has_batched_input(
        trunk_width_sigma,
        left_width_sigma,
        right_width_sigma,
        trunk_tangent_angle_rad,
        left_tangent_angle_rad,
        right_tangent_angle_rad,
        center_x,
        center_y,
        branch_gate_sigma,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    trunk_width_batch = as_batch(trunk_width_sigma, batch_size, device, dtype)
    left_width_batch = as_batch(left_width_sigma, batch_size, device, dtype)
    right_width_batch = as_batch(right_width_sigma, batch_size, device, dtype)
    trunk_angle_batch = as_batch(trunk_tangent_angle_rad, batch_size, device, dtype)
    left_angle_batch = as_batch(left_tangent_angle_rad, batch_size, device, dtype)
    right_angle_batch = as_batch(right_tangent_angle_rad, batch_size, device, dtype)
    center_x_batch = as_batch(center_x, batch_size, device, dtype)
    center_y_batch = as_batch(center_y, batch_size, device, dtype)
    branch_gate_sigma_batch = as_batch(branch_gate_sigma, batch_size, device, dtype)

    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch

    trunk_r, trunk_mask = _branch_truth(
        x_from_center,
        y_from_center,
        trunk_angle_batch,
        trunk_width_batch,
        side=-1.0,
    )
    left_r, left_mask = _branch_truth(
        x_from_center,
        y_from_center,
        left_angle_batch,
        left_width_batch,
        side=1.0,
    )
    right_r, right_mask = _branch_truth(
        x_from_center,
        y_from_center,
        right_angle_batch,
        right_width_batch,
        side=1.0,
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
    junction_radius = torch.maximum(
        torch.maximum(branch_gate_sigma_batch, trunk_width_batch),
        torch.maximum(left_width_batch, right_width_batch),
    )
    junction_distance = torch.sqrt(x_from_center * x_from_center + y_from_center * y_from_center)
    junction_mask = junction_distance <= junction_radius
    return {
        "centerline_mask": _maybe_drop_batch(centerline_mask, has_batched_input),
        "branch_label": _maybe_drop_batch(branch_label, has_batched_input),
        "junction_mask": _maybe_drop_batch(junction_mask, has_batched_input),
        "radius_map": _maybe_drop_batch(radius_map.to(dtype=dtype), has_batched_input),
    }


def _normal_coordinate(
    xx: torch.Tensor,
    yy: torch.Tensor,
    normal_angle: torch.Tensor,
    center_offset: torch.Tensor,
) -> torch.Tensor:
    return xx * torch.cos(normal_angle) + yy * torch.sin(normal_angle) - center_offset


def _branch_truth(
    x_from_center: torch.Tensor,
    y_from_center: torch.Tensor,
    tangent_angle: torch.Tensor,
    width_sigma: torch.Tensor,
    *,
    side: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tangent_x = torch.cos(tangent_angle)
    tangent_y = torch.sin(tangent_angle)
    normal_x = -tangent_y
    normal_y = tangent_x
    tangent_coord = x_from_center * tangent_x + y_from_center * tangent_y
    normal_coord = x_from_center * normal_x + y_from_center * normal_y
    radius = torch.abs(normal_coord)
    mask = (radius <= width_sigma) & (side * tangent_coord >= 0.0)
    return radius, mask


def _has_batched_input(*params: Numeric) -> bool:
    return any(isinstance(param, torch.Tensor) and param.ndim == 1 for param in params)


def _maybe_drop_batch(tensor: torch.Tensor, has_batched_input: bool) -> torch.Tensor:
    if has_batched_input:
        return tensor
    return tensor[0]
