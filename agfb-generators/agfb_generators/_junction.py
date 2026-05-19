"""Shared signed-distance renderer for stroked junction graphs."""

from __future__ import annotations

from collections.abc import Sequence

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


def junction_frame(
    height: int,
    width: int,
    *,
    angles_rad: Sequence[Numeric],
    arm_width_px: Numeric,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a smoothed stroked junction from analytic signed distances.

    Junction wrappers use this shared renderer for L-, T-, Y-, and X-shaped
    fields. Each arm is a half-infinite stroked segment starting at the junction
    point, and the shared node is a round join with radius `arm_width_px / 2`.
    The final intensity is obtained by smoothing the signed distance to the
    union of the arm strokes and join disk, which avoids the divots and ad hoc
    endpoint caps produced by multiplying separate half-bar masks.
    """
    device = device or torch.device("cpu")
    batch_size = infer_batch_size(*angles_rad, arm_width_px, x0, y0, contrast, sigma_e)
    xx, yy = coord_grid(height, width, device, dtype)

    angle_batches = [as_batch(angle, batch_size, device, dtype) for angle in angles_rad]
    arm_width_batch = as_batch(arm_width_px, batch_size, device, dtype)
    center_x_batch = as_batch(x0, batch_size, device, dtype)
    center_y_batch = as_batch(y0, batch_size, device, dtype)
    contrast_batch = as_batch(contrast, batch_size, device, dtype)
    edge_sigma_batch = as_batch(sigma_e, batch_size, device, dtype)

    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    stroke_radius = arm_width_batch / 2.0

    best_distance, best_gradient_x, best_gradient_y = _disk_signed_distance(
        x_from_center,
        y_from_center,
        stroke_radius,
    )

    for angle in angle_batches:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        normal_x = -sin_angle
        normal_y = cos_angle

        tangent_coord = x_from_center * cos_angle + y_from_center * sin_angle
        normal_coord = x_from_center * normal_x + y_from_center * normal_y
        distance, gradient_x, gradient_y = _half_strip_signed_distance(
            tangent_coord,
            normal_coord,
            stroke_radius,
            cos_angle,
            sin_angle,
            normal_x,
            normal_y,
        )
        use_arm = distance < best_distance
        best_distance = torch.where(use_arm, distance, best_distance)
        best_gradient_x = torch.where(use_arm, gradient_x, best_gradient_x)
        best_gradient_y = torch.where(use_arm, gradient_y, best_gradient_y)

    normalized_distance = -best_distance / edge_sigma_batch
    intensity = contrast_batch * gauss_Phi(normalized_distance)
    edge_derivative = -(contrast_batch / edge_sigma_batch) * gauss_phi(normalized_distance)
    gradient_x = edge_derivative * best_gradient_x
    gradient_y = edge_derivative * best_gradient_y
    return pack(intensity, gradient_x, gradient_y)


def _disk_signed_distance(
    x_from_center: torch.Tensor,
    y_from_center: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radial_distance = torch.sqrt(
        x_from_center * x_from_center + y_from_center * y_from_center
    ).clamp_min(1e-12)
    distance = radial_distance - radius
    gradient_x = x_from_center / radial_distance
    gradient_y = y_from_center / radial_distance
    return distance, gradient_x, gradient_y


def _half_strip_signed_distance(
    tangent_coord: torch.Tensor,
    normal_coord: torch.Tensor,
    stroke_radius: torch.Tensor,
    tangent_x: torch.Tensor,
    tangent_y: torch.Tensor,
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed distance to one half-infinite strip arm and its gradient."""
    cap_distance = -tangent_coord
    side_distance = torch.abs(normal_coord) - stroke_radius

    outside_cap = torch.clamp(cap_distance, min=0.0)
    outside_side = torch.clamp(side_distance, min=0.0)
    outside_distance = torch.sqrt(outside_cap * outside_cap + outside_side * outside_side)

    inside_distance = torch.minimum(
        torch.maximum(cap_distance, side_distance),
        torch.zeros_like(cap_distance),
    )
    distance = outside_distance + inside_distance

    normal_sign = torch.sign(normal_coord)
    cap_gradient_x = -tangent_x
    cap_gradient_y = -tangent_y
    side_gradient_x = normal_sign * normal_x
    side_gradient_y = normal_sign * normal_y

    outside_mask = outside_distance > 0.0
    outside_gradient_x = torch.zeros_like(distance)
    outside_gradient_y = torch.zeros_like(distance)
    outside_gradient_x = outside_gradient_x + torch.where(
        cap_distance > 0.0,
        outside_cap * cap_gradient_x,
        torch.zeros_like(distance),
    )
    outside_gradient_y = outside_gradient_y + torch.where(
        cap_distance > 0.0,
        outside_cap * cap_gradient_y,
        torch.zeros_like(distance),
    )
    outside_gradient_x = outside_gradient_x + torch.where(
        side_distance > 0.0,
        outside_side * side_gradient_x,
        torch.zeros_like(distance),
    )
    outside_gradient_y = outside_gradient_y + torch.where(
        side_distance > 0.0,
        outside_side * side_gradient_y,
        torch.zeros_like(distance),
    )
    outside_gradient_x = torch.where(
        outside_mask,
        outside_gradient_x / outside_distance.clamp_min(1e-12),
        outside_gradient_x,
    )
    outside_gradient_y = torch.where(
        outside_mask,
        outside_gradient_y / outside_distance.clamp_min(1e-12),
        outside_gradient_y,
    )

    use_cap_inside = cap_distance >= side_distance
    inside_gradient_x = torch.where(use_cap_inside, cap_gradient_x, side_gradient_x)
    inside_gradient_y = torch.where(use_cap_inside, cap_gradient_y, side_gradient_y)

    gradient_x = torch.where(outside_mask, outside_gradient_x, inside_gradient_x)
    gradient_y = torch.where(outside_mask, outside_gradient_y, inside_gradient_y)
    return distance, gradient_x, gradient_y
