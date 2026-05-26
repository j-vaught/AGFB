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
    infer_device,
    normalize_contrast,
    validate_amplitude,
    validate_positive,
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
    fields. Each arm is a stroked half-infinite centerline ray starting at the
    junction point, so the endpoint naturally has a round cap with radius
    `arm_width_px / 2`. The final intensity is obtained by smoothing the signed
    distance to the union of those stroked rays, which avoids the divots and
    ad hoc endpoint caps produced by multiplying separate half-bar masks.
    """
    validate_amplitude("contrast", contrast)
    validate_positive("arm_width_px", arm_width_px)
    validate_positive("sigma_e", sigma_e)
    device = infer_device(device, *angles_rad, arm_width_px, x0, y0, contrast, sigma_e)
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

    distances: list[torch.Tensor] = []
    gradients_x: list[torch.Tensor] = []
    gradients_y: list[torch.Tensor] = []

    for angle in angle_batches:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        normal_x = -sin_angle
        normal_y = cos_angle

        tangent_coord = x_from_center * cos_angle + y_from_center * sin_angle
        normal_coord = x_from_center * normal_x + y_from_center * normal_y
        distance, gradient_x, gradient_y = _ray_stroke_signed_distance(
            tangent_coord,
            normal_coord,
            stroke_radius,
            cos_angle,
            sin_angle,
            normal_x,
            normal_y,
        )
        distances.append(distance)
        gradients_x.append(gradient_x)
        gradients_y.append(gradient_y)

    blend_width = torch.clamp(edge_sigma_batch * 0.25, max=1.0).clamp_min(1e-6)
    signed_distance, signed_distance_gx, signed_distance_gy = _smooth_min_signed_distance(
        distances,
        gradients_x,
        gradients_y,
        blend_width,
    )

    normalized_distance = -signed_distance / edge_sigma_batch
    intensity = contrast_batch * gauss_Phi(normalized_distance)
    edge_derivative = -(contrast_batch / edge_sigma_batch) * gauss_phi(normalized_distance)
    gradient_x = edge_derivative * signed_distance_gx
    gradient_y = edge_derivative * signed_distance_gy
    return normalize_contrast(intensity, gradient_x, gradient_y, contrast_batch)


def _smooth_min_signed_distance(
    distances: Sequence[torch.Tensor],
    gradients_x: Sequence[torch.Tensor],
    gradients_y: Sequence[torch.Tensor],
    blend_width: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend SDF gradients without expanding the zero-distance boundary."""
    distance_stack = torch.stack(tuple(distances), dim=0)
    gradient_x_stack = torch.stack(tuple(gradients_x), dim=0)
    gradient_y_stack = torch.stack(tuple(gradients_y), dim=0)

    hard_min_distance = torch.amin(distance_stack, dim=0)
    relative_distance = distance_stack - hard_min_distance
    tie_bias = blend_width * torch.logsumexp(-relative_distance / blend_width, dim=0)
    weights = torch.softmax(-distance_stack / blend_width, dim=0)
    signed_distance = (
        -blend_width * torch.logsumexp(-distance_stack / blend_width, dim=0) + tie_bias
    )
    gradient_x = torch.sum(weights * gradient_x_stack, dim=0)
    gradient_y = torch.sum(weights * gradient_y_stack, dim=0)
    return signed_distance, gradient_x, gradient_y


def _ray_stroke_signed_distance(
    tangent_coord: torch.Tensor,
    normal_coord: torch.Tensor,
    stroke_radius: torch.Tensor,
    tangent_x: torch.Tensor,
    tangent_y: torch.Tensor,
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed distance to one stroked ray and its gradient."""
    behind_endpoint = tangent_coord < 0.0
    endpoint_distance = torch.sqrt(
        tangent_coord * tangent_coord + normal_coord * normal_coord
    ).clamp_min(1e-12)
    centerline_distance = torch.where(
        behind_endpoint,
        endpoint_distance,
        torch.abs(normal_coord),
    )
    distance = centerline_distance - stroke_radius

    endpoint_gradient_x = (tangent_coord * tangent_x + normal_coord * normal_x) / endpoint_distance
    endpoint_gradient_y = (tangent_coord * tangent_y + normal_coord * normal_y) / endpoint_distance
    normal_sign = torch.sign(normal_coord)
    side_gradient_x = normal_sign * normal_x
    side_gradient_y = normal_sign * normal_y

    gradient_x = torch.where(behind_endpoint, endpoint_gradient_x, side_gradient_x)
    gradient_y = torch.where(behind_endpoint, endpoint_gradient_y, side_gradient_y)
    return distance, gradient_x, gradient_y
