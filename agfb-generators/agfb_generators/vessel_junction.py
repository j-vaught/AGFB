"""Vessel crossing and bifurcation generators."""

from __future__ import annotations

import math

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

_DEFAULT_CROSSING_BRANCH_A_NORMAL_ANGLE_RAD = math.radians(25.0)
_DEFAULT_CROSSING_BRANCH_B_NORMAL_ANGLE_RAD = math.radians(115.0)
_DEFAULT_BIFURCATION_TRUNK_TANGENT_ANGLE_RAD = -math.pi / 2.0
_DEFAULT_BIFURCATION_LEFT_TANGENT_ANGLE_RAD = math.radians(35.0)
_DEFAULT_BIFURCATION_RIGHT_TANGENT_ANGLE_RAD = math.radians(145.0)


def vessel_crossing(
    height: int,
    width: int,
    *,
    branch_a_width_sigma: Numeric = 5.0,
    branch_b_width_sigma: Numeric = 4.0,
    branch_a_normal_angle_rad: Numeric = _DEFAULT_CROSSING_BRANCH_A_NORMAL_ANGLE_RAD,
    branch_b_normal_angle_rad: Numeric = _DEFAULT_CROSSING_BRANCH_B_NORMAL_ANGLE_RAD,
    branch_a_amplitude: Numeric = 1.0,
    branch_b_amplitude: Numeric = 1.0,
    amplitude: Numeric = 1.0,
    branch_a_center_offset: Numeric = 0.0,
    branch_b_center_offset: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched two-vessel crossing from independent Gaussian ridges.

    The benchmark uses this generator when a filter should preserve two
    overlapping vessel-like ridges with different widths, orientations, and
    amplitudes. It is useful for checking crossing response, branch contrast
    imbalance, and whether a filter merges or suppresses one ridge at the
    intersection. The visual notebook, cross-section document, and vessel
    junction preview use this function as the two-branch crossing example.

    `branch_a_width_sigma` and `branch_b_width_sigma` set each ridge width.
    `branch_a_normal_angle_rad` and `branch_b_normal_angle_rad` are the ridge
    normal directions in radians, measured from image `+x`. `branch_a_amplitude`
    and `branch_b_amplitude` set the relative ridge weights before final
    normalization, and `amplitude` controls the realized peak-to-trough
    contrast.
    `branch_a_center_offset` and `branch_b_center_offset` shift each ridge
    across its normal coordinate in the shared centered coordinate system.

    The raw intensity is the sum of two straight Gaussian ridge fields, then it
    is affinely normalized into `[0, 1]`. The returned `Frame` contains that
    intensity image and the closed-form summed gradients with respect to image
    `x` and `y`. If `device` is omitted and a tensor parameter is passed, the
    render stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    validate_positive("branch_a_width_sigma", branch_a_width_sigma)
    validate_positive("branch_b_width_sigma", branch_b_width_sigma)
    device = infer_device(
        device,
        branch_a_width_sigma,
        branch_b_width_sigma,
        branch_a_normal_angle_rad,
        branch_b_normal_angle_rad,
        branch_a_amplitude,
        branch_b_amplitude,
        amplitude,
        branch_a_center_offset,
        branch_b_center_offset,
    )
    batch_size = infer_batch_size(
        branch_a_width_sigma,
        branch_b_width_sigma,
        branch_a_normal_angle_rad,
        branch_b_normal_angle_rad,
        branch_a_amplitude,
        branch_b_amplitude,
        amplitude,
        branch_a_center_offset,
        branch_b_center_offset,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    branch_a_width_batch = as_batch(branch_a_width_sigma, batch_size, device, dtype)
    branch_b_width_batch = as_batch(branch_b_width_sigma, batch_size, device, dtype)
    branch_a_angle_batch = as_batch(branch_a_normal_angle_rad, batch_size, device, dtype)
    branch_b_angle_batch = as_batch(branch_b_normal_angle_rad, batch_size, device, dtype)
    branch_a_amplitude_batch = as_batch(branch_a_amplitude, batch_size, device, dtype)
    branch_b_amplitude_batch = as_batch(branch_b_amplitude, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    branch_a_offset_batch = as_batch(branch_a_center_offset, batch_size, device, dtype)
    branch_b_offset_batch = as_batch(branch_b_center_offset, batch_size, device, dtype)

    branch_a = _straight_ridge(
        xx,
        yy,
        branch_a_width_batch,
        branch_a_angle_batch,
        branch_a_amplitude_batch,
        branch_a_offset_batch,
    )
    branch_b = _straight_ridge(
        xx,
        yy,
        branch_b_width_batch,
        branch_b_angle_batch,
        branch_b_amplitude_batch,
        branch_b_offset_batch,
    )
    intensity = branch_a[0] + branch_b[0]
    gradient_x = branch_a[1] + branch_b[1]
    gradient_y = branch_a[2] + branch_b[2]
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)


def vessel_bifurcation(
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
    amplitude: Numeric = 1.0,
    branch_gate_sigma: Numeric = 4.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched smooth three-branch vessel bifurcation.

    The benchmark uses this generator when a filter should preserve a
    vessel-like trunk that splits into two outgoing branches. It is useful for
    testing branch continuity, centerline preservation, and gradient behavior
    near a smooth Y-shaped meeting point. The visual notebook, cross-section
    document, and vessel junction preview use this function as the bifurcation
    example.

    `trunk_width_sigma`, `left_width_sigma`, and `right_width_sigma` set the
    Gaussian width of each branch. Each `*_tangent_angle_rad` parameter is a
    branch tangent direction in radians, measured from image `+x`. `center_x`
    and `center_y` place the bifurcation point in the shared centered
    coordinate system. `amplitude` controls the realized peak-to-trough
    contrast, and `branch_gate_sigma` controls the smooth one-sided gate that
    limits each branch to its side of the junction.

    Each branch is a Gaussian ridge across its normal coordinate, multiplied by
    a smooth one-sided gate along its tangent coordinate. The trunk gate keeps
    the negative tangent side so the trunk points into the junction. The left
    and right gates keep the positive tangent side so the branches point away
    from it. Product-rule terms from both the ridge and gate are included in
    the returned analytic gradients. The combined raw field is affinely
    normalized into `[0, 1]`. If `device` is omitted and a tensor parameter is
    passed, the render stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
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
        amplitude,
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
        amplitude,
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
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    branch_gate_sigma_batch = as_batch(branch_gate_sigma, batch_size, device, dtype)

    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    trunk = _gated_branch(
        x_from_center,
        y_from_center,
        trunk_width_batch,
        trunk_angle_batch,
        amplitude_batch,
        branch_gate_sigma_batch,
        side=-1.0,
    )
    left = _gated_branch(
        x_from_center,
        y_from_center,
        left_width_batch,
        left_angle_batch,
        amplitude_batch,
        branch_gate_sigma_batch,
        side=1.0,
    )
    right = _gated_branch(
        x_from_center,
        y_from_center,
        right_width_batch,
        right_angle_batch,
        amplitude_batch,
        branch_gate_sigma_batch,
        side=1.0,
    )

    intensity = trunk[0] + left[0] + right[0]
    gradient_x = trunk[1] + left[1] + right[1]
    gradient_y = trunk[2] + left[2] + right[2]
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)


def _straight_ridge(
    xx: torch.Tensor,
    yy: torch.Tensor,
    width_sigma: torch.Tensor,
    normal_angle: torch.Tensor,
    amplitude: torch.Tensor,
    center_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cos_angle = torch.cos(normal_angle)
    sin_angle = torch.sin(normal_angle)
    normal_coord = xx * cos_angle + yy * sin_angle - center_offset
    width_sigma_sq = width_sigma * width_sigma
    intensity = amplitude * torch.exp(-(normal_coord * normal_coord) / (2.0 * width_sigma_sq))
    normal_gradient = -intensity * normal_coord / width_sigma_sq
    return intensity, normal_gradient * cos_angle, normal_gradient * sin_angle


def _gated_branch(
    x_from_center: torch.Tensor,
    y_from_center: torch.Tensor,
    width_sigma: torch.Tensor,
    tangent_angle: torch.Tensor,
    amplitude: torch.Tensor,
    branch_gate_sigma: torch.Tensor,
    *,
    side: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tangent_x = torch.cos(tangent_angle)
    tangent_y = torch.sin(tangent_angle)
    normal_x = -tangent_y
    normal_y = tangent_x

    tangent_coord = x_from_center * tangent_x + y_from_center * tangent_y
    normal_coord = x_from_center * normal_x + y_from_center * normal_y
    width_sigma_sq = width_sigma * width_sigma
    ridge = torch.exp(-(normal_coord * normal_coord) / (2.0 * width_sigma_sq))

    gate_coord = side * tangent_coord / branch_gate_sigma
    gate = gauss_Phi(gate_coord)
    gate_derivative = side * gauss_phi(gate_coord) / branch_gate_sigma

    ridge_derivative = -ridge * normal_coord / width_sigma_sq
    intensity = amplitude * ridge * gate
    gradient_x = amplitude * (
        gate * ridge_derivative * normal_x + ridge * gate_derivative * tangent_x
    )
    gradient_y = amplitude * (
        gate * ridge_derivative * normal_y + ridge * gate_derivative * tangent_y
    )
    return intensity, gradient_x, gradient_y
