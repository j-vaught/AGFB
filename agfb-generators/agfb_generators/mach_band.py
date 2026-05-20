"""Mach-band ramp generator."""

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
    infer_device,
    pack,
)


def mach_band(
    height: int,
    width: int,
    *,
    ramp_width: Numeric | None = None,
    angle_rad: Numeric | None = None,
    center_offset: Numeric | None = None,
    amplitude: Numeric | None = None,
    edge_sigma: Numeric | None = None,
    shoulder_amplitude: Numeric | None = None,
    shoulder_sigma: Numeric | None = None,
    width_px: Numeric | None = None,
    theta_rad: Numeric | None = None,
    x0: Numeric | None = None,
    contrast: Numeric | None = None,
    sigma_e: Numeric | None = None,
    band_strength: Numeric | None = None,
    band_sigma: Numeric | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched smoothed ramp with paired Mach-band shoulders.

    The benchmark uses this generator when a filter should respond to a finite
    transition whose apparent local contrast is distorted near the two ramp
    edges. It is useful for checking whether a filter overreacts to bright and
    dark shoulders instead of the underlying ramp transition.

    `ramp_width` is the distance in pixels between the low and high ramp edges.
    `angle_rad` is the ramp normal direction in radians, measured from the
    image `+x` direction. `center_offset` shifts the midpoint of the ramp in
    the shared centered coordinate system. `amplitude` is the base ramp
    contrast. `edge_sigma` controls Gaussian smoothing of the finite ramp.
    `shoulder_amplitude` sets the signed shoulder strength as a fraction of
    `amplitude`, and `shoulder_sigma` controls each shoulder width.

    The projected coordinate is
    `z = x * cos(angle) + y * sin(angle) - center_offset`. The base intensity is
    the Gaussian-smoothed finite ramp over `-ramp_width / 2 <= z <=
    ramp_width / 2`. The Mach-band term subtracts a Gaussian shoulder at the
    low edge and adds a matching shoulder at the high edge. The returned
    `Frame` contains the intensity image and the closed-form gradients with
    respect to image `x` and `y`. If `device` is omitted and a tensor parameter
    is passed, the render stays on that tensor's device.
    """
    ramp_width = _resolve_required_parameter("ramp_width", ramp_width, "width_px", width_px)
    angle_rad = _resolve_required_parameter("angle_rad", angle_rad, "theta_rad", theta_rad)
    center_offset = _resolve_optional_parameter("center_offset", center_offset, "x0", x0, 0.0)
    amplitude = _resolve_optional_parameter("amplitude", amplitude, "contrast", contrast, 1.0)
    edge_sigma = _resolve_optional_parameter("edge_sigma", edge_sigma, "sigma_e", sigma_e, 1.0)
    shoulder_amplitude = _resolve_optional_parameter(
        "shoulder_amplitude",
        shoulder_amplitude,
        "band_strength",
        band_strength,
        0.08,
    )
    shoulder_sigma = _resolve_optional_parameter(
        "shoulder_sigma",
        shoulder_sigma,
        "band_sigma",
        band_sigma,
        2.0,
    )

    device = infer_device(
        device,
        ramp_width,
        angle_rad,
        center_offset,
        amplitude,
        edge_sigma,
        shoulder_amplitude,
        shoulder_sigma,
    )
    batch_size = infer_batch_size(
        ramp_width,
        angle_rad,
        center_offset,
        amplitude,
        edge_sigma,
        shoulder_amplitude,
        shoulder_sigma,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    ramp_width_batch = as_batch(ramp_width, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    edge_sigma_batch = as_batch(edge_sigma, batch_size, device, dtype)
    shoulder_amplitude_batch = as_batch(shoulder_amplitude, batch_size, device, dtype)
    shoulder_sigma_batch = as_batch(shoulder_sigma, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    ramp_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    half_width = ramp_width_batch / 2.0
    low_edge_coord = ramp_coord + half_width
    high_edge_coord = ramp_coord - half_width

    inv_edge_sigma = torch.reciprocal(edge_sigma_batch)
    low_edge_scaled = low_edge_coord * inv_edge_sigma
    high_edge_scaled = high_edge_coord * inv_edge_sigma
    low_edge_integral = low_edge_coord * gauss_Phi(low_edge_scaled) + edge_sigma_batch * gauss_phi(
        low_edge_scaled
    )
    high_edge_integral = high_edge_coord * gauss_Phi(
        high_edge_scaled
    ) + edge_sigma_batch * gauss_phi(high_edge_scaled)
    inv_ramp_width = torch.reciprocal(ramp_width_batch)
    base_intensity = amplitude_batch * inv_ramp_width * (low_edge_integral - high_edge_integral)
    base_normal_gradient = (
        amplitude_batch
        * inv_ramp_width
        * (gauss_Phi(low_edge_scaled) - gauss_Phi(high_edge_scaled))
    )

    inv_shoulder_sigma = torch.reciprocal(shoulder_sigma_batch)
    inv_shoulder_sigma_sq = inv_shoulder_sigma * inv_shoulder_sigma
    low_shoulder_scaled = low_edge_coord * inv_shoulder_sigma
    high_shoulder_scaled = high_edge_coord * inv_shoulder_sigma
    dark_shoulder = torch.exp(-0.5 * low_shoulder_scaled * low_shoulder_scaled)
    bright_shoulder = torch.exp(-0.5 * high_shoulder_scaled * high_shoulder_scaled)
    shoulder_scale = amplitude_batch * shoulder_amplitude_batch
    shoulder_intensity = shoulder_scale * (bright_shoulder - dark_shoulder)
    shoulder_normal_gradient = shoulder_scale * (
        low_edge_coord * inv_shoulder_sigma_sq * dark_shoulder
        - high_edge_coord * inv_shoulder_sigma_sq * bright_shoulder
    )

    intensity = base_intensity + shoulder_intensity
    normal_gradient = base_normal_gradient + shoulder_normal_gradient
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return pack(intensity, gradient_x, gradient_y)


def _resolve_required_parameter(
    parameter_name: str,
    value: Numeric | None,
    legacy_name: str,
    legacy_value: Numeric | None,
) -> Numeric:
    if value is not None and legacy_value is not None:
        raise TypeError(f"pass either {parameter_name} or {legacy_name}, not both")
    if value is not None:
        return value
    if legacy_value is not None:
        return legacy_value
    raise TypeError(f"missing required keyword-only argument: {parameter_name}")


def _resolve_optional_parameter(
    parameter_name: str,
    value: Numeric | None,
    legacy_name: str,
    legacy_value: Numeric | None,
    default: Numeric,
) -> Numeric:
    if value is not None and legacy_value is not None:
        raise TypeError(f"pass either {parameter_name} or {legacy_name}, not both")
    if value is not None:
        return value
    if legacy_value is not None:
        return legacy_value
    return default
