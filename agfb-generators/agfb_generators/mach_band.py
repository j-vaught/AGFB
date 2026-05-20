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
    ramp_width: Numeric = 64.0,
    angle_rad: Numeric = 0.0,
    center_offset: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    edge_sigma: Numeric = 3.0,
    shoulder_amplitude: Numeric = 0.08,
    shoulder_sigma: Numeric = 4.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    **legacy_kwargs: Numeric,
) -> Frame:
    """Render a batched smoothed ramp with paired Mach-band shoulders.

    The benchmark uses this generator when a filter should respond to a finite
    transition whose apparent local contrast is distorted near the two ramp
    edges. It is useful for checking whether a filter overreacts to bright and
    dark shoulders instead of the underlying ramp transition.

    The default call renders a horizontal 64 px ramp centered in the image,
    smoothed with a 3 px Gaussian edge spread and 8 percent Mach shoulders.

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
    (
        ramp_width,
        angle_rad,
        center_offset,
        amplitude,
        edge_sigma,
        shoulder_amplitude,
        shoulder_sigma,
    ) = _apply_legacy_aliases(
        legacy_kwargs,
        ramp_width=ramp_width,
        angle_rad=angle_rad,
        center_offset=center_offset,
        amplitude=amplitude,
        edge_sigma=edge_sigma,
        shoulder_amplitude=shoulder_amplitude,
        shoulder_sigma=shoulder_sigma,
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


def _apply_legacy_aliases(
    legacy_kwargs: dict[str, Numeric],
    *,
    ramp_width: Numeric,
    angle_rad: Numeric,
    center_offset: Numeric,
    amplitude: Numeric,
    edge_sigma: Numeric,
    shoulder_amplitude: Numeric,
    shoulder_sigma: Numeric,
) -> tuple[Numeric, Numeric, Numeric, Numeric, Numeric, Numeric, Numeric]:
    alias_map = {
        "width_px": "ramp_width",
        "theta_rad": "angle_rad",
        "x0": "center_offset",
        "contrast": "amplitude",
        "sigma_e": "edge_sigma",
        "band_strength": "shoulder_amplitude",
        "band_sigma": "shoulder_sigma",
    }
    params = {
        "ramp_width": ramp_width,
        "angle_rad": angle_rad,
        "center_offset": center_offset,
        "amplitude": amplitude,
        "edge_sigma": edge_sigma,
        "shoulder_amplitude": shoulder_amplitude,
        "shoulder_sigma": shoulder_sigma,
    }
    for legacy_name, value in legacy_kwargs.items():
        parameter_name = alias_map.get(legacy_name)
        if parameter_name is None:
            raise TypeError(f"unexpected keyword argument: {legacy_name}")
        params[parameter_name] = value
    return (
        params["ramp_width"],
        params["angle_rad"],
        params["center_offset"],
        params["amplitude"],
        params["edge_sigma"],
        params["shoulder_amplitude"],
        params["shoulder_sigma"],
    )
