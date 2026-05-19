"""Localized rotated Gabor packet generator."""

from __future__ import annotations

import math

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    infer_batch_size,
    infer_device,
    pack,
)


def gabor_packet(
    height: int,
    width: int,
    *,
    carrier_frequency: Numeric,
    angle_rad: Numeric,
    envelope_length_sigma: Numeric,
    envelope_width_sigma: Numeric,
    center_x: Numeric = 0.0,
    center_y: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    phase_rad: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched sinusoid inside a rotated Gaussian envelope.

    The benchmark uses this generator when a filter should handle a local
    frequency pattern instead of a full-image grating. It is useful for
    checking whether a response is localized to an oriented packet and whether
    the analytic gradient reflects both the carrier wave and the envelope
    falloff.

    `carrier_frequency` is measured in cycles per pixel along the oriented
    carrier coordinate. `angle_rad` is that carrier coordinate angle in
    radians, measured from the image `+x` direction. `envelope_length_sigma`
    controls the Gaussian window along the carrier coordinate, and
    `envelope_width_sigma` controls the perpendicular window width. `center_x`
    and `center_y` move the packet center in the shared centered coordinate
    system. `amplitude` scales the output, and `phase_rad` shifts the carrier
    sinusoid.

    The returned `Frame` contains the intensity image and the closed-form
    gradients with respect to image `x` and `y`. If `device` is omitted and a
    tensor parameter is passed, the render stays on that tensor's device.
    """
    device = infer_device(
        device,
        carrier_frequency,
        angle_rad,
        envelope_length_sigma,
        envelope_width_sigma,
        center_x,
        center_y,
        amplitude,
        phase_rad,
    )
    batch_size = infer_batch_size(
        carrier_frequency,
        angle_rad,
        envelope_length_sigma,
        envelope_width_sigma,
        center_x,
        center_y,
        amplitude,
        phase_rad,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    carrier_frequency_batch = as_batch(carrier_frequency, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    envelope_length_sigma_batch = as_batch(envelope_length_sigma, batch_size, device, dtype)
    envelope_width_sigma_batch = as_batch(envelope_width_sigma, batch_size, device, dtype)
    center_x_batch = as_batch(center_x, batch_size, device, dtype)
    center_y_batch = as_batch(center_y, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    phase_batch = as_batch(phase_rad, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    x_from_center = xx - center_x_batch
    y_from_center = yy - center_y_batch
    carrier_coord = x_from_center * cos_angle + y_from_center * sin_angle
    transverse_coord = -x_from_center * sin_angle + y_from_center * cos_angle
    envelope_length_sigma_sq = envelope_length_sigma_batch * envelope_length_sigma_batch
    envelope_width_sigma_sq = envelope_width_sigma_batch * envelope_width_sigma_batch

    envelope = torch.exp(
        -0.5
        * (
            (carrier_coord * carrier_coord) / envelope_length_sigma_sq
            + (transverse_coord * transverse_coord) / envelope_width_sigma_sq
        )
    )
    phase_arg = 2.0 * math.pi * carrier_frequency_batch * carrier_coord + phase_batch
    sin_phase = torch.sin(phase_arg)
    cos_phase = torch.cos(phase_arg)

    intensity = amplitude_batch * envelope * sin_phase
    carrier_derivative = (
        amplitude_batch
        * envelope
        * (
            (2.0 * math.pi * carrier_frequency_batch) * cos_phase
            - (carrier_coord / envelope_length_sigma_sq) * sin_phase
        )
    )
    transverse_derivative = (
        amplitude_batch * envelope * (-(transverse_coord / envelope_width_sigma_sq) * sin_phase)
    )
    gradient_x = carrier_derivative * cos_angle - transverse_derivative * sin_angle
    gradient_y = carrier_derivative * sin_angle + transverse_derivative * cos_angle
    return pack(intensity, gradient_x, gradient_y)
