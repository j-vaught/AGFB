"""Linear chirp grating generator."""

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
    normalize_contrast,
    validate_amplitude,
)


def chirp(
    height: int,
    width: int,
    *,
    base_frequency: Numeric,
    frequency_slope: Numeric,
    angle_rad: Numeric,
    amplitude: Numeric = 1.0,
    phase_rad: Numeric = 0.0,
    center_offset: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched oriented sinusoid whose frequency changes linearly.

    The benchmark uses this generator to test whether filters handle local
    frequency changes instead of assuming one constant wavelength. It is the
    scale and frequency family case where the frequency is known at every
    pixel and the analytic gradient should grow or shrink with that local
    frequency.

    `base_frequency` is the frequency in cycles per pixel at the oriented
    coordinate origin. `frequency_slope` is the linear change in frequency per
    pixel along that coordinate. `angle_rad` is the coordinate-axis angle in
    radians measured from the image `+x` direction. `center_offset` shifts the
    coordinate origin, `phase_rad` shifts the sinusoid phase, and `amplitude`
    controls the realized peak-to-trough contrast.

    The phase is
    `2 pi * (base_frequency * u + 0.5 * frequency_slope * u^2) + phase_rad`,
    where `u = x * cos(angle) + y * sin(angle) - center_offset`. The raw chirp
    is affinely normalized into `[0, 1]`. The returned `Frame` contains the
    intensity image and the closed-form gradients with respect to image `x`
    and `y`. If `device` is omitted and a tensor parameter is passed, the
    render stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    device = infer_device(
        device,
        base_frequency,
        frequency_slope,
        angle_rad,
        amplitude,
        phase_rad,
        center_offset,
    )
    batch_size = infer_batch_size(
        base_frequency,
        frequency_slope,
        angle_rad,
        amplitude,
        phase_rad,
        center_offset,
    )
    xx, yy = coord_grid(height, width, device, dtype)

    base_frequency_batch = as_batch(base_frequency, batch_size, device, dtype)
    frequency_slope_batch = as_batch(frequency_slope, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    phase_batch = as_batch(phase_rad, batch_size, device, dtype)
    center_offset_batch = as_batch(center_offset, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    oriented_coord = xx * cos_angle + yy * sin_angle - center_offset_batch
    phase_arg = (
        2.0
        * math.pi
        * (
            base_frequency_batch * oriented_coord
            + 0.5 * frequency_slope_batch * oriented_coord * oriented_coord
        )
        + phase_batch
    )

    intensity = amplitude_batch * torch.sin(phase_arg)
    instantaneous_frequency = base_frequency_batch + frequency_slope_batch * oriented_coord
    normal_gradient = (
        amplitude_batch * torch.cos(phase_arg) * 2.0 * math.pi * instantaneous_frequency
    )
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)
