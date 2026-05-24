"""Single-frequency sinusoidal grating generator."""

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


def sinusoid(
    height: int,
    width: int,
    *,
    spatial_frequency: Numeric = 0.05,
    angle_rad: Numeric = 0.0,
    amplitude: Numeric = 1.0,
    phase_rad: Numeric = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched single-frequency sinusoidal grating.

    The benchmark uses this generator when a filter should respond to a pure
    periodic field with known spatial frequency and orientation. It is useful
    for checking frequency response, orientation selectivity, phase sensitivity,
    and whether analytic-gradient metrics behave correctly on signed fields.
    The visual notebook and cross-section document use this function as the
    constant-frequency grating example.

    The default call renders a horizontal 0.05 cycles/pixel sinusoid centered
    on the shared coordinate grid. `spatial_frequency` is measured in
    cycles/pixel. `angle_rad` is the grating normal direction in radians,
    measured from the image `+x` direction. `amplitude` controls the realized
    peak-to-trough contrast, and `phase_rad` shifts the sinusoid phase in radians.

    The projected coordinate is `s = x * cos(angle) + y * sin(angle)`. The
    raw grating is `sin(2 * pi * spatial_frequency * s + phase_rad)`, then it
    is affinely normalized into `[0, 1]`. The returned `Frame` contains that
    intensity image and the closed-form gradients with respect to image `x`
    and `y`. If `device` is omitted and a tensor parameter is passed, the
    render stays on that tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    device = infer_device(device, spatial_frequency, angle_rad, amplitude, phase_rad)
    batch_size = infer_batch_size(spatial_frequency, angle_rad, amplitude, phase_rad)
    xx, yy = coord_grid(height, width, device, dtype)

    spatial_frequency_batch = as_batch(spatial_frequency, batch_size, device, dtype)
    angle_batch = as_batch(angle_rad, batch_size, device, dtype)
    amplitude_batch = as_batch(amplitude, batch_size, device, dtype)
    phase_batch = as_batch(phase_rad, batch_size, device, dtype)

    cos_angle = torch.cos(angle_batch)
    sin_angle = torch.sin(angle_batch)
    grating_coord = xx * cos_angle + yy * sin_angle
    angular_frequency = 2.0 * math.pi * spatial_frequency_batch
    wave_phase = angular_frequency * grating_coord + phase_batch

    intensity = amplitude_batch * torch.sin(wave_phase)
    normal_gradient = angular_frequency * amplitude_batch * torch.cos(wave_phase)
    gradient_x = normal_gradient * cos_angle
    gradient_y = normal_gradient * sin_angle
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)
