"""Two-dimensional polynomial field generator.

Polynomial fields provide exact closed-form gradients with configurable degree
and mixed terms, making them useful for checking whether filters reproduce
known local Taylor structure.
"""

from __future__ import annotations

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    infer_device,
    normalize_contrast,
    validate_amplitude,
)


def polynomial(
    height: int,
    width: int,
    *,
    coefficients: torch.Tensor | None = None,
    coordinate_scale: float | None = None,
    amplitude: Numeric = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a batched two-dimensional polynomial surface.

    The benchmark uses this generator when a filter should reproduce exact
    low-order image structure rather than respond only to edges, blobs, or
    ridges. Polynomial fields are useful for checking gradients over planar,
    quadratic, saddle, and mixed-term surfaces with known closed-form
    derivatives.

    `coefficients[b, i, j]` is the coefficient multiplying `x^i y^j` for batch
    item `b`. Passing a 2-D coefficient matrix renders one image; passing a
    3-D tensor renders a batch. If `coefficients` is omitted, the generator
    uses a degree-13 polynomial fit to the common MATLAB peaks demo surface for
    quick previews. `coordinate_scale` divides the centered image coordinates
    before evaluating the polynomial, which keeps coefficients readable on
    pixel-sized grids. If `coordinate_scale` is omitted, the shorter image side
    is scaled to span roughly `[-3, 3]`, matching the fitted default surface.
    `amplitude` controls the realized peak-to-trough contrast after affine
    normalization.

    The returned `Frame` contains the polynomial intensity image and the
    closed-form gradients with respect to image `x` and `y`. If `device` is
    omitted and a coefficient tensor is passed, the render stays on that
    tensor's device.
    """
    validate_amplitude("amplitude", amplitude)
    coordinate_scale_value = _coordinate_scale(height, width, coordinate_scale)
    if coordinate_scale_value == 0.0:
        raise ValueError("coordinate_scale must be nonzero")
    resolved_device = (
        infer_device(device, coefficients, amplitude)
        if coefficients is not None
        else infer_device(device, amplitude)
    )
    coefficient_batch = _coefficient_batch(coefficients, resolved_device, dtype)
    batch_size, x_term_count, y_term_count = coefficient_batch.shape
    if isinstance(amplitude, torch.Tensor) and amplitude.ndim == 1:
        amplitude_batch_size = int(amplitude.shape[0])
        coefficients_are_single = coefficients is None or coefficients.ndim == 2
        if batch_size == 1 and coefficients_are_single:
            coefficient_batch = coefficient_batch.expand(amplitude_batch_size, -1, -1)
            batch_size = amplitude_batch_size
        elif amplitude_batch_size != batch_size:
            raise ValueError(
                "amplitude tensor must match polynomial batch size, "
                f"got ({amplitude_batch_size},) for batch size {batch_size}"
            )
    amplitude_batch = as_batch(amplitude, batch_size, resolved_device, dtype)

    xx, yy = coord_grid(height, width, resolved_device, dtype)
    x_coord = xx / coordinate_scale_value
    y_coord = yy / coordinate_scale_value

    x_powers = _coordinate_powers(x_coord, x_term_count)
    y_powers = _coordinate_powers(y_coord, y_term_count)
    intensity = torch.einsum("bij,ihw,jhw->bhw", coefficient_batch, x_powers, y_powers)

    gradient_x = torch.zeros(batch_size, height, width, device=resolved_device, dtype=dtype)
    gradient_y = torch.zeros_like(gradient_x)
    if x_term_count > 1:
        x_exponents = torch.arange(1, x_term_count, device=resolved_device, dtype=dtype)
        x_derivative_coefficients = coefficient_batch[:, 1:, :] * x_exponents.view(1, -1, 1)
        gradient_x = torch.einsum(
            "bij,ihw,jhw->bhw",
            x_derivative_coefficients / coordinate_scale_value,
            x_powers[:-1],
            y_powers,
        )
    if y_term_count > 1:
        y_exponents = torch.arange(1, y_term_count, device=resolved_device, dtype=dtype)
        y_derivative_coefficients = coefficient_batch[:, :, 1:] * y_exponents.view(1, 1, -1)
        gradient_y = torch.einsum(
            "bij,ihw,jhw->bhw",
            y_derivative_coefficients / coordinate_scale_value,
            x_powers,
            y_powers[:-1],
        )
    return normalize_contrast(intensity, gradient_x, gradient_y, amplitude_batch)


def _coefficient_batch(
    coefficients: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if coefficients is None:
        return _default_coefficients(device, dtype)
    coefficient_batch = coefficients.to(device=device, dtype=dtype)
    if coefficient_batch.ndim == 2:
        coefficient_batch = coefficient_batch.unsqueeze(0)
    if coefficient_batch.ndim != 3:
        raise ValueError(
            "coefficients must have shape (x_terms, y_terms) or "
            f"(batch, x_terms, y_terms), got {tuple(coefficient_batch.shape)}"
        )
    return coefficient_batch


def _default_coefficients(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [
        [
            0.999303858,
            -2.975908265,
            1.120536878,
            7.641041290,
            -0.793536066,
            -2.577501173,
            0.195165147,
            0.261568094,
            -0.023701696,
            0.008054284,
            0.001442529,
            -0.002845236,
            -0.000035246,
            0.000127128,
        ],
        [
            -2.047202806,
            3.478445257,
            -2.809632579,
            -1.199088790,
            1.641388763,
            0.148513773,
            -0.337843120,
            -0.006920979,
            0.033502438,
            0.000001779,
            -0.001607518,
            0.000005739,
            0.000029473,
            0.0,
        ],
        [
            -0.184801218,
            -1.093290363,
            -0.284889297,
            -3.498588560,
            0.184633824,
            1.291231973,
            -0.035605067,
            -0.163436979,
            0.002827319,
            0.008328010,
            -0.000080656,
            -0.000128970,
            0.0,
            0.0,
        ],
        [
            7.171718999,
            -2.261160507,
            -0.717092553,
            0.623145885,
            -0.299900611,
            -0.060192154,
            0.066058843,
            0.002229569,
            -0.004728971,
            -0.000020077,
            0.000115050,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.247513716,
            1.614550828,
            0.032250870,
            0.333653302,
            -0.011541934,
            -0.159898239,
            0.001842681,
            0.015736821,
            -0.000082244,
            -0.000464140,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -4.135564721,
            0.568100180,
            0.695465420,
            -0.114907939,
            -0.014565013,
            0.007426196,
            -0.003265276,
            -0.000145417,
            0.000156430,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.119555427,
            -0.476916929,
            -0.006055438,
            0.025304288,
            -0.000013906,
            0.006009523,
            -0.000014678,
            -0.000401893,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.044583707,
            -0.068797553,
            -0.141430644,
            0.009027275,
            0.005469327,
            -0.000289627,
            -0.000001813,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.021597270,
            0.061024130,
            0.000865941,
            -0.005054156,
            0.000009426,
            0.000003953,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.136009192,
            0.004000752,
            0.011719184,
            -0.000256266,
            -0.000264761,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.001770214,
            -0.003608791,
            -0.000041054,
            0.000188276,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.008976521,
            -0.000088729,
            -0.000353439,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.000055087,
            0.000080005,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.000238420,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]
    return torch.tensor([values], device=device, dtype=dtype)


def _coordinate_powers(coordinate: torch.Tensor, term_count: int) -> torch.Tensor:
    powers = [torch.ones_like(coordinate)]
    for _ in range(1, term_count):
        powers.append(powers[-1] * coordinate)
    return torch.stack(powers)


def _coordinate_scale(height: int, width: int, coordinate_scale: float | None) -> float:
    if coordinate_scale is not None:
        return float(coordinate_scale)
    return (min(height, width) - 1) / 6.0
