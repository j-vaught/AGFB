"""Spatially correlated Gaussian and speckle noise."""

from __future__ import annotations

import math

import torch

from agfb_noise.helpers.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    randn_like,
    resolve_generator,
    validate_nonnegative,
    validate_positive,
)

NOISE_SPECS = (
    {
        "name": "correlated_gaussian",
        "function": "add_correlated_gaussian",
        "description": "additive spatially correlated Gaussian random field noise",
        "aliases": ("gaussian_random_field", "grf"),
    },
    {
        "name": "powerlaw_gaussian",
        "function": "add_powerlaw_gaussian",
        "description": "additive power-law colored Gaussian noise",
        "aliases": ("colored_gaussian", "colored_noise"),
    },
    {
        "name": "anisotropic_gaussian",
        "function": "add_anisotropic_gaussian",
        "description": "additive anisotropic Gaussian random field noise",
    },
    {
        "name": "correlated_speckle",
        "function": "add_correlated_speckle",
        "description": "multiplicative spatially correlated Gaussian speckle noise",
    },
)


def add_correlated_gaussian(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    correlation_length: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add isotropic Gaussian random field noise with a Gaussian correlation kernel."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    mean_t = batch_param(mean, image, name="mean")
    length_t = batch_param(correlation_length, image, name="correlation_length")
    validate_nonnegative(sigma_t, "sigma")
    validate_positive(length_t, "correlation_length")
    noise = _correlated_field(image, gen, length_y=length_t, length_x=length_t)
    return apply_clamp(image + noise * sigma_t + mean_t, clamp)


def add_powerlaw_gaussian(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    beta: Numeric = 1.0,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add colored Gaussian noise whose power spectral density scales as 1/f^beta."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    mean_t = batch_param(mean, image, name="mean")
    beta_t = batch_param(beta, image, name="beta")
    validate_nonnegative(sigma_t, "sigma")
    noise = _powerlaw_field(image, gen, beta_t)
    return apply_clamp(image + noise * sigma_t + mean_t, clamp)


def add_anisotropic_gaussian(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    correlation_length_y: Numeric,
    correlation_length_x: Numeric,
    angle: Numeric = 0.0,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add anisotropic Gaussian random field noise with rotated correlation axes."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    mean_t = batch_param(mean, image, name="mean")
    length_y_t = batch_param(correlation_length_y, image, name="correlation_length_y")
    length_x_t = batch_param(correlation_length_x, image, name="correlation_length_x")
    angle_t = batch_param(angle, image, name="angle")
    validate_nonnegative(sigma_t, "sigma")
    validate_positive(length_y_t, "correlation_length_y")
    validate_positive(length_x_t, "correlation_length_x")
    noise = _correlated_field(
        image,
        gen,
        length_y=length_y_t,
        length_x=length_x_t,
        angle=angle_t,
    )
    return apply_clamp(image + noise * sigma_t + mean_t, clamp)


def add_correlated_speckle(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    correlation_length: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply `image + image * n`, where `n` is spatially correlated normal noise."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    mean_t = batch_param(mean, image, name="mean")
    length_t = batch_param(correlation_length, image, name="correlation_length")
    validate_nonnegative(sigma_t, "sigma")
    validate_positive(length_t, "correlation_length")
    multiplier_noise = _correlated_field(image, gen, length_y=length_t, length_x=length_t)
    multiplier_noise = multiplier_noise * sigma_t + mean_t
    return apply_clamp(image + image * multiplier_noise, clamp)


def _correlated_field(
    image: torch.Tensor,
    generator: torch.Generator | None,
    *,
    length_y: torch.Tensor,
    length_x: torch.Tensor,
    angle: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_spatial_image(image)
    frequency_y, frequency_x = _frequency_grid(image)
    if angle is None:
        rotated_y = frequency_y
        rotated_x = frequency_x
    else:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotated_y = frequency_y * cos_angle - frequency_x * sin_angle
        rotated_x = frequency_y * sin_angle + frequency_x * cos_angle
    spectrum_filter = torch.exp(
        -2.0 * (math.pi**2) * ((length_y * rotated_y) ** 2 + (length_x * rotated_x) ** 2)
    )
    white = randn_like(image, generator)
    filtered = torch.fft.irfft2(
        torch.fft.rfft2(white, dim=(-2, -1)) * spectrum_filter,
        s=image.shape[-2:],
        dim=(-2, -1),
    )
    return _standardize_spatial(filtered)


def _powerlaw_field(
    image: torch.Tensor,
    generator: torch.Generator | None,
    beta: torch.Tensor,
) -> torch.Tensor:
    _validate_spatial_image(image)
    frequency_y, frequency_x = _frequency_grid(image)
    radius = torch.sqrt(frequency_y.square() + frequency_x.square())
    min_frequency = 1.0 / float(max(image.shape[-2:]))
    amplitude = radius.clamp_min(min_frequency).pow(-0.5 * beta)
    amplitude = torch.where(
        radius == 0, torch.zeros((), dtype=image.dtype, device=image.device), amplitude
    )
    white = randn_like(image, generator)
    filtered = torch.fft.irfft2(
        torch.fft.rfft2(white, dim=(-2, -1)) * amplitude,
        s=image.shape[-2:],
        dim=(-2, -1),
    )
    return _standardize_spatial(filtered)


def _frequency_grid(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    freq_y = torch.fft.fftfreq(height, d=1.0, device=image.device)
    freq_x = torch.fft.rfftfreq(width, d=1.0, device=image.device)
    dtype = image.dtype
    prefix = (1,) * (image.ndim - 2)
    frequency_y = freq_y.to(dtype=dtype).view(*prefix, height, 1)
    frequency_x = freq_x.to(dtype=dtype).view(*prefix, 1, freq_x.shape[0])
    return frequency_y, frequency_x


def _standardize_spatial(field: torch.Tensor) -> torch.Tensor:
    dims = (-2, -1)
    centered = field - field.mean(dim=dims, keepdim=True)
    scale = centered.std(dim=dims, keepdim=True, unbiased=False).clamp_min(
        torch.finfo(field.dtype).eps
    )
    return centered / scale


def _validate_spatial_image(image: torch.Tensor) -> None:
    if image.ndim < 2:
        raise ValueError("image must have at least two spatial dimensions")
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("spatial dimensions must be at least 2 pixels")
