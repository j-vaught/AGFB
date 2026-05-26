"""Sensor-physics artifact models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from agfb_noise.helpers.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    rand_like,
    randn_like,
    randn_shape,
    resolve_generator,
    validate_nonnegative,
    validate_positive,
    validate_probability,
)

NOISE_SPECS = (
    {
        "name": "prnu",
        "function": "add_prnu",
        "description": "photo-response nonuniformity gain variation",
    },
    {
        "name": "row_correlated_read_noise",
        "function": "add_row_correlated_read_noise",
        "description": "row-correlated read noise and horizontal banding",
        "aliases": ("row_read_noise", "banding_read_noise"),
    },
    {
        "name": "rts_noise",
        "function": "add_rts_noise",
        "description": "random telegraph signal pixel switching noise",
        "aliases": ("burst_noise",),
    },
    {
        "name": "saturation_clip",
        "function": "add_saturation_clip",
        "description": "sensor saturation and clipping",
    },
    {
        "name": "photon_transfer_chain",
        "function": "add_photon_transfer_chain",
        "description": "shot, dark, PRNU, DSNU, read, and ADC sensor chain",
    },
    {
        "name": "dsnu",
        "function": "add_dsnu",
        "description": "dark-signal nonuniformity offset variation",
    },
    {
        "name": "ktc_reset",
        "function": "add_ktc_reset",
        "description": "kTC reset noise offset",
    },
    {
        "name": "amp_glow",
        "function": "add_amp_glow",
        "description": "amplifier glow corner bias",
    },
    {
        "name": "blooming_smear",
        "function": "add_blooming_smear",
        "description": "vertical blooming and smear from saturated pixels",
    },
    {
        "name": "rolling_shutter",
        "function": "add_rolling_shutter",
        "description": "row-wise rolling-shutter displacement",
    },
    {
        "name": "adc_nonlinearity",
        "function": "add_adc_nonlinearity",
        "description": "ADC integral and differential nonlinearity artifact",
        "aliases": ("dnl_inl",),
    },
    {
        "name": "hot_pixel_clusters",
        "function": "add_hot_pixel_clusters",
        "description": "clustered hot-pixel defects",
    },
    {
        "name": "vignetting",
        "function": "add_vignetting",
        "description": "radial lens vignetting falloff",
    },
)


def add_prnu(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply multiplicative photo-response nonuniformity."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    validate_nonnegative(sigma_t, "sigma")
    return apply_clamp(image * (1.0 + randn_like(image, gen) * sigma_t), clamp)


def add_row_correlated_read_noise(
    image: torch.Tensor,
    *,
    row_sigma: Numeric,
    pixel_sigma: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add row-wise offsets plus optional independent read noise."""
    image = check_image(image)
    _validate_spatial_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    row_sigma_t = batch_param(row_sigma, image, name="row_sigma")
    pixel_sigma_t = batch_param(pixel_sigma, image, name="pixel_sigma")
    validate_nonnegative(row_sigma_t, "row_sigma")
    validate_nonnegative(pixel_sigma_t, "pixel_sigma")
    row_shape = (*image.shape[:-1], 1)
    row_noise = randn_shape(row_shape, image, gen) * row_sigma_t
    pixel_noise = randn_like(image, gen) * pixel_sigma_t
    return apply_clamp(image + row_noise + pixel_noise, clamp)


def add_rts_noise(
    image: torch.Tensor,
    *,
    amount: Numeric,
    amplitude: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add sparse two-state random telegraph offsets."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    amplitude_t = batch_param(amplitude, image, name="amplitude")
    validate_probability(amount_t, "amount")
    validate_nonnegative(amplitude_t, "amplitude")
    active = rand_like(image, gen) < amount_t
    sign = torch.where(rand_like(image, gen) < 0.5, -1.0, 1.0)
    return apply_clamp(image + torch.where(active, sign * amplitude_t, 0.0), clamp)


def add_saturation_clip(
    image: torch.Tensor,
    *,
    min_value: Numeric = 0.0,
    max_value: Numeric = 1.0,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Clip values to a sensor saturation interval."""
    image = check_image(image)
    low = torch.as_tensor(min_value, dtype=image.dtype, device=image.device)
    high = torch.as_tensor(max_value, dtype=image.dtype, device=image.device)
    if bool((high <= low).any().item()):
        raise ValueError("max_value must be greater than min_value")
    return apply_clamp(torch.clamp(image, min=low, max=high), clamp)


def add_photon_transfer_chain(
    image: torch.Tensor,
    *,
    peak: Numeric,
    read_sigma: Numeric = 0.0,
    prnu_sigma: Numeric = 0.0,
    dsnu_sigma: Numeric = 0.0,
    dark_current: Numeric = 0.0,
    black_level: Numeric = 0.0,
    levels: int | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Simulate a compact photon-transfer sensor chain in normalized units."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    peak_t = batch_param(peak, image, name="peak")
    read_sigma_t = batch_param(read_sigma, image, name="read_sigma")
    prnu_sigma_t = batch_param(prnu_sigma, image, name="prnu_sigma")
    dsnu_sigma_t = batch_param(dsnu_sigma, image, name="dsnu_sigma")
    dark_current_t = batch_param(dark_current, image, name="dark_current")
    black_level_t = batch_param(black_level, image, name="black_level")
    validate_positive(peak_t, "peak")
    validate_nonnegative(read_sigma_t, "read_sigma")
    validate_nonnegative(prnu_sigma_t, "prnu_sigma")
    validate_nonnegative(dsnu_sigma_t, "dsnu_sigma")
    validate_nonnegative(dark_current_t, "dark_current")
    signal = image.clamp_min(0.0) * (1.0 + randn_like(image, gen) * prnu_sigma_t)
    lam = (signal + dark_current_t).clamp_min(0.0) * peak_t
    counts = torch.poisson(lam) if gen is None else torch.poisson(lam, generator=gen)
    out = counts / peak_t
    out = out + randn_like(image, gen) * read_sigma_t + randn_like(image, gen) * dsnu_sigma_t
    out = out + black_level_t
    if levels is not None:
        levels_int = int(levels)
        if levels_int < 2:
            raise ValueError("levels must be at least two")
        out = torch.round(out.clamp(0.0, 1.0) * (levels_int - 1)) / float(levels_int - 1)
    return apply_clamp(out, clamp)


def add_dsnu(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add dark-signal nonuniformity offsets."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    validate_nonnegative(sigma_t, "sigma")
    return apply_clamp(image + randn_like(image, gen) * sigma_t, clamp)


def add_ktc_reset(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add reset-noise offsets."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    validate_nonnegative(sigma_t, "sigma")
    return apply_clamp(image + randn_like(image, gen) * sigma_t, clamp)


def add_amp_glow(
    image: torch.Tensor,
    *,
    strength: Numeric,
    corner: str = "bottom_right",
    decay: Numeric = 3.0,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add exponential corner glow."""
    image = check_image(image)
    _validate_spatial_image(image)
    strength_t = batch_param(strength, image, name="strength")
    decay_t = batch_param(decay, image, name="decay")
    validate_nonnegative(strength_t, "strength")
    validate_positive(decay_t, "decay")
    yy, xx = _normalized_grid(image)
    cy = 1.0 if "bottom" in corner else -1.0
    cx = 1.0 if "right" in corner else -1.0
    distance = torch.sqrt((yy - cy).square() + (xx - cx).square())
    glow = torch.exp(-decay_t * distance)
    return apply_clamp(image + strength_t * glow, clamp)


def add_blooming_smear(
    image: torch.Tensor,
    *,
    threshold: Numeric = 1.0,
    strength: Numeric = 0.25,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Smear saturated excess vertically along columns."""
    image = check_image(image)
    _validate_spatial_image(image)
    threshold_t = batch_param(threshold, image, name="threshold")
    strength_t = batch_param(strength, image, name="strength")
    validate_nonnegative(strength_t, "strength")
    excess = (image - threshold_t).clamp_min(0.0)
    forward = torch.cumsum(excess, dim=-2)
    backward = torch.flip(torch.cumsum(torch.flip(excess, dims=(-2,)), dim=-2), dims=(-2,))
    smear = (forward + backward) / float(image.shape[-2])
    return apply_clamp(image.clamp_max(threshold_t) + strength_t * smear, clamp)


def add_rolling_shutter(
    image: torch.Tensor,
    *,
    max_shift: int = 4,
    direction: str = "horizontal",
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply a deterministic row-wise rolling-shutter skew."""
    image = check_image(image)
    _validate_spatial_image(image)
    shift = int(max_shift)
    if direction != "horizontal":
        raise ValueError("direction must be 'horizontal'")
    width = image.shape[-1]
    row_shifts = torch.linspace(-shift, shift, image.shape[-2], device=image.device)
    row_shifts = row_shifts.round().to(torch.long)
    columns = torch.arange(width, device=image.device).view(1, width)
    gather = (columns - row_shifts.view(-1, 1)).remainder(width)
    gather = gather.view(*((1,) * (image.ndim - 2)), image.shape[-2], width).expand_as(image)
    return apply_clamp(torch.gather(image, dim=-1, index=gather), clamp)


def add_adc_nonlinearity(
    image: torch.Tensor,
    *,
    levels: int = 4096,
    inl_amplitude: Numeric = 0.0,
    dnl_amplitude: Numeric = 0.0,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Quantize with smooth integral and differential nonlinearity."""
    image = check_image(image)
    levels_int = int(levels)
    if levels_int < 2:
        raise ValueError("levels must be at least two")
    inl_t = torch.as_tensor(inl_amplitude, dtype=image.dtype, device=image.device)
    dnl_t = torch.as_tensor(dnl_amplitude, dtype=image.dtype, device=image.device)
    validate_nonnegative(inl_t, "inl_amplitude")
    validate_nonnegative(dnl_t, "dnl_amplitude")
    x = image.clamp(0.0, 1.0)
    warped = (x + inl_t * torch.sin(2.0 * torch.pi * x)).clamp(0.0, 1.0)
    codes = torch.round(warped * (levels_int - 1))
    out = codes / float(levels_int - 1)
    out = (out + dnl_t * torch.sin(2.0 * torch.pi * codes / float(levels_int))).clamp(0.0, 1.0)
    return apply_clamp(out, clamp)


def add_hot_pixel_clusters(
    image: torch.Tensor,
    *,
    amount: Numeric,
    radius: int = 1,
    hot_value: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random clustered defects with a hot value."""
    image = check_image(image)
    _validate_spatial_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    hot_value_t = batch_param(hot_value, image, name="hot_value")
    validate_probability(amount_t, "amount")
    radius_int = int(radius)
    if radius_int < 0:
        raise ValueError("radius must be nonnegative")
    mask = (rand_like(image, gen) < amount_t).to(image.dtype)
    if radius_int > 0:
        mask = _max_pool_spatial(mask, kernel_size=2 * radius_int + 1)
    return apply_clamp(torch.where(mask > 0, hot_value_t.expand_as(image), image), clamp)


def add_vignetting(
    image: torch.Tensor,
    *,
    strength: Numeric = 0.5,
    power: Numeric = 2.0,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply radial vignetting attenuation."""
    image = check_image(image)
    _validate_spatial_image(image)
    strength_t = batch_param(strength, image, name="strength")
    power_t = batch_param(power, image, name="power")
    validate_nonnegative(strength_t, "strength")
    validate_positive(power_t, "power")
    yy, xx = _normalized_grid(image)
    radius = torch.sqrt(xx.square() + yy.square()).clamp(0.0, 1.0)
    falloff = (1.0 - strength_t * radius.pow(power_t)).clamp_min(0.0)
    return apply_clamp(image * falloff, clamp)


def _normalized_grid(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, dtype=image.dtype, device=image.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=image.dtype, device=image.device)
    prefix = (1,) * (image.ndim - 2)
    return y.view(*prefix, height, 1), x.view(*prefix, 1, width)


def _max_pool_spatial(image: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    original_shape = image.shape
    flat = image.reshape(-1, 1, *original_shape[-2:])
    pooled = F.max_pool2d(flat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return pooled.reshape(original_shape)


def _validate_spatial_image(image: torch.Tensor) -> None:
    if image.ndim < 2:
        raise ValueError("image must have at least two spatial dimensions")
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("spatial dimensions must be at least 2 pixels")
