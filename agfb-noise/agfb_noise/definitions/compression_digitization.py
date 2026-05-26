"""Compression and digitization artifact models."""

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
    resolve_generator,
    validate_nonnegative,
    validate_probability,
)

NOISE_SPECS = (
    {
        "name": "jpeg",
        "function": "add_jpeg",
        "description": "blocky JPEG-like quantization artifact",
    },
    {
        "name": "gradient_banding",
        "function": "add_gradient_banding",
        "description": "low-amplitude gradient color banding",
    },
    {
        "name": "overshoot",
        "function": "add_overshoot",
        "description": "edge overshoot and ringing artifact",
    },
    {
        "name": "block_dropout",
        "function": "add_block_dropout",
        "description": "random macroblock dropout artifact",
    },
    {
        "name": "aliasing",
        "function": "add_aliasing",
        "description": "downsampled and resampled aliasing artifact",
    },
    {
        "name": "mosquito_noise",
        "function": "add_mosquito_noise",
        "description": "edge-localized high-frequency mosquito noise",
    },
    {
        "name": "wavelet_ringing",
        "function": "add_wavelet_ringing",
        "description": "wavelet-compression-style ringing artifact",
    },
    {
        "name": "posterization",
        "function": "add_posterization",
        "description": "posterization from low tonal depth",
    },
    {
        "name": "dither",
        "function": "add_dither",
        "description": "additive dither before optional quantization",
    },
)


def add_jpeg(
    image: torch.Tensor,
    *,
    quality: Numeric = 50.0,
    block_size: int = 8,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply JPEG-like block averaging and scalar quantization."""
    image = check_image(image)
    _validate_spatial_image(image)
    quality_t = torch.as_tensor(quality, dtype=image.dtype, device=image.device)
    if bool(((quality_t <= 0) | (quality_t > 100)).any().item()):
        raise ValueError("quality must be in (0, 100]")
    block = int(block_size)
    if block < 1:
        raise ValueError("block_size must be positive")
    levels = torch.round(quality_t / 100.0 * 255.0).clamp_min(2.0)
    quantized = torch.round(image.clamp(0.0, 1.0) * (levels - 1.0)) / (levels - 1.0)
    blocky = _block_average(quantized, block)
    mix = ((100.0 - quality_t) / 100.0).clamp(0.0, 1.0)
    return apply_clamp(quantized * (1.0 - mix) + blocky * mix, clamp)


def add_gradient_banding(
    image: torch.Tensor,
    *,
    levels: int = 32,
    strength: Numeric = 1.0,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Quantize tonal gradients to visible bands."""
    image = check_image(image)
    levels_int = int(levels)
    if levels_int < 2:
        raise ValueError("levels must be at least two")
    strength_t = batch_param(strength, image, name="strength")
    validate_nonnegative(strength_t, "strength")
    banded = torch.round(image.clamp(0.0, 1.0) * (levels_int - 1)) / float(levels_int - 1)
    return apply_clamp(image * (1.0 - strength_t) + banded * strength_t, clamp)


def add_overshoot(
    image: torch.Tensor,
    *,
    strength: Numeric = 0.1,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add edge overshoot using an unsharp high-pass residual."""
    image = check_image(image)
    _validate_spatial_image(image)
    strength_t = batch_param(strength, image, name="strength")
    validate_nonnegative(strength_t, "strength")
    blurred = _blur_spatial(image)
    return apply_clamp(image + (image - blurred) * strength_t, clamp)


def add_block_dropout(
    image: torch.Tensor,
    *,
    amount: Numeric,
    block_size: int = 16,
    fill_value: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random macroblocks with `fill_value`."""
    image = check_image(image)
    _validate_spatial_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    fill_t = batch_param(fill_value, image, name="fill_value")
    validate_probability(amount_t, "amount")
    block = int(block_size)
    if block < 1:
        raise ValueError("block_size must be positive")
    low_shape = (
        *image.shape[:-2],
        _ceil_div(image.shape[-2], block),
        _ceil_div(image.shape[-1], block),
    )
    low_mask = torch.zeros(low_shape, dtype=image.dtype, device=image.device)
    low_mask = torch.where(rand_like(low_mask, gen) < amount_t, 1.0, low_mask)
    mask = _nearest_resize(low_mask, image.shape[-2:])
    return apply_clamp(torch.where(mask > 0, fill_t.expand_as(image), image), clamp)


def add_aliasing(
    image: torch.Tensor,
    *,
    factor: int = 2,
    mode: str = "nearest",
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Downsample and upsample without adequate antialiasing."""
    image = check_image(image)
    _validate_spatial_image(image)
    factor_int = int(factor)
    if factor_int < 2:
        raise ValueError("factor must be at least two")
    flat, shape = _flatten_spatial(image)
    low = F.interpolate(flat, scale_factor=1.0 / factor_int, mode="nearest")
    if mode == "nearest":
        out = F.interpolate(low, size=image.shape[-2:], mode="nearest")
    elif mode in {"bilinear", "bicubic"}:
        out = F.interpolate(low, size=image.shape[-2:], mode=mode, align_corners=False)
    else:
        raise ValueError("mode must be 'nearest', 'bilinear', or 'bicubic'")
    return apply_clamp(_unflatten_spatial(out, shape), clamp)


def add_mosquito_noise(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    edge_threshold: Numeric = 0.02,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add high-frequency noise concentrated around edges."""
    image = check_image(image)
    _validate_spatial_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    edge_threshold_t = batch_param(edge_threshold, image, name="edge_threshold")
    validate_nonnegative(sigma_t, "sigma")
    validate_nonnegative(edge_threshold_t, "edge_threshold")
    edges = (image - _blur_spatial(image)).abs()
    edge_weight = (edges / edge_threshold_t.clamp_min(torch.finfo(image.dtype).eps)).clamp(0.0, 1.0)
    high_frequency = randn_like(image, gen) - _blur_spatial(randn_like(image, gen))
    return apply_clamp(image + high_frequency * sigma_t * edge_weight, clamp)


def add_wavelet_ringing(
    image: torch.Tensor,
    *,
    strength: Numeric = 0.08,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add smooth ringing from a coarse residual approximation."""
    image = check_image(image)
    _validate_spatial_image(image)
    strength_t = batch_param(strength, image, name="strength")
    validate_nonnegative(strength_t, "strength")
    coarse = _nearest_resize(_block_average(image, 4)[..., ::4, ::4], image.shape[-2:])
    ringing = _blur_spatial(image - coarse)
    return apply_clamp(image + ringing * strength_t, clamp)


def add_posterization(
    image: torch.Tensor,
    *,
    levels: int = 8,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Reduce tonal values to a small number of levels."""
    image = check_image(image)
    levels_int = int(levels)
    if levels_int < 2:
        raise ValueError("levels must be at least two")
    out = torch.round(image.clamp(0.0, 1.0) * (levels_int - 1)) / float(levels_int - 1)
    return apply_clamp(out, clamp)


def add_dither(
    image: torch.Tensor,
    *,
    amplitude: Numeric,
    levels: int | None = None,
    triangular: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add uniform or triangular dither before optional quantization."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amplitude_t = batch_param(amplitude, image, name="amplitude")
    validate_nonnegative(amplitude_t, "amplitude")
    noise = rand_like(image, gen) - 0.5
    if triangular:
        noise = noise + rand_like(image, gen) - 0.5
    out = image + noise * amplitude_t
    if levels is not None:
        levels_int = int(levels)
        if levels_int < 2:
            raise ValueError("levels must be at least two")
        out = torch.round(out.clamp(0.0, 1.0) * (levels_int - 1)) / float(levels_int - 1)
    return apply_clamp(out, clamp)


def _block_average(image: torch.Tensor, block_size: int) -> torch.Tensor:
    flat, shape = _flatten_spatial(image)
    pad_h = (-image.shape[-2]) % block_size
    pad_w = (-image.shape[-1]) % block_size
    padded = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    pooled = F.avg_pool2d(padded, kernel_size=block_size, stride=block_size)
    resized = F.interpolate(pooled, size=padded.shape[-2:], mode="nearest")
    resized = resized[..., : image.shape[-2], : image.shape[-1]]
    return _unflatten_spatial(resized, shape)


def _blur_spatial(image: torch.Tensor) -> torch.Tensor:
    flat, shape = _flatten_spatial(image)
    blurred = F.avg_pool2d(F.pad(flat, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    return _unflatten_spatial(blurred, shape)


def _nearest_resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    flat, shape = _flatten_spatial(image)
    resized = F.interpolate(flat, size=size, mode="nearest")
    return _unflatten_spatial(resized, (*shape[:-2], *size))


def _flatten_spatial(image: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    shape = image.shape
    return image.reshape(-1, 1, *shape[-2:]), shape


def _unflatten_spatial(image: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    return image.reshape(*shape[:-2], *image.shape[-2:])


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_spatial_image(image: torch.Tensor) -> None:
    if image.ndim < 2:
        raise ValueError("image must have at least two spatial dimensions")
    if image.shape[-2] < 2 or image.shape[-1] < 2:
        raise ValueError("spatial dimensions must be at least 2 pixels")
