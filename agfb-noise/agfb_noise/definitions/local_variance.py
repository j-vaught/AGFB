"""Gaussian noise with local variance."""

from __future__ import annotations

import torch

from agfb_noise.helpers.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    image_param,
    randn_like,
    resolve_generator,
    validate_nonnegative,
)

NOISE_SPECS = (
    {
        "name": "local_variance",
        "function": "add_local_variance",
        "description": "additive Gaussian noise with scalar, batched, or per-pixel variance",
        "aliases": ("localvar", "local_variance_gaussian"),
    },
)


def add_local_variance(
    image: torch.Tensor,
    *,
    variance: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add Gaussian noise whose variance may vary by image or pixel."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    variance_t = image_param(variance, image, name="variance")
    mean_t = batch_param(mean, image, name="mean")
    validate_nonnegative(variance_t, "variance")
    noise = randn_like(image, gen) * torch.sqrt(variance_t) + mean_t
    return apply_clamp(image + noise, clamp)
