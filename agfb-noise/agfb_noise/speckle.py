"""Multiplicative Gaussian speckle noise."""

from __future__ import annotations

import torch

from agfb_noise.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    randn_like,
    resolve_generator,
    validate_nonnegative,
)

NOISE_SPECS = (
    {
        "name": "speckle",
        "function": "add_speckle",
        "description": "multiplicative Gaussian speckle noise",
    },
)


def add_speckle(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply `image + image * n`, where `n` is normal noise."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    mean_t = batch_param(mean, image, name="mean")
    validate_nonnegative(sigma_t, "sigma")
    multiplier_noise = randn_like(image, gen) * sigma_t + mean_t
    return apply_clamp(image + image * multiplier_noise, clamp)
