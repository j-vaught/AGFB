"""Additive independent Gaussian noise."""

from __future__ import annotations

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
)

NOISE_SPECS = (
    {
        "name": "gaussian",
        "function": "add_gaussian",
        "description": "additive independent Gaussian noise",
        "aliases": ("normal", "awgn"),
    },
)


def add_gaussian(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add independent normal noise to every pixel."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    mean_t = batch_param(mean, image, name="mean")
    validate_nonnegative(sigma_t, "sigma")
    noise = randn_like(image, gen) * sigma_t + mean_t
    return apply_clamp(image + noise, clamp)
