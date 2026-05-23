"""Additive independent uniform noise."""

from __future__ import annotations

import torch

from agfb_noise.helpers.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    rand_like,
    resolve_generator,
    validate_nonnegative,
)

NOISE_SPECS = (
    {
        "name": "uniform",
        "function": "add_uniform",
        "description": "additive uniform noise over a centered interval",
    },
)


def add_uniform(
    image: torch.Tensor,
    *,
    half_width: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add uniform noise on `[mean - half_width, mean + half_width]`."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    half_width_t = batch_param(half_width, image, name="half_width")
    mean_t = batch_param(mean, image, name="mean")
    validate_nonnegative(half_width_t, "half_width")
    noise = (2.0 * rand_like(image, gen) - 1.0) * half_width_t + mean_t
    return apply_clamp(image + noise, clamp)
