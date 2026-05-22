"""Random-valued impulse noise."""

from __future__ import annotations

import torch

from agfb_noise.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    rand_like,
    resolve_generator,
    validate_probability,
)

NOISE_SPECS = (
    {
        "name": "random_impulse",
        "function": "add_random_impulse",
        "description": "impulse noise with random replacement values",
        "aliases": ("random_valued_impulse", "rvin"),
    },
)


def add_random_impulse(
    image: torch.Tensor,
    *,
    amount: Numeric,
    low: Numeric = 0.0,
    high: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random pixels with values sampled uniformly from `[low, high]`."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    low_t = batch_param(low, image, name="low")
    high_t = batch_param(high, image, name="high")
    validate_probability(amount_t, "amount")
    replacement = low_t + (high_t - low_t) * rand_like(image, gen)
    out = torch.where(rand_like(image, gen) < amount_t, replacement, image)
    return apply_clamp(out, clamp)
