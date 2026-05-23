"""Pepper impulse noise."""

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
    validate_probability,
)

NOISE_SPECS = (
    {
        "name": "pepper",
        "function": "add_pepper",
        "description": "random low-valued impulse replacement",
    },
)


def add_pepper(
    image: torch.Tensor,
    *,
    amount: Numeric,
    pepper_value: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random pixels with `pepper_value`."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    pepper_value_t = batch_param(pepper_value, image, name="pepper_value")
    validate_probability(amount_t, "amount")
    out = torch.where(rand_like(image, gen) < amount_t, pepper_value_t.expand_as(image), image)
    return apply_clamp(out, clamp)
