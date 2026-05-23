"""Salt-and-pepper impulse noise."""

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
        "name": "salt_pepper",
        "function": "add_salt_pepper",
        "description": "random low- and high-valued impulse replacement",
        "aliases": ("s&p", "salt-and-pepper"),
    },
)


def add_salt_pepper(
    image: torch.Tensor,
    *,
    amount: Numeric,
    salt_vs_pepper: Numeric = 0.5,
    salt_value: Numeric = 1.0,
    pepper_value: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random pixels with salt and pepper values."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    salt_vs_pepper_t = batch_param(salt_vs_pepper, image, name="salt_vs_pepper")
    salt_value_t = batch_param(salt_value, image, name="salt_value")
    pepper_value_t = batch_param(pepper_value, image, name="pepper_value")
    validate_probability(amount_t, "amount")
    validate_probability(salt_vs_pepper_t, "salt_vs_pepper")
    draws = rand_like(image, gen)
    salt_threshold = amount_t * salt_vs_pepper_t
    pepper_threshold = amount_t
    salted = draws < salt_threshold
    peppered = (draws >= salt_threshold) & (draws < pepper_threshold)
    out = torch.where(salted, salt_value_t.expand_as(image), image)
    out = torch.where(peppered, pepper_value_t.expand_as(image), out)
    return apply_clamp(out, clamp)
