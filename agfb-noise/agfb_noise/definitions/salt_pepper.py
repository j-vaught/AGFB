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
    salt: bool = True,
    pepper: bool = True,
    salt_vs_pepper: Numeric = 0.5,
    salt_value: Numeric = 1.0,
    pepper_value: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random pixels with optional salt and pepper values."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    validate_probability(amount_t, "amount")

    if not isinstance(salt, bool):
        raise TypeError("salt must be a bool")
    if not isinstance(pepper, bool):
        raise TypeError("pepper must be a bool")
    if not salt and not pepper:
        return apply_clamp(image, clamp)

    salt_value_t = batch_param(salt_value, image, name="salt_value")
    pepper_value_t = batch_param(pepper_value, image, name="pepper_value")
    draws = rand_like(image, gen)

    if salt and not pepper:
        return apply_clamp(
            torch.where(draws < amount_t, salt_value_t.expand_as(image), image), clamp
        )
    if pepper and not salt:
        return apply_clamp(
            torch.where(draws < amount_t, pepper_value_t.expand_as(image), image), clamp
        )

    salt_vs_pepper_t = batch_param(salt_vs_pepper, image, name="salt_vs_pepper")
    validate_probability(salt_vs_pepper_t, "salt_vs_pepper")
    salt_threshold = amount_t * salt_vs_pepper_t
    pepper_threshold = amount_t
    salted = draws < salt_threshold
    peppered = (draws >= salt_threshold) & (draws < pepper_threshold)
    out = torch.where(salted, salt_value_t.expand_as(image), image)
    out = torch.where(peppered, pepper_value_t.expand_as(image), out)
    return apply_clamp(out, clamp)
