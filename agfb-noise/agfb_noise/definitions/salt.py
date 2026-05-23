"""Salt impulse noise."""

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
        "name": "salt",
        "function": "add_salt",
        "description": "random high-valued impulse replacement",
    },
)


def add_salt(
    image: torch.Tensor,
    *,
    amount: Numeric,
    salt_value: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random pixels with `salt_value`."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    salt_value_t = batch_param(salt_value, image, name="salt_value")
    validate_probability(amount_t, "amount")
    out = torch.where(rand_like(image, gen) < amount_t, salt_value_t.expand_as(image), image)
    return apply_clamp(out, clamp)
