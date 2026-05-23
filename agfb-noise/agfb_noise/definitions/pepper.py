"""Pepper impulse noise."""

from __future__ import annotations

import torch

from agfb_noise.definitions.salt_pepper import add_salt_pepper
from agfb_noise.helpers.base import ClampRange, Numeric

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
    return add_salt_pepper(
        image,
        amount=amount,
        salt=False,
        pepper_value=pepper_value,
        seed=seed,
        generator=generator,
        clamp=clamp,
    )
