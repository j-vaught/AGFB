"""Salt impulse noise."""

from __future__ import annotations

import torch

from agfb_noise.definitions.salt_pepper import add_salt_pepper
from agfb_noise.helpers.base import ClampRange, Numeric

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
    return add_salt_pepper(
        image,
        amount=amount,
        pepper=False,
        salt_value=salt_value,
        seed=seed,
        generator=generator,
        clamp=clamp,
    )
