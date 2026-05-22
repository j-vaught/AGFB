"""Multiplicative gamma speckle noise."""

from __future__ import annotations

import torch

from agfb_noise.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    check_image,
    rand_shape,
    resolve_generator,
)

NOISE_SPECS = (
    {
        "name": "gamma_speckle",
        "function": "add_gamma_speckle",
        "description": "multiplicative unit-mean gamma speckle for integer looks",
        "aliases": ("multilook_speckle",),
    },
)


def add_gamma_speckle(
    image: torch.Tensor,
    *,
    looks: Numeric = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Multiply by a unit-mean gamma field generated from integer looks."""
    image = check_image(image)
    looks_int = int(torch.as_tensor(looks).item())
    if looks_int < 1:
        raise ValueError("looks must be a positive integer")
    gen = resolve_generator(image, seed=seed, generator=generator)
    uniforms = rand_shape((looks_int, *image.shape), image, gen).clamp_min(1e-30)
    multiplier = -torch.log(uniforms).sum(dim=0) / float(looks_int)
    return apply_clamp(image * multiplier, clamp)
