"""Rician magnitude noise."""

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
        "name": "rician",
        "function": "add_rician",
        "description": "Rician magnitude noise from two independent Gaussian channels",
        "aliases": ("rice",),
    },
)


def add_rician(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Return `sqrt((image + n1)^2 + n2^2)` with Gaussian channel noise."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    validate_nonnegative(sigma_t, "sigma")
    n1 = randn_like(image, gen) * sigma_t
    n2 = randn_like(image, gen) * sigma_t
    return apply_clamp(torch.sqrt((image + n1).square() + n2.square()), clamp)
