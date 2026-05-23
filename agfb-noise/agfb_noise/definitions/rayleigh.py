"""Rayleigh-distributed positive noise."""

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
        "name": "rayleigh",
        "function": "add_rayleigh",
        "description": "additive Rayleigh-distributed positive noise",
    },
)


def add_rayleigh(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add Rayleigh noise sampled by inverse transform."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    validate_nonnegative(sigma_t, "sigma")
    u = rand_like(image, gen).clamp_min(1e-30).clamp_max(1.0 - 1e-7)
    rayleigh = sigma_t * torch.sqrt(-2.0 * torch.log1p(-u))
    return apply_clamp(image + rayleigh, clamp)
