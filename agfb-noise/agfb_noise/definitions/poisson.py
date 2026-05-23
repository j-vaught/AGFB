"""Poisson shot noise."""

from __future__ import annotations

import torch

from agfb_noise.helpers.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    resolve_generator,
    validate_positive,
)

NOISE_SPECS = (
    {
        "name": "poisson",
        "function": "add_poisson",
        "description": "Poisson-distributed shot noise generated from nonnegative intensity",
        "aliases": ("shot", "shot_noise"),
    },
)


def add_poisson(
    image: torch.Tensor,
    *,
    peak: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply Poisson shot noise using `peak` counts per unit intensity."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    peak_t = batch_param(peak, image, name="peak")
    validate_positive(peak_t, "peak")
    lam = image.clamp_min(0.0) * peak_t
    counts = torch.poisson(lam) if gen is None else torch.poisson(lam, generator=gen)
    return apply_clamp(counts / peak_t, clamp)
