"""Dead- and hot-pixel defect noise."""

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
        "name": "dead_pixel",
        "function": "add_dead_pixels",
        "description": "random dead and hot pixel defects",
        "aliases": ("dead_pixels", "defect_pixel"),
    },
)


def add_dead_pixels(
    image: torch.Tensor,
    *,
    amount: Numeric,
    hot_fraction: Numeric = 0.0,
    dead_value: Numeric = 0.0,
    hot_value: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random pixels with dead or hot values."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    amount_t = batch_param(amount, image, name="amount")
    hot_fraction_t = batch_param(hot_fraction, image, name="hot_fraction")
    dead_value_t = batch_param(dead_value, image, name="dead_value")
    hot_value_t = batch_param(hot_value, image, name="hot_value")
    validate_probability(amount_t, "amount")
    validate_probability(hot_fraction_t, "hot_fraction")
    defect = rand_like(image, gen) < amount_t
    hot = rand_like(image, gen) < hot_fraction_t
    replacement = torch.where(hot, hot_value_t.expand_as(image), dead_value_t.expand_as(image))
    return apply_clamp(torch.where(defect, replacement, image), clamp)
