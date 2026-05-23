"""Fixed-pattern offset and gain noise."""

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
        "name": "fixed_pattern",
        "function": "add_fixed_pattern",
        "description": "pixelwise offset and gain nonuniformity",
        "aliases": ("fpn",),
    },
)


def add_fixed_pattern(
    image: torch.Tensor,
    *,
    offset_sigma: Numeric = 0.0,
    gain_sigma: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply pixelwise dark-signal and photo-response nonuniformity."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    offset_sigma_t = batch_param(offset_sigma, image, name="offset_sigma")
    gain_sigma_t = batch_param(gain_sigma, image, name="gain_sigma")
    validate_nonnegative(offset_sigma_t, "offset_sigma")
    validate_nonnegative(gain_sigma_t, "gain_sigma")
    offset = randn_like(image, gen) * offset_sigma_t
    gain = randn_like(image, gen) * gain_sigma_t
    return apply_clamp(image * (1.0 + gain) + offset, clamp)
