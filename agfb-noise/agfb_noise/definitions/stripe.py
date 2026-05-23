"""Row and column stripe noise."""

from __future__ import annotations

import torch

from agfb_noise.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    randn_shape,
    resolve_generator,
    validate_nonnegative,
)

NOISE_SPECS = (
    {
        "name": "stripe",
        "function": "add_stripe",
        "description": "row and column correlated offset noise",
        "aliases": ("banding", "row_column"),
    },
)


def add_stripe(
    image: torch.Tensor,
    *,
    row_sigma: Numeric = 0.0,
    column_sigma: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add independent row and column offsets along the last two axes."""
    image = check_image(image)
    if image.ndim < 2:
        raise ValueError("stripe noise requires at least two image dimensions")
    gen = resolve_generator(image, seed=seed, generator=generator)
    row_sigma_t = batch_param(row_sigma, image, name="row_sigma")
    column_sigma_t = batch_param(column_sigma, image, name="column_sigma")
    validate_nonnegative(row_sigma_t, "row_sigma")
    validate_nonnegative(column_sigma_t, "column_sigma")
    row_shape = (*image.shape[:-1], 1)
    column_shape = (*image.shape[:-2], 1, image.shape[-1])
    row_offsets = randn_shape(row_shape, image, gen) * row_sigma_t
    column_offsets = randn_shape(column_shape, image, gen) * column_sigma_t
    return apply_clamp(image + row_offsets + column_offsets, clamp)
