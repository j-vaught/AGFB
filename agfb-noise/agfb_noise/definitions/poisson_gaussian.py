"""Signal-dependent Poisson-Gaussian sensor noise."""

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
    validate_positive,
)

NOISE_SPECS = (
    {
        "name": "poisson_gaussian",
        "function": "add_poisson_gaussian",
        "description": "Poisson shot noise plus signal-independent Gaussian read noise",
        "aliases": ("poisson-gaussian", "pg"),
    },
)


def add_poisson_gaussian(
    image: torch.Tensor,
    *,
    peak: Numeric = 1.0,
    read_sigma: Numeric = 0.0,
    read_mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply Poisson shot noise and additive Gaussian read noise."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    peak_t = batch_param(peak, image, name="peak")
    read_sigma_t = batch_param(read_sigma, image, name="read_sigma")
    read_mean_t = batch_param(read_mean, image, name="read_mean")
    validate_positive(peak_t, "peak")
    validate_nonnegative(read_sigma_t, "read_sigma")
    lam = image.clamp_min(0.0) * peak_t
    counts = torch.poisson(lam) if gen is None else torch.poisson(lam, generator=gen)
    read_noise = randn_like(image, gen) * read_sigma_t + read_mean_t
    return apply_clamp(counts / peak_t + read_noise, clamp)
