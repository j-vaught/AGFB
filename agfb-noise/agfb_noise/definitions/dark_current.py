"""Dark-current shot noise and read noise."""

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
        "name": "dark_current",
        "function": "add_dark_current",
        "description": "Poisson dark-current background plus optional Gaussian read noise",
    },
)


def add_dark_current(
    image: torch.Tensor,
    *,
    dark_rate: Numeric,
    exposure_time: Numeric = 1.0,
    peak: Numeric = 1.0,
    read_sigma: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add dark current in count units before scaling back to image intensity."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    dark_rate_t = batch_param(dark_rate, image, name="dark_rate")
    exposure_time_t = batch_param(exposure_time, image, name="exposure_time")
    peak_t = batch_param(peak, image, name="peak")
    read_sigma_t = batch_param(read_sigma, image, name="read_sigma")
    validate_nonnegative(dark_rate_t, "dark_rate")
    validate_nonnegative(exposure_time_t, "exposure_time")
    validate_positive(peak_t, "peak")
    validate_nonnegative(read_sigma_t, "read_sigma")
    signal_counts = image.clamp_min(0.0) * peak_t
    dark_counts = dark_rate_t * exposure_time_t
    lam = signal_counts + dark_counts
    counts = torch.poisson(lam) if gen is None else torch.poisson(lam, generator=gen)
    read_noise = randn_like(image, gen) * read_sigma_t
    return apply_clamp(counts / peak_t + read_noise, clamp)
