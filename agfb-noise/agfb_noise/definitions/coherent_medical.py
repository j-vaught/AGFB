"""Coherent-imaging and medical magnitude noise models."""

from __future__ import annotations

import torch

from agfb_noise.helpers.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    randn_like,
    randn_shape,
    resolve_generator,
    validate_nonnegative,
    validate_positive,
)

NOISE_SPECS = (
    {
        "name": "nakagami_speckle",
        "function": "add_nakagami_speckle",
        "description": "multiplicative Nakagami-distributed amplitude speckle",
    },
    {
        "name": "k_speckle",
        "function": "add_k_speckle",
        "description": "multiplicative K-distributed coherent-imaging speckle",
    },
    {
        "name": "oct_speckle",
        "function": "add_oct_speckle",
        "description": "multiplicative OCT-style log-amplitude speckle",
    },
    {
        "name": "sar_multilook",
        "function": "add_sar_multilook",
        "description": "unit-mean gamma SAR multilook speckle controlled by ENL",
        "aliases": ("enl_speckle",),
    },
    {
        "name": "noncentral_chi",
        "function": "add_noncentral_chi",
        "description": "noncentral chi magnitude noise",
    },
    {
        "name": "log_speckle",
        "function": "add_log_speckle",
        "description": "homomorphic log-domain multiplicative speckle",
    },
    {
        "name": "lognormal_scintillation",
        "function": "add_lognormal_scintillation",
        "description": "unit-mean log-normal multiplicative scintillation",
    },
)


def add_nakagami_speckle(
    image: torch.Tensor,
    *,
    m: Numeric = 1.0,
    omega: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Multiply by Nakagami amplitude noise with shape `m` and spread `omega`."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    m_t = batch_param(m, image, name="m")
    omega_t = batch_param(omega, image, name="omega")
    if bool((m_t < 0.5).any().item()):
        raise ValueError("m must be at least 0.5")
    validate_positive(omega_t, "omega")
    gamma = _standard_gamma(m_t.expand_as(image), gen)
    multiplier = torch.sqrt(gamma * omega_t / m_t)
    return apply_clamp(image * multiplier, clamp)


def add_k_speckle(
    image: torch.Tensor,
    *,
    texture_shape: Numeric = 1.0,
    looks: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Multiply by unit-mean K-style speckle from gamma texture and speckle terms."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    texture_shape_t = batch_param(texture_shape, image, name="texture_shape")
    looks_t = batch_param(looks, image, name="looks")
    validate_positive(texture_shape_t, "texture_shape")
    validate_positive(looks_t, "looks")
    texture = _standard_gamma(texture_shape_t.expand_as(image), gen) / texture_shape_t
    speckle = _standard_gamma(looks_t.expand_as(image), gen) / looks_t
    return apply_clamp(image * texture * speckle, clamp)


def add_oct_speckle(
    image: torch.Tensor,
    *,
    sigma: Numeric = 0.25,
    contrast: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply OCT-style multiplicative log-amplitude speckle."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    contrast_t = batch_param(contrast, image, name="contrast")
    validate_nonnegative(sigma_t, "sigma")
    validate_nonnegative(contrast_t, "contrast")
    log_multiplier = randn_like(image, gen) * sigma_t * contrast_t
    multiplier = torch.exp(log_multiplier - 0.5 * (sigma_t * contrast_t).square())
    return apply_clamp(image * multiplier, clamp)


def add_sar_multilook(
    image: torch.Tensor,
    *,
    enl: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Multiply by unit-mean gamma speckle with equivalent number of looks `enl`."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    enl_t = batch_param(enl, image, name="enl")
    validate_positive(enl_t, "enl")
    multiplier = _standard_gamma(enl_t.expand_as(image), gen) / enl_t
    return apply_clamp(image * multiplier, clamp)


def add_noncentral_chi(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    channels: int = 2,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply noncentral chi magnitude noise with `channels` Gaussian components."""
    image = check_image(image)
    if channels < 1:
        raise ValueError("channels must be positive")
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    validate_positive(sigma_t, "sigma")
    components = randn_shape((channels, *image.shape), image, gen) * sigma_t
    components[0] = components[0] + image
    magnitude = torch.sqrt(components.square().sum(dim=0))
    return apply_clamp(magnitude, clamp)


def add_log_speckle(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    mean: Numeric = 0.0,
    eps: Numeric = 1e-12,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add normal noise in log space and transform back to the image domain."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    mean_t = batch_param(mean, image, name="mean")
    eps_t = torch.as_tensor(eps, dtype=image.dtype, device=image.device)
    validate_nonnegative(sigma_t, "sigma")
    validate_positive(eps_t, "eps")
    log_image = torch.log(image.clamp_min(eps_t))
    out = torch.exp(log_image + randn_like(image, gen) * sigma_t + mean_t)
    return apply_clamp(out, clamp)


def add_lognormal_scintillation(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Multiply by unit-mean log-normal scintillation noise."""
    image = check_image(image)
    gen = resolve_generator(image, seed=seed, generator=generator)
    sigma_t = batch_param(sigma, image, name="sigma")
    validate_nonnegative(sigma_t, "sigma")
    multiplier = torch.exp(randn_like(image, gen) * sigma_t - 0.5 * sigma_t.square())
    return apply_clamp(image * multiplier, clamp)


def _standard_gamma(shape: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return (
        torch._standard_gamma(shape)
        if generator is None
        else torch._standard_gamma(shape, generator=generator)
    ).clamp_min(torch.finfo(shape.dtype).tiny)
