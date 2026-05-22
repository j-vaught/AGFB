"""Vectorized noise models for floating-point image tensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

Numeric = float | int | torch.Tensor
ClampRange = tuple[float | None, float | None] | None
NoiseKind = Literal["none", "gaussian", "uniform", "salt_pepper", "poisson", "speckle", "rician"]


@dataclass(frozen=True)
class NoiseSpec:
    """Named noise model with keyword parameters."""

    kind: NoiseKind
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def add_noise(
    image: torch.Tensor,
    spec: NoiseKind | NoiseSpec,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch a named noise model."""
    if isinstance(spec, NoiseSpec):
        params = dict(spec.kwargs)
        params.update(kwargs)
        kind = spec.kind
    else:
        params = dict(kwargs)
        kind = spec

    if kind == "none":
        return _apply_clamp(_check_image(image), clamp)
    if kind == "gaussian":
        return add_gaussian(
            image,
            seed=seed,
            generator=generator,
            clamp=clamp,
            **params,
        )
    if kind == "uniform":
        return add_uniform(
            image,
            seed=seed,
            generator=generator,
            clamp=clamp,
            **params,
        )
    if kind == "salt_pepper":
        return add_salt_pepper(
            image,
            seed=seed,
            generator=generator,
            clamp=clamp,
            **params,
        )
    if kind == "poisson":
        return add_poisson(
            image,
            seed=seed,
            generator=generator,
            clamp=clamp,
            **params,
        )
    if kind == "speckle":
        return add_speckle(
            image,
            seed=seed,
            generator=generator,
            clamp=clamp,
            **params,
        )
    if kind == "rician":
        return add_rician(
            image,
            seed=seed,
            generator=generator,
            clamp=clamp,
            **params,
        )
    raise ValueError(f"unknown noise kind {kind!r}")


def add_gaussian(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add independent normal noise."""
    image = _check_image(image)
    gen = _resolve_generator(image, seed=seed, generator=generator)
    sigma_t = _batch_param(sigma, image, name="sigma")
    mean_t = _batch_param(mean, image, name="mean")
    noise = _randn(image, gen) * sigma_t + mean_t
    return _apply_clamp(image + noise, clamp)


def add_uniform(
    image: torch.Tensor,
    *,
    half_width: Numeric,
    mean: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add independent uniform noise on `[mean - half_width, mean + half_width]`."""
    image = _check_image(image)
    gen = _resolve_generator(image, seed=seed, generator=generator)
    half_width_t = _batch_param(half_width, image, name="half_width")
    mean_t = _batch_param(mean, image, name="mean")
    noise = (2.0 * _rand(image, gen) - 1.0) * half_width_t + mean_t
    return _apply_clamp(image + noise, clamp)


def add_salt_pepper(
    image: torch.Tensor,
    *,
    amount: Numeric,
    salt_vs_pepper: Numeric = 0.5,
    salt_value: Numeric = 1.0,
    pepper_value: Numeric = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Replace random pixels with salt and pepper values."""
    image = _check_image(image)
    gen = _resolve_generator(image, seed=seed, generator=generator)
    amount_t = _batch_param(amount, image, name="amount")
    salt_vs_pepper_t = _batch_param(salt_vs_pepper, image, name="salt_vs_pepper")
    salt_value_t = _batch_param(salt_value, image, name="salt_value")
    pepper_value_t = _batch_param(pepper_value, image, name="pepper_value")
    _validate_probability(amount_t, "amount")
    _validate_probability(salt_vs_pepper_t, "salt_vs_pepper")

    draws = _rand(image, gen)
    salt_threshold = amount_t * salt_vs_pepper_t
    pepper_threshold = amount_t
    salted = draws < salt_threshold
    peppered = (draws >= salt_threshold) & (draws < pepper_threshold)
    out = torch.where(salted, salt_value_t.expand_as(image), image)
    out = torch.where(peppered, pepper_value_t.expand_as(image), out)
    return _apply_clamp(out, clamp)


def add_poisson(
    image: torch.Tensor,
    *,
    peak: Numeric = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply Poisson shot noise to nonnegative intensity."""
    image = _check_image(image)
    gen = _resolve_generator(image, seed=seed, generator=generator)
    peak_t = _batch_param(peak, image, name="peak")
    _validate_positive(peak_t, "peak")
    lam = image.clamp_min(0.0) * peak_t
    counts = torch.poisson(lam) if gen is None else torch.poisson(lam, generator=gen)
    return _apply_clamp(counts / peak_t, clamp)


def add_speckle(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Add multiplicative normal noise."""
    image = _check_image(image)
    gen = _resolve_generator(image, seed=seed, generator=generator)
    sigma_t = _batch_param(sigma, image, name="sigma")
    noise = _randn(image, gen) * sigma_t
    return _apply_clamp(image + image * noise, clamp)


def add_rician(
    image: torch.Tensor,
    *,
    sigma: Numeric,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply Rician magnitude noise."""
    image = _check_image(image)
    gen = _resolve_generator(image, seed=seed, generator=generator)
    sigma_t = _batch_param(sigma, image, name="sigma")
    n1 = _randn(image, gen) * sigma_t
    n2 = _randn(image, gen) * sigma_t
    out = torch.sqrt((image + n1).square() + n2.square())
    return _apply_clamp(out, clamp)


def _check_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if not image.is_floating_point():
        raise TypeError(f"image must be floating point; got {image.dtype}")
    if image.ndim == 0:
        raise ValueError("image must have at least one dimension")
    return image


def _batch_param(value: Numeric, image: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=image.dtype, device=image.device)
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional tensor")
    if image.ndim < 2:
        raise ValueError(f"{name} can be one-dimensional only for batched image tensors")
    if tensor.shape[0] != image.shape[0]:
        raise ValueError(
            f"{name} length {tensor.shape[0]} must match image batch size {image.shape[0]}"
        )
    return tensor.view(tensor.shape[0], *([1] * (image.ndim - 1)))


def _resolve_generator(
    image: torch.Tensor,
    *,
    seed: int | None,
    generator: torch.Generator | None,
) -> torch.Generator | None:
    if seed is not None and generator is not None:
        raise ValueError("seed and generator cannot both be provided")
    if seed is None:
        return generator
    try:
        gen = torch.Generator(device=image.device)
    except RuntimeError:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _rand(image: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "dtype": image.dtype,
        "device": image.device,
    }
    if generator is not None:
        kwargs["generator"] = generator
    return torch.rand(image.shape, **kwargs)


def _randn(image: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "dtype": image.dtype,
        "device": image.device,
    }
    if generator is not None:
        kwargs["generator"] = generator
    return torch.randn(image.shape, **kwargs)


def _apply_clamp(image: torch.Tensor, clamp: ClampRange) -> torch.Tensor:
    if clamp is None:
        return image
    low, high = clamp
    return torch.clamp(image, min=low, max=high)


def _validate_positive(value: torch.Tensor, name: str) -> None:
    if bool((value <= 0).any().item()):
        raise ValueError(f"{name} must be positive")


def _validate_probability(value: torch.Tensor, name: str) -> None:
    if bool(((value < 0) | (value > 1)).any().item()):
        raise ValueError(f"{name} must be in [0, 1]")
