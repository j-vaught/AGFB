"""Shared tensor helpers for batched AGFB noise models."""

from __future__ import annotations

from typing import Any

import torch

Numeric = float | int | torch.Tensor
ClampRange = tuple[float | None, float | None] | None


def check_image(image: torch.Tensor) -> torch.Tensor:
    """Validate a floating-point image tensor."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if not image.is_floating_point():
        raise TypeError(f"image must be floating point; got {image.dtype}")
    if image.ndim == 0:
        raise ValueError("image must have at least one dimension")
    return image


def batch_param(value: Numeric, image: torch.Tensor, *, name: str) -> torch.Tensor:
    """Return a scalar or batch-broadcast tensor for one model parameter."""
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


def image_param(value: Numeric, image: torch.Tensor, *, name: str) -> torch.Tensor:
    """Return a scalar, batch, or per-pixel image parameter tensor."""
    tensor = torch.as_tensor(value, dtype=image.dtype, device=image.device)
    if tensor.shape == image.shape:
        return tensor
    return batch_param(tensor, image, name=name)


def resolve_generator(
    image: torch.Tensor,
    *,
    seed: int | None,
    generator: torch.Generator | None,
) -> torch.Generator | None:
    """Resolve local random state without mutating the global torch seed."""
    if seed is not None and generator is not None:
        raise ValueError("seed and generator cannot both be provided")
    if seed is None:
        return generator
    try:
        resolved = torch.Generator(device=image.device)
    except RuntimeError:
        resolved = torch.Generator()
    resolved.manual_seed(int(seed))
    return resolved


def rand_like(image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return uniform random values with image shape, dtype, and device."""
    kwargs: dict[str, Any] = {"dtype": image.dtype, "device": image.device}
    if generator is not None:
        kwargs["generator"] = generator
    return torch.rand(image.shape, **kwargs)


def randn_like(image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Return normal random values with image shape, dtype, and device."""
    kwargs: dict[str, Any] = {"dtype": image.dtype, "device": image.device}
    if generator is not None:
        kwargs["generator"] = generator
    return torch.randn(image.shape, **kwargs)


def rand_shape(
    shape: tuple[int, ...] | torch.Size,
    image: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return uniform random values using image dtype and device for an arbitrary shape."""
    kwargs: dict[str, Any] = {"dtype": image.dtype, "device": image.device}
    if generator is not None:
        kwargs["generator"] = generator
    return torch.rand(tuple(shape), **kwargs)


def randn_shape(
    shape: tuple[int, ...] | torch.Size,
    image: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return normal random values using image dtype and device for an arbitrary shape."""
    kwargs: dict[str, Any] = {"dtype": image.dtype, "device": image.device}
    if generator is not None:
        kwargs["generator"] = generator
    return torch.randn(tuple(shape), **kwargs)


def apply_clamp(image: torch.Tensor, clamp: ClampRange) -> torch.Tensor:
    """Clamp only when an explicit output interval is supplied."""
    if clamp is None:
        return image
    low, high = clamp
    return torch.clamp(image, min=low, max=high)


def validate_positive(value: torch.Tensor, name: str) -> None:
    """Validate a positive scalar or tensor parameter."""
    if bool((value <= 0).any().item()):
        raise ValueError(f"{name} must be positive")


def validate_nonnegative(value: torch.Tensor, name: str) -> None:
    """Validate a nonnegative scalar or tensor parameter."""
    if bool((value < 0).any().item()):
        raise ValueError(f"{name} must be nonnegative")


def validate_probability(value: torch.Tensor, name: str) -> None:
    """Validate a probability scalar or tensor parameter."""
    if bool(((value < 0) | (value > 1)).any().item()):
        raise ValueError(f"{name} must be in [0, 1]")
