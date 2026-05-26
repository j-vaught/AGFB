"""Name-based noise execution helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from agfb_noise.helpers.base import ClampRange, apply_clamp
from agfb_noise.helpers.registry import get_noise_registration


@dataclass(frozen=True)
class NoiseSpec:
    """Named noise model with keyword parameters."""

    name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def add_noise(
    image: torch.Tensor,
    spec: str | NoiseSpec,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    clamp: ClampRange = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Apply one named noise model."""
    if isinstance(spec, NoiseSpec):
        name = spec.name
        params = dict(spec.kwargs)
        params.update(kwargs)
    else:
        name = str(spec)
        params = dict(kwargs)
    if name == "none":
        return apply_clamp(image, clamp)
    registration = get_noise_registration(name)
    return registration.apply(
        image,
        seed=seed,
        generator=generator,
        clamp=clamp,
        **params,
    )


def apply_noise_sequence(
    image: torch.Tensor,
    specs: Sequence[str | NoiseSpec],
    *,
    seed: int | None = None,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Apply a deterministic sequence of named noise models."""
    out = image
    for index, spec in enumerate(specs):
        step_seed = None if seed is None else int(seed) + index
        out = add_noise(out, spec, seed=step_seed, clamp=clamp)
    return out
