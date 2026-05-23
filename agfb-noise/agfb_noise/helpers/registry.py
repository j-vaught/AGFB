"""Noise-model registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

from agfb_noise.base import ClampRange
from agfb_noise.catalog import shipped_noise_specs

NoiseFunction = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class NoiseRegistration:
    """Registered callable for one AGFB noise model."""

    name: str
    function: NoiseFunction
    description: str = ""
    default_kwargs: Mapping[str, Any] | None = None

    def apply(
        self,
        image: torch.Tensor,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        clamp: ClampRange = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        params = {} if self.default_kwargs is None else dict(self.default_kwargs)
        params.update(kwargs)
        return self.function(
            image,
            seed=seed,
            generator=generator,
            clamp=clamp,
            **params,
        )


_REGISTRY: dict[str, NoiseRegistration] = {}
_BUILTINS_REGISTERED = False


def register_noise(
    name: str,
    function: NoiseFunction,
    *,
    description: str = "",
    default_kwargs: Mapping[str, Any] | None = None,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> NoiseRegistration:
    """Register a noise model callable."""
    _ensure_builtin_noises()
    registration = _store_noise(
        name,
        function,
        description=description,
        default_kwargs=default_kwargs,
        replace=replace,
    )
    for alias in aliases:
        _store_noise(
            alias,
            function,
            description=description,
            default_kwargs=default_kwargs,
            replace=replace,
        )
    return registration


def get_noise_registration(name: str) -> NoiseRegistration:
    """Return a registered noise model by name or alias."""
    _ensure_builtin_noises()
    normalized = _normalize_name(name)
    try:
        return _REGISTRY[normalized]
    except KeyError as error:
        choices = ", ".join(registered_noises())
        raise KeyError(
            f"unknown noise model {normalized!r}; available models are {choices}"
        ) from error


def registered_noises() -> tuple[str, ...]:
    """Return registered noise names and aliases."""
    _ensure_builtin_noises()
    return tuple(sorted(_REGISTRY))


def _store_noise(
    name: str,
    function: NoiseFunction,
    *,
    description: str,
    default_kwargs: Mapping[str, Any] | None,
    replace: bool,
) -> NoiseRegistration:
    normalized = _normalize_name(name)
    if not callable(function):
        raise TypeError("function must be callable")
    if normalized in _REGISTRY and not replace:
        raise ValueError(f"noise model {normalized!r} is already registered")
    registration = NoiseRegistration(
        name=normalized,
        function=function,
        description=description,
        default_kwargs=default_kwargs,
    )
    _REGISTRY[normalized] = registration
    return registration


def _normalize_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("noise name must not be empty")
    return normalized


def _ensure_builtin_noises() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for spec in shipped_noise_specs():
        module = import_module(spec.module)
        function = getattr(module, spec.function)
        registration = _store_noise(
            spec.name,
            function,
            description=spec.description,
            default_kwargs=dict(spec.default_kwargs),
            replace=False,
        )
        for alias in spec.aliases:
            _store_noise(
                alias,
                registration.function,
                description=spec.description,
                default_kwargs=dict(spec.default_kwargs),
                replace=False,
            )
    _BUILTINS_REGISTERED = True
