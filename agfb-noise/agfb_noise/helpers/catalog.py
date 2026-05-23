"""Catalog collection for shipped AGFB noise models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class BuiltInNoiseSpec:
    """Metadata needed to expose and register one shipped noise model."""

    name: str
    module: str
    function: str
    description: str
    aliases: tuple[str, ...] = ()
    default_kwargs: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))


_NOISE_MODULES = (
    "agfb_noise.definitions.gaussian",
    "agfb_noise.definitions.local_variance",
    "agfb_noise.definitions.uniform",
    "agfb_noise.definitions.poisson",
    "agfb_noise.definitions.poisson_gaussian",
    "agfb_noise.definitions.dark_current",
    "agfb_noise.definitions.salt",
    "agfb_noise.definitions.pepper",
    "agfb_noise.definitions.salt_pepper",
    "agfb_noise.definitions.random_impulse",
    "agfb_noise.definitions.dead_pixel",
    "agfb_noise.definitions.speckle",
    "agfb_noise.definitions.gamma_speckle",
    "agfb_noise.definitions.rician",
    "agfb_noise.definitions.rayleigh",
    "agfb_noise.definitions.quantization",
    "agfb_noise.definitions.fixed_pattern",
    "agfb_noise.definitions.stripe",
)


def shipped_noise_specs() -> tuple[BuiltInNoiseSpec, ...]:
    """Return metadata for noise models shipped with the package."""
    specs: list[BuiltInNoiseSpec] = []
    for module_name in _NOISE_MODULES:
        module = import_module(module_name)
        for spec_data in getattr(module, "NOISE_SPECS", ()):
            specs.append(_built_in_noise_spec(module_name, spec_data))
    return tuple(specs)


def _built_in_noise_spec(module_name: str, data: Mapping[str, Any]) -> BuiltInNoiseSpec:
    return BuiltInNoiseSpec(
        name=str(data["name"]),
        module=module_name,
        function=str(data["function"]),
        description=str(data["description"]),
        aliases=tuple(str(alias) for alias in data.get("aliases", ())),
        default_kwargs=MappingProxyType(dict(data.get("default_kwargs", {}))),
    )
