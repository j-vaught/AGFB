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
    "agfb_noise.gaussian",
    "agfb_noise.local_variance",
    "agfb_noise.uniform",
    "agfb_noise.poisson",
    "agfb_noise.poisson_gaussian",
    "agfb_noise.dark_current",
    "agfb_noise.salt",
    "agfb_noise.pepper",
    "agfb_noise.salt_pepper",
    "agfb_noise.random_impulse",
    "agfb_noise.dead_pixel",
    "agfb_noise.speckle",
    "agfb_noise.gamma_speckle",
    "agfb_noise.rician",
    "agfb_noise.rayleigh",
    "agfb_noise.quantization",
    "agfb_noise.fixed_pattern",
    "agfb_noise.stripe",
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
