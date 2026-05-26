"""Filter registry for built-in and user-defined gradient filters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any

from agfb_filters.filters.catalog import shipped_filter_specs
from agfb_filters.filters.definitions import GradientFilterDefinition

DefinitionFactory = Callable[..., GradientFilterDefinition]


@dataclass(frozen=True)
class FilterRegistration:
    """Registered factory for a gradient filter definition."""

    name: str
    definition_factory: DefinitionFactory
    description: str = ""

    def build(self, *args: Any, **kwargs: Any) -> GradientFilterDefinition:
        definition = self.definition_factory(*args, **kwargs)
        if not isinstance(definition, GradientFilterDefinition):
            raise TypeError(
                f"filter factory for {self.name!r} returned {type(definition).__name__}, "
                "expected GradientFilterDefinition"
            )
        return definition


_REGISTRY: dict[str, FilterRegistration] = {}
_BUILTINS_REGISTERED = False


def register_filter(
    name: str,
    definition_factory: DefinitionFactory,
    *,
    description: str = "",
    replace: bool = False,
) -> FilterRegistration:
    """Register a reusable filter definition factory."""
    _ensure_builtin_filters()
    return _store_filter(
        name,
        definition_factory,
        description=description,
        replace=replace,
    )


def get_filter_registration(name: str) -> FilterRegistration:
    """Return a registered filter factory by name."""
    _ensure_builtin_filters()
    normalized_name = _normalize_name(name)
    try:
        return _REGISTRY[normalized_name]
    except KeyError as error:
        raise KeyError(f"unknown filter {normalized_name!r}") from error


def get_filter_definition(
    name: str,
    *args: Any,
    **kwargs: Any,
) -> GradientFilterDefinition:
    """Build a filter definition from a registered factory."""
    return get_filter_registration(name).build(*args, **kwargs)


def registered_filters() -> tuple[str, ...]:
    """Return registered filter names."""
    _ensure_builtin_filters()
    return tuple(sorted(_REGISTRY))


def _store_filter(
    name: str,
    definition_factory: DefinitionFactory,
    *,
    description: str = "",
    replace: bool = False,
) -> FilterRegistration:
    normalized_name = _normalize_name(name)
    if not callable(definition_factory):
        raise TypeError("definition_factory must be callable")
    if normalized_name in _REGISTRY and not replace:
        raise ValueError(f"filter {normalized_name!r} is already registered")
    registration = FilterRegistration(
        name=normalized_name,
        definition_factory=definition_factory,
        description=description,
    )
    _REGISTRY[normalized_name] = registration
    return registration


def _normalize_name(name: str) -> str:
    normalized_name = str(name).strip()
    if not normalized_name:
        raise ValueError("filter name must not be empty")
    return normalized_name


def _ensure_builtin_filters() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    for spec in shipped_filter_specs():
        module = import_module(spec.module)
        factory = getattr(module, spec.definition_factory)
        if spec.registry_kwargs:
            factory = partial(factory, **dict(spec.registry_kwargs))
        _store_filter(spec.name, factory, description=spec.description)
    _BUILTINS_REGISTERED = True
