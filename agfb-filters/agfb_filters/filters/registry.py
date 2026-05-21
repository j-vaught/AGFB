"""Filter registry for built-in and user-defined gradient filters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

    from agfb_filters.filters.central_difference import central_difference_definition
    from agfb_filters.filters.cpgf import cpgf_definition
    from agfb_filters.filters.derivative_of_gaussian import derivative_of_gaussian_definition
    from agfb_filters.filters.farid_simoncelli import farid_simoncelli_5_definition
    from agfb_filters.filters.freeman_adelson import freeman_adelson_g1_definition
    from agfb_filters.filters.prewitt import prewitt_3_definition
    from agfb_filters.filters.roberts import roberts_definition
    from agfb_filters.filters.savitzky_golay import savitzky_golay_definition
    from agfb_filters.filters.scharr import scharr_3_definition
    from agfb_filters.filters.sobel import sobel_definition

    _BUILTINS_REGISTERED = True
    for name, factory, description in (
        ("central_difference", central_difference_definition, "central finite difference"),
        ("farid_simoncelli_5", farid_simoncelli_5_definition, "Farid-Simoncelli 5-tap"),
        ("prewitt_3", prewitt_3_definition, "Prewitt 3-tap"),
        ("roberts", roberts_definition, "Roberts cross"),
        ("scharr_3", scharr_3_definition, "Scharr 3-tap"),
        ("sobel_3", lambda: sobel_definition(3), "Sobel 3-tap"),
        ("sobel_5", lambda: sobel_definition(5), "Sobel 5-tap"),
        ("sobel_7", lambda: sobel_definition(7), "Sobel 7-tap"),
        ("cpgf", cpgf_definition, "circular polynomial gradient filter"),
        (
            "derivative_of_gaussian",
            derivative_of_gaussian_definition,
            "first derivative of Gaussian",
        ),
        ("freeman_adelson_g1", freeman_adelson_g1_definition, "Freeman-Adelson G1"),
        ("savitzky_golay", savitzky_golay_definition, "Savitzky-Golay square fit"),
    ):
        _store_filter(name, factory, description=description)
