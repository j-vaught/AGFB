"""Batched, GPU-accelerated filters for the Analytical Gradient Filter Benchmark."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from agfb_filters.filters.catalog import root_export_modules

_EXPORT_MODULES = root_export_modules()

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
