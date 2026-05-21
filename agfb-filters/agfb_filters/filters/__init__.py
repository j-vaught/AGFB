"""Filter definitions and kernel design helpers."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

from agfb_filters.filters.catalog import filter_export_modules

_EXPORT_MODULES = filter_export_modules()

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _load_export(name)


def _load_export(name: str) -> Any:
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


class _FilterPackageModule(ModuleType):
    def __getattribute__(self, name: str) -> Any:
        export_modules = ModuleType.__getattribute__(self, "_EXPORT_MODULES")
        if name in export_modules:
            namespace = ModuleType.__getattribute__(self, "__dict__")
            current_value = namespace.get(name)
            if current_value is not None and not isinstance(current_value, ModuleType):
                return current_value
            load_export = ModuleType.__getattribute__(self, "_load_export")
            return load_export(name)
        return ModuleType.__getattribute__(self, name)


sys.modules[__name__].__class__ = _FilterPackageModule
