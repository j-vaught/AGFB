"""Runtime execution and boundary handling."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "BoundaryCondition": "agfb_filters.runtime.execution",
    "BoundaryMode": "agfb_filters.runtime.execution",
    "CollapsedOrientationBank": "agfb_filters.runtime.runner",
    "ExecutionPath": "agfb_filters.runtime.execution",
    "OrientationBankResult": "agfb_filters.runtime.runner",
    "collapse_orientation_bank": "agfb_filters.runtime.runner",
    "run_filter": "agfb_filters.runtime.runner",
    "run_orientation_bank": "agfb_filters.runtime.runner",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
