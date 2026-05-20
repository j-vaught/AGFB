"""Filter definitions and kernel design helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "CPGF": "agfb_filters.filters.cpgf",
    "DerivativeOfGaussian": "agfb_filters.filters.derivative_of_gaussian",
    "FreemanAdelsonG1": "agfb_filters.filters.freeman_adelson",
    "GradientFilterDefinition": "agfb_filters.filters.definitions",
    "SavitzkyGolay": "agfb_filters.filters.savitzky_golay",
    "central_difference": "agfb_filters.filters.central_difference",
    "central_difference_definition": "agfb_filters.filters.central_difference",
    "cpgf_definition": "agfb_filters.filters.cpgf",
    "cpgf_kernels": "agfb_filters.filters.cpgf",
    "derivative_of_gaussian_definition": "agfb_filters.filters.derivative_of_gaussian",
    "farid_simoncelli_5": "agfb_filters.filters.farid_simoncelli",
    "farid_simoncelli_5_definition": "agfb_filters.filters.farid_simoncelli",
    "freeman_adelson_g1_definition": "agfb_filters.filters.freeman_adelson",
    "prewitt_3": "agfb_filters.filters.prewitt",
    "prewitt_3_definition": "agfb_filters.filters.prewitt",
    "roberts": "agfb_filters.filters.roberts",
    "roberts_definition": "agfb_filters.filters.roberts",
    "savitzky_golay_definition": "agfb_filters.filters.savitzky_golay",
    "savitzky_golay_kernels": "agfb_filters.filters.savitzky_golay",
    "scharr_3": "agfb_filters.filters.scharr",
    "scharr_3_definition": "agfb_filters.filters.scharr",
    "sobel_3": "agfb_filters.filters.sobel",
    "sobel_5": "agfb_filters.filters.sobel",
    "sobel_7": "agfb_filters.filters.sobel",
    "sobel_definition": "agfb_filters.filters.sobel",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
