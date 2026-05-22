"""Catalog collection for shipped filter metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class BuiltInFilterSpec:
    """Metadata needed to expose, register, and smoke-test a shipped filter."""

    name: str
    module: str
    definition_factory: str
    description: str
    exports: tuple[str, ...]
    registry_kwargs: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    smoke_kwargs: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    smoke_path: str = "spatial_dense"
    output_api: str = "gradient"


_FILTER_MODULES = (
    "agfb_filters.filters.central_difference",
    "agfb_filters.filters.farid_simoncelli",
    "agfb_filters.filters.prewitt",
    "agfb_filters.filters.roberts",
    "agfb_filters.filters.scharr",
    "agfb_filters.filters.sobel",
    "agfb_filters.filters.cpgf",
    "agfb_filters.filters.derivative_of_gaussian",
    "agfb_filters.filters.freeman_adelson",
    "agfb_filters.filters.savitzky_golay",
    "agfb_filters.filters.sparse",
    "agfb_filters.filters.box",
    "agfb_filters.filters.recursive",
    "agfb_filters.filters.riesz",
    "agfb_filters.filters.nonlinear",
    "agfb_filters.filters.iterative",
    "agfb_filters.filters.orientation_bank",
    "agfb_filters.filters.orientable",
)

_FILTER_CORE_EXPORTS = {
    "BuiltInFilterSpec": "agfb_filters.filters.catalog",
    "FilterImplementationKind": "agfb_filters.filters.definitions",
    "FilterRegistration": "agfb_filters.filters.registry",
    "GradientFilterDefinition": "agfb_filters.filters.definitions",
    "GradientFilterImplementation": "agfb_filters.filters.definitions",
    "define_box_gradient_filter": "agfb_filters.filters.definitions",
    "define_dense_filter": "agfb_filters.filters.definitions",
    "define_iterative_filter": "agfb_filters.filters.definitions",
    "define_nonlinear_window_filter": "agfb_filters.filters.definitions",
    "define_orientation_bank_filter": "agfb_filters.filters.definitions",
    "define_recursive_filter": "agfb_filters.filters.definitions",
    "define_riesz_filter": "agfb_filters.filters.definitions",
    "define_separable_filter": "agfb_filters.filters.definitions",
    "define_sparse_offset_filter": "agfb_filters.filters.definitions",
    "get_filter_definition": "agfb_filters.filters.registry",
    "get_filter_registration": "agfb_filters.filters.registry",
    "multiscale_gaussian_derivative_orientation_banks": "agfb_filters.filters.orientable",
    "recursive_gaussian_derivative_orientation_bank": "agfb_filters.filters.orientable",
    "register_filter": "agfb_filters.filters.registry",
    "registered_filters": "agfb_filters.filters.registry",
    "riesz_orientation_bank": "agfb_filters.filters.orientable",
    "shipped_filter_specs": "agfb_filters.filters.catalog",
}

_RUNTIME_EXPORTS = {
    "BoundaryCondition": "agfb_filters.runtime.execution",
    "BoundaryMode": "agfb_filters.runtime.execution",
    "CollapsedOrientationBank": "agfb_filters.runtime.runner",
    "ExecutionPath": "agfb_filters.runtime.execution",
    "OrientationBankResult": "agfb_filters.runtime.runner",
    "collapse_orientation_bank": "agfb_filters.runtime.runner",
    "orientation_angles": "agfb_filters.runtime.runner",
    "run_filter": "agfb_filters.runtime.runner",
    "run_orientation_bank": "agfb_filters.runtime.runner",
    "run_steered_filter_bank": "agfb_filters.runtime.runner",
    "steer_gradient": "agfb_filters.runtime.runner",
}


def shipped_filter_specs() -> tuple[BuiltInFilterSpec, ...]:
    """Return metadata for filters shipped with the package."""
    specs: list[BuiltInFilterSpec] = []
    for module_name in _FILTER_MODULES:
        module = import_module(module_name)
        for spec_data in getattr(module, "FILTER_SPECS", ()):
            specs.append(_built_in_filter_spec(module_name, spec_data))
    return tuple(specs)


def filter_export_modules() -> dict[str, str]:
    """Return public exports for `agfb_filters.filters`."""
    exports = dict(_FILTER_CORE_EXPORTS)
    for spec in shipped_filter_specs():
        for export_name in spec.exports:
            exports[export_name] = spec.module
    return exports


def root_export_modules() -> dict[str, str]:
    """Return public exports for `agfb_filters`."""
    exports = filter_export_modules()
    exports.update(_RUNTIME_EXPORTS)
    return exports


def _built_in_filter_spec(module_name: str, data: MappingProxyType[str, Any] | dict[str, Any]):
    if not isinstance(data, MappingProxyType):
        data = MappingProxyType(dict(data))
    return BuiltInFilterSpec(
        name=str(data["name"]),
        module=module_name,
        definition_factory=str(data["definition_factory"]),
        description=str(data["description"]),
        exports=tuple(str(export) for export in data["exports"]),
        registry_kwargs=MappingProxyType(dict(data.get("registry_kwargs", {}))),
        smoke_kwargs=MappingProxyType(dict(data.get("smoke_kwargs", {}))),
        smoke_path=str(data.get("smoke_path", "spatial_dense")),
        output_api=str(data.get("output_api", "gradient")),
    )
