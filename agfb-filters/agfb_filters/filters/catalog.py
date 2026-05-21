"""Single source of truth for shipped filter metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
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


_FILTER_CORE_EXPORTS = {
    "BuiltInFilterSpec": "agfb_filters.filters.catalog",
    "FilterRegistration": "agfb_filters.filters.registry",
    "GradientFilterDefinition": "agfb_filters.filters.definitions",
    "define_dense_filter": "agfb_filters.filters.definitions",
    "define_separable_filter": "agfb_filters.filters.definitions",
    "get_filter_definition": "agfb_filters.filters.registry",
    "get_filter_registration": "agfb_filters.filters.registry",
    "register_filter": "agfb_filters.filters.registry",
    "registered_filters": "agfb_filters.filters.registry",
    "shipped_filter_specs": "agfb_filters.filters.catalog",
}

_RUNTIME_EXPORTS = {
    "AutoRunner": "agfb_filters.runtime.autorunner",
    "BenchmarkConfig": "agfb_filters.runtime.execution",
    "BenchmarkResult": "agfb_filters.runtime.execution",
    "BoundaryCondition": "agfb_filters.runtime.execution",
    "BoundaryMode": "agfb_filters.runtime.execution",
    "ExecutionPath": "agfb_filters.runtime.execution",
    "ExecutionPlan": "agfb_filters.runtime.execution",
    "InputSignature": "agfb_filters.runtime.execution",
    "run_filter": "agfb_filters.runtime.runner",
}

_SHIPPED_FILTER_SPECS = (
    BuiltInFilterSpec(
        name="central_difference",
        module="agfb_filters.filters.central_difference",
        definition_factory="central_difference_definition",
        description="central finite difference",
        exports=("central_difference", "central_difference_definition"),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="farid_simoncelli_5",
        module="agfb_filters.filters.farid_simoncelli",
        definition_factory="farid_simoncelli_5_definition",
        description="Farid-Simoncelli 5-tap",
        exports=("farid_simoncelli_5", "farid_simoncelli_5_definition"),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="prewitt_3",
        module="agfb_filters.filters.prewitt",
        definition_factory="prewitt_3_definition",
        description="Prewitt 3-tap",
        exports=("prewitt_3", "prewitt_3_definition"),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="roberts",
        module="agfb_filters.filters.roberts",
        definition_factory="roberts_definition",
        description="Roberts cross",
        exports=("roberts", "roberts_definition"),
        smoke_path="stencil",
    ),
    BuiltInFilterSpec(
        name="scharr_3",
        module="agfb_filters.filters.scharr",
        definition_factory="scharr_3_definition",
        description="Scharr 3-tap",
        exports=("scharr_3", "scharr_3_definition"),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="sobel_3",
        module="agfb_filters.filters.sobel",
        definition_factory="sobel_definition",
        description="Sobel 3-tap",
        exports=("sobel_3", "sobel_definition"),
        registry_kwargs=MappingProxyType({"kernel_size": 3}),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="sobel_5",
        module="agfb_filters.filters.sobel",
        definition_factory="sobel_definition",
        description="Sobel 5-tap",
        exports=("sobel_5",),
        registry_kwargs=MappingProxyType({"kernel_size": 5}),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="sobel_7",
        module="agfb_filters.filters.sobel",
        definition_factory="sobel_definition",
        description="Sobel 7-tap",
        exports=("sobel_7",),
        registry_kwargs=MappingProxyType({"kernel_size": 7}),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="cpgf",
        module="agfb_filters.filters.cpgf",
        definition_factory="cpgf_definition",
        description="circular polynomial gradient filter",
        exports=("CPGF", "cpgf_definition", "cpgf_kernels"),
        smoke_kwargs=MappingProxyType({"radius": 2, "degree": 2}),
        smoke_path="sparse_offsets",
    ),
    BuiltInFilterSpec(
        name="derivative_of_gaussian",
        module="agfb_filters.filters.derivative_of_gaussian",
        definition_factory="derivative_of_gaussian_definition",
        description="first derivative of Gaussian",
        exports=("DerivativeOfGaussian", "derivative_of_gaussian_definition"),
        smoke_kwargs=MappingProxyType({"sigma": 1.0}),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="freeman_adelson_g1",
        module="agfb_filters.filters.freeman_adelson",
        definition_factory="freeman_adelson_g1_definition",
        description="Freeman-Adelson G1",
        exports=("FreemanAdelsonG1", "freeman_adelson_g1_definition"),
        smoke_kwargs=MappingProxyType({"sigma": 1.0}),
        smoke_path="separable",
    ),
    BuiltInFilterSpec(
        name="savitzky_golay",
        module="agfb_filters.filters.savitzky_golay",
        definition_factory="savitzky_golay_definition",
        description="Savitzky-Golay square fit",
        exports=("SavitzkyGolay", "savitzky_golay_definition", "savitzky_golay_kernels"),
        smoke_kwargs=MappingProxyType({"radius": 2, "degree": 2}),
        smoke_path="spatial_dense",
    ),
)


def shipped_filter_specs() -> tuple[BuiltInFilterSpec, ...]:
    """Return metadata for filters shipped with the package."""
    return _SHIPPED_FILTER_SPECS


def filter_export_modules() -> dict[str, str]:
    """Return public exports for `agfb_filters.filters`."""
    exports = dict(_FILTER_CORE_EXPORTS)
    for spec in _SHIPPED_FILTER_SPECS:
        for export_name in spec.exports:
            exports[export_name] = spec.module
    return exports


def root_export_modules() -> dict[str, str]:
    """Return public exports for `agfb_filters`."""
    exports = filter_export_modules()
    exports.update(_RUNTIME_EXPORTS)
    return exports
