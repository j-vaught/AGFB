"""Batched, GPU-accelerated filters for the Analytical Gradient Filter Benchmark."""

from agfb_filters.filters.central_difference import (
    central_difference,
    central_difference_definition,
)
from agfb_filters.filters.cpgf import CPGF, cpgf_definition, cpgf_kernels
from agfb_filters.filters.definitions import (
    GradientFilterDefinition,
    define_dense_filter,
    define_separable_filter,
)
from agfb_filters.filters.derivative_of_gaussian import (
    DerivativeOfGaussian,
    derivative_of_gaussian_definition,
)
from agfb_filters.filters.farid_simoncelli import (
    farid_simoncelli_5,
    farid_simoncelli_5_definition,
)
from agfb_filters.filters.freeman_adelson import FreemanAdelsonG1, freeman_adelson_g1_definition
from agfb_filters.filters.prewitt import prewitt_3, prewitt_3_definition
from agfb_filters.filters.registry import (
    FilterRegistration,
    get_filter_definition,
    get_filter_registration,
    register_filter,
    registered_filters,
)
from agfb_filters.filters.roberts import roberts, roberts_definition
from agfb_filters.filters.savitzky_golay import (
    SavitzkyGolay,
    savitzky_golay_definition,
    savitzky_golay_kernels,
)
from agfb_filters.filters.scharr import scharr_3, scharr_3_definition
from agfb_filters.filters.sobel import sobel_3, sobel_5, sobel_7, sobel_definition
from agfb_filters.runtime.autorunner import AutoRunner
from agfb_filters.runtime.execution import (
    BenchmarkConfig,
    BenchmarkResult,
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    ExecutionPlan,
    InputSignature,
)
from agfb_filters.runtime.runner import run_filter

__all__ = [
    "AutoRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BoundaryCondition",
    "BoundaryMode",
    "CPGF",
    "DerivativeOfGaussian",
    "ExecutionPath",
    "ExecutionPlan",
    "FilterRegistration",
    "FreemanAdelsonG1",
    "GradientFilterDefinition",
    "InputSignature",
    "SavitzkyGolay",
    "central_difference",
    "central_difference_definition",
    "cpgf_definition",
    "cpgf_kernels",
    "define_dense_filter",
    "define_separable_filter",
    "derivative_of_gaussian_definition",
    "farid_simoncelli_5",
    "farid_simoncelli_5_definition",
    "freeman_adelson_g1_definition",
    "get_filter_definition",
    "get_filter_registration",
    "prewitt_3",
    "prewitt_3_definition",
    "register_filter",
    "registered_filters",
    "roberts",
    "roberts_definition",
    "run_filter",
    "scharr_3",
    "scharr_3_definition",
    "savitzky_golay_definition",
    "savitzky_golay_kernels",
    "sobel_3",
    "sobel_5",
    "sobel_7",
    "sobel_definition",
]
