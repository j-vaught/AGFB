"""Batched, GPU-accelerated filters for the Analytical Gradient Filter Benchmark."""

from agfb_filters.autorunner import AutoRunner
from agfb_filters.central_difference import central_difference, central_difference_definition
from agfb_filters.cpgf import CPGF, cpgf_definition, cpgf_kernels
from agfb_filters.definitions import GradientFilterDefinition
from agfb_filters.derivative_of_gaussian import (
    DerivativeOfGaussian,
    derivative_of_gaussian_definition,
)
from agfb_filters.execution import (
    BenchmarkConfig,
    BenchmarkResult,
    ExecutionPath,
    ExecutionPlan,
    InputSignature,
)
from agfb_filters.farid_simoncelli import (
    farid_simoncelli_5,
    farid_simoncelli_5_definition,
)
from agfb_filters.freeman_adelson import FreemanAdelsonG1, freeman_adelson_g1_definition
from agfb_filters.prewitt import prewitt_3, prewitt_3_definition
from agfb_filters.roberts import roberts, roberts_definition
from agfb_filters.runner import run_filter
from agfb_filters.savitzky_golay import (
    SavitzkyGolay,
    savitzky_golay_definition,
    savitzky_golay_kernels,
)
from agfb_filters.scharr import scharr_3, scharr_3_definition
from agfb_filters.sobel import sobel_3, sobel_5, sobel_7, sobel_definition

__all__ = [
    "AutoRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "CPGF",
    "DerivativeOfGaussian",
    "ExecutionPath",
    "ExecutionPlan",
    "FreemanAdelsonG1",
    "GradientFilterDefinition",
    "InputSignature",
    "SavitzkyGolay",
    "central_difference",
    "central_difference_definition",
    "cpgf_definition",
    "cpgf_kernels",
    "derivative_of_gaussian_definition",
    "farid_simoncelli_5",
    "farid_simoncelli_5_definition",
    "freeman_adelson_g1_definition",
    "prewitt_3",
    "prewitt_3_definition",
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
