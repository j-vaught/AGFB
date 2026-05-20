"""Shared filter definitions used by generators and runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import torch

MetadataValue = int | float | str


class ExecutionStrategy(StrEnum):
    """Runner strategies supported by AGFB."""

    AUTO = "auto"
    SEPARABLE = "separable"
    SPATIAL = "spatial"
    FFT = "fft"


@dataclass(frozen=True)
class GradientFilterDefinition:
    """Math and weight definition for one gradient filter."""

    name: str
    padding_mode: str
    kernel_x: torch.Tensor | None = None
    kernel_y: torch.Tensor | None = None
    smooth_kernel_1d: torch.Tensor | None = None
    derivative_kernel_1d: torch.Tensor | None = None
    strategy_hint: ExecutionStrategy = ExecutionStrategy.AUTO
    spatial_padding: tuple[int, int, int, int] | None = None
    support: str = "dense"
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_kernel_x = self.kernel_x is not None
        has_kernel_y = self.kernel_y is not None
        has_smooth_kernel = self.smooth_kernel_1d is not None
        has_derivative_kernel = self.derivative_kernel_1d is not None

        if has_kernel_x != has_kernel_y:
            raise ValueError("kernel_x and kernel_y must be provided together")
        if has_smooth_kernel != has_derivative_kernel:
            raise ValueError("smooth_kernel_1d and derivative_kernel_1d must be provided together")
        if not self.has_dense_kernels and not self.has_separable_kernels:
            raise ValueError("a filter definition must provide dense or separable kernels")
        if self.strategy_hint == ExecutionStrategy.SEPARABLE and not self.has_separable_kernels:
            raise ValueError("separable strategy requires 1-D smooth and derivative kernels")
        if (
            self.strategy_hint in {ExecutionStrategy.SPATIAL, ExecutionStrategy.FFT}
            and not self.has_dense_kernels
        ):
            raise ValueError("spatial and FFT strategies require dense kernels")

    @property
    def has_dense_kernels(self) -> bool:
        return self.kernel_x is not None and self.kernel_y is not None

    @property
    def has_separable_kernels(self) -> bool:
        return self.smooth_kernel_1d is not None and self.derivative_kernel_1d is not None

    def dense_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kernel_x is None or self.kernel_y is None:
            raise ValueError(f"{self.name} does not define dense kernels")
        return self.kernel_x, self.kernel_y

    def separable_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.smooth_kernel_1d is None or self.derivative_kernel_1d is None:
            raise ValueError(f"{self.name} does not define separable kernels")
        return self.smooth_kernel_1d, self.derivative_kernel_1d
