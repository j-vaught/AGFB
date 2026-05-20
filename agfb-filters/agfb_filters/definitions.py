"""Shared filter definitions used by generators and runners."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from agfb_filters.execution import BoundaryCondition

MetadataValue = int | float | str


@dataclass(frozen=True)
class GradientFilterDefinition:
    """Math and weight definition for one gradient filter."""

    name: str
    default_boundary: BoundaryCondition
    kernel_x: torch.Tensor | None = None
    kernel_y: torch.Tensor | None = None
    smooth_kernel_1d: torch.Tensor | None = None
    derivative_kernel_1d: torch.Tensor | None = None
    spatial_padding: tuple[int, int, int, int] | None = None
    support: str = "dense"
    symmetry: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.default_boundary, BoundaryCondition):
            object.__setattr__(self, "default_boundary", BoundaryCondition(self.default_boundary))

        has_kernel_x = self.kernel_x is not None
        has_kernel_y = self.kernel_y is not None
        has_smooth_kernel = self.smooth_kernel_1d is not None
        has_derivative_kernel = self.derivative_kernel_1d is not None

        if has_kernel_x != has_kernel_y:
            raise ValueError("kernel_x and kernel_y must be provided together")
        if has_smooth_kernel != has_derivative_kernel:
            raise ValueError("smooth_kernel_1d and derivative_kernel_1d must be provided together")
        if not self.has_dense_kernels and self.has_separable_kernels:
            kernel_x, kernel_y = dense_kernels_from_separable(
                self.smooth_kernel_1d,
                self.derivative_kernel_1d,
            )
            object.__setattr__(self, "kernel_x", kernel_x)
            object.__setattr__(self, "kernel_y", kernel_y)
        if not self.has_dense_kernels and not self.has_separable_kernels:
            raise ValueError("a filter definition must provide dense or separable kernels")
        if self.spatial_padding is not None and len(self.spatial_padding) != 4:
            raise ValueError("spatial_padding must be (left, right, top, bottom)")

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

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the filter structure and weights."""
        hasher = hashlib.sha256()
        payload = {
            "name": self.name,
            "default_boundary": self.default_boundary.to_json_dict(),
            "spatial_padding": self.spatial_padding,
            "support": self.support,
            "symmetry": self.symmetry,
            "metadata": dict(sorted(self.metadata.items())),
        }
        hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for label, tensor in (
            ("kernel_x", self.kernel_x),
            ("kernel_y", self.kernel_y),
            ("smooth_kernel_1d", self.smooth_kernel_1d),
            ("derivative_kernel_1d", self.derivative_kernel_1d),
        ):
            if tensor is not None:
                _update_tensor_hash(hasher, label, tensor)
        return hasher.hexdigest()


def dense_kernels_from_separable(
    smooth_kernel_1d: torch.Tensor | None,
    derivative_kernel_1d: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build centered dense gradient kernels from separable 1-D kernels."""
    if smooth_kernel_1d is None or derivative_kernel_1d is None:
        raise ValueError("separable kernels must be provided together")
    if smooth_kernel_1d.ndim != 1 or derivative_kernel_1d.ndim != 1:
        raise ValueError("separable kernels must be 1-D")
    smooth_size = int(smooth_kernel_1d.shape[0])
    derivative_size = int(derivative_kernel_1d.shape[0])
    if smooth_size % 2 == 0 or derivative_size % 2 == 0:
        raise ValueError("separable kernels must have odd lengths")

    dense_size = max(smooth_size, derivative_size)
    dtype = torch.promote_types(smooth_kernel_1d.dtype, derivative_kernel_1d.dtype)
    device = smooth_kernel_1d.device
    smooth = smooth_kernel_1d.to(device=device, dtype=dtype)
    derivative = derivative_kernel_1d.to(device=device, dtype=dtype)
    kernel_x = torch.zeros(dense_size, dense_size, dtype=dtype, device=device)
    kernel_y = torch.zeros(dense_size, dense_size, dtype=dtype, device=device)

    smooth_start = (dense_size - smooth_size) // 2
    derivative_start = (dense_size - derivative_size) // 2
    kernel_x[
        smooth_start : smooth_start + smooth_size,
        derivative_start : derivative_start + derivative_size,
    ] = torch.outer(smooth, derivative)
    kernel_y[
        derivative_start : derivative_start + derivative_size,
        smooth_start : smooth_start + smooth_size,
    ] = torch.outer(derivative, smooth)
    return kernel_x, kernel_y


def _update_tensor_hash(
    hasher: Any,
    label: str,
    tensor: torch.Tensor,
) -> None:
    hasher.update(label.encode("utf-8"))
    hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
    hasher.update(str(tensor.dtype).encode("utf-8"))
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    hasher.update(cpu_tensor.numpy().tobytes())
