"""Shared filter definitions used by generators and runners."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode

MetadataValue = int | float | str
TensorLike1D = torch.Tensor | Sequence[float]
TensorLike2D = torch.Tensor | Sequence[Sequence[float]]
_DEFAULT_CUSTOM_BOUNDARY = BoundaryCondition(BoundaryMode.REPLICATE)


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
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

        has_kernel_x = self.kernel_x is not None
        has_kernel_y = self.kernel_y is not None
        has_smooth_kernel = self.smooth_kernel_1d is not None
        has_derivative_kernel = self.derivative_kernel_1d is not None

        if has_kernel_x != has_kernel_y:
            raise ValueError("kernel_x and kernel_y must be provided together")
        if has_smooth_kernel != has_derivative_kernel:
            raise ValueError("smooth_kernel_1d and derivative_kernel_1d must be provided together")
        if self.has_dense_kernels:
            _validate_dense_kernel_pair(self.kernel_x, self.kernel_y)
        if self.has_separable_kernels:
            _validate_separable_kernel_pair(self.smooth_kernel_1d, self.derivative_kernel_1d)
        if not self.has_dense_kernels and self.has_separable_kernels:
            kernel_x, kernel_y = dense_kernels_from_separable(
                self.smooth_kernel_1d,
                self.derivative_kernel_1d,
            )
            object.__setattr__(self, "kernel_x", kernel_x)
            object.__setattr__(self, "kernel_y", kernel_y)
        if not self.has_dense_kernels and not self.has_separable_kernels:
            raise ValueError("a filter definition must provide dense or separable kernels")
        if self.spatial_padding is not None:
            spatial_padding = _validated_spatial_padding(self.spatial_padding)
            object.__setattr__(self, "spatial_padding", spatial_padding)
        if self.has_dense_kernels:
            _validate_dense_padding(self.kernel_x, self.spatial_padding)

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


def define_dense_filter(
    *,
    name: str,
    kernel_x: TensorLike2D,
    kernel_y: TensorLike2D,
    default_boundary: BoundaryCondition = _DEFAULT_CUSTOM_BOUNDARY,
    spatial_padding: tuple[int, int, int, int] | None = None,
    support: str = "dense",
    symmetry: str | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> GradientFilterDefinition:
    """Create a dense custom gradient filter definition."""
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        kernel_x=_as_tensor(kernel_x, name="kernel_x", ndim=2, dtype=dtype, device=device),
        kernel_y=_as_tensor(kernel_y, name="kernel_y", ndim=2, dtype=dtype, device=device),
        spatial_padding=spatial_padding,
        support=support,
        symmetry=symmetry,
        metadata={} if metadata is None else metadata,
    )


def define_separable_filter(
    *,
    name: str,
    smooth_kernel_1d: TensorLike1D,
    derivative_kernel_1d: TensorLike1D,
    default_boundary: BoundaryCondition = _DEFAULT_CUSTOM_BOUNDARY,
    support: str = "separable",
    symmetry: str | None = "odd",
    metadata: Mapping[str, MetadataValue] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> GradientFilterDefinition:
    """Create a separable custom gradient filter definition."""
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        smooth_kernel_1d=_as_tensor(
            smooth_kernel_1d,
            name="smooth_kernel_1d",
            ndim=1,
            dtype=dtype,
            device=device,
        ),
        derivative_kernel_1d=_as_tensor(
            derivative_kernel_1d,
            name="derivative_kernel_1d",
            ndim=1,
            dtype=dtype,
            device=device,
        ),
        support=support,
        symmetry=symmetry,
        metadata={} if metadata is None else metadata,
    )


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


def _as_tensor(
    value: torch.Tensor | Sequence[Any],
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device).clone().detach()
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {tensor.ndim} dimensions")
    return tensor


def _validated_metadata(metadata: Mapping[str, MetadataValue]) -> dict[str, MetadataValue]:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    validated: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if not isinstance(value, int | float | str):
            raise ValueError(f"metadata value for {key!r} must be int, float, or str")
        validated[key] = value
    return validated


def _validate_tensor(
    tensor: torch.Tensor | None,
    *,
    name: str,
    ndim: int,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {tensor.ndim} dimensions")
    if not tensor.dtype.is_floating_point:
        raise ValueError(f"{name} must use a floating-point dtype")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validate_dense_kernel_pair(
    kernel_x: torch.Tensor | None,
    kernel_y: torch.Tensor | None,
) -> None:
    kernel_x = _validate_tensor(kernel_x, name="kernel_x", ndim=2)
    kernel_y = _validate_tensor(kernel_y, name="kernel_y", ndim=2)
    if kernel_x.shape != kernel_y.shape:
        raise ValueError(f"kernel shapes must match, got {kernel_x.shape} vs {kernel_y.shape}")
    if kernel_x.device != kernel_y.device:
        raise ValueError("kernel_x and kernel_y must be on the same device")


def _validate_separable_kernel_pair(
    smooth_kernel_1d: torch.Tensor | None,
    derivative_kernel_1d: torch.Tensor | None,
) -> None:
    smooth_kernel_1d = _validate_tensor(
        smooth_kernel_1d,
        name="smooth_kernel_1d",
        ndim=1,
    )
    derivative_kernel_1d = _validate_tensor(
        derivative_kernel_1d,
        name="derivative_kernel_1d",
        ndim=1,
    )
    if smooth_kernel_1d.shape[0] % 2 == 0 or derivative_kernel_1d.shape[0] % 2 == 0:
        raise ValueError("separable kernels must have odd lengths")
    if smooth_kernel_1d.device != derivative_kernel_1d.device:
        raise ValueError("smooth_kernel_1d and derivative_kernel_1d must be on the same device")


def _validated_spatial_padding(
    spatial_padding: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if len(spatial_padding) != 4:
        raise ValueError("spatial_padding must be (left, right, top, bottom)")
    left, right, top, bottom = (int(amount) for amount in spatial_padding)
    padding = (left, right, top, bottom)
    if any(amount < 0 for amount in padding):
        raise ValueError(f"spatial_padding amounts must be nonnegative, got {padding}")
    return padding


def _validate_dense_padding(
    kernel_x: torch.Tensor | None,
    spatial_padding: tuple[int, int, int, int] | None,
) -> None:
    kernel = _validate_tensor(kernel_x, name="kernel_x", ndim=2)
    kernel_height = int(kernel.shape[0])
    kernel_width = int(kernel.shape[1])
    if spatial_padding is None:
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("even-sized dense kernels require explicit spatial_padding")
        return

    left, right, top, bottom = spatial_padding
    if left + right != kernel_width - 1 or top + bottom != kernel_height - 1:
        raise ValueError(
            "spatial_padding must preserve input shape for dense kernels; "
            f"got {spatial_padding} for kernel shape {kernel_height}x{kernel_width}"
        )
