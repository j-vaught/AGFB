"""Execution path planning types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

PATH_VERSION = "2"


class ExecutionPath(StrEnum):
    """Concrete runner paths supported by AGFB."""

    SEPARABLE = "separable"
    SPATIAL_DENSE = "spatial_dense"
    FFT = "fft"
    SPARSE_OFFSETS = "sparse_offsets"
    ANTIPODAL_PAIRS = "antipodal_pairs"
    STENCIL = "stencil"


class BoundaryMode(StrEnum):
    """Boundary modes supported by the runner."""

    REFLECT = "reflect"
    REPLICATE = "replicate"
    CONSTANT = "constant"
    CIRCULAR = "circular"


@dataclass(frozen=True)
class BoundaryCondition:
    """Padding boundary condition for same-shape filter execution."""

    mode: BoundaryMode
    value: float = 0.0

    def __post_init__(self) -> None:
        try:
            mode = (
                self.mode if isinstance(self.mode, BoundaryMode) else BoundaryMode(str(self.mode))
            )
        except ValueError as error:
            supported = ", ".join(mode.value for mode in BoundaryMode)
            raise ValueError(
                f"unsupported boundary mode {self.mode!r}; supported modes are {supported}"
            ) from error

        value = float(self.value)
        if mode != BoundaryMode.CONSTANT and value != 0.0:
            raise ValueError("boundary value is only supported for constant mode")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "value", value)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "value": self.value,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> BoundaryCondition:
        return cls(
            mode=BoundaryMode(str(data["mode"])),
            value=float(data.get("value", 0.0)),
        )


def dtype_name(dtype: torch.dtype | str) -> str:
    """Return a stable short dtype name such as `float32`."""
    if isinstance(dtype, str):
        return dtype.removeprefix("torch.")
    return str(dtype).removeprefix("torch.")


def torch_dtype(dtype: str) -> torch.dtype:
    """Resolve a short dtype name into a torch dtype."""
    resolved = getattr(torch, dtype.removeprefix("torch."), None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"unsupported dtype {dtype!r}")
    return resolved


def device_name(device: torch.device | str) -> str:
    """Return a cache-stable device name for the current runtime."""
    device_object = torch.device(device)
    if device_object.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device_object)
    return device_object.type


@dataclass(frozen=True)
class InputSignature:
    """Shape, dtype, device, and gradient requirements for a filter input."""

    batch: int
    height: int
    width: int
    dtype: str = "float32"
    device_type: str = "cpu"
    device_name: str = "cpu"
    requires_grad: bool = False

    @classmethod
    def from_tensor(cls, image: torch.Tensor) -> InputSignature:
        """Build an input signature from an existing tensor."""
        return cls(
            batch=int(image.shape[0]),
            height=int(image.shape[1]),
            width=int(image.shape[2]),
            dtype=dtype_name(image.dtype),
            device_type=image.device.type,
            device_name=device_name(image.device),
            requires_grad=bool(image.requires_grad),
        )

    @classmethod
    def from_values(
        cls,
        *,
        batch: int,
        height: int,
        width: int,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        requires_grad: bool = False,
    ) -> InputSignature:
        """Build an input signature without allocating an input tensor."""
        device_object = torch.device(device)
        return cls(
            batch=int(batch),
            height=int(height),
            width=int(width),
            dtype=dtype_name(dtype),
            device_type=device_object.type,
            device_name=device_name(device_object),
            requires_grad=bool(requires_grad),
        )

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch_dtype(self.dtype)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device_type)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "height": self.height,
            "width": self.width,
            "dtype": self.dtype,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "requires_grad": self.requires_grad,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> InputSignature:
        return cls(
            batch=int(data["batch"]),
            height=int(data["height"]),
            width=int(data["width"]),
            dtype=str(data["dtype"]),
            device_type=str(data["device_type"]),
            device_name=str(data["device_name"]),
            requires_grad=bool(data["requires_grad"]),
        )


@dataclass(frozen=True)
class BenchmarkResult:
    """Empirical timing for one candidate execution path."""

    path: ExecutionPath
    median_seconds: float
    iqr_seconds: float
    number_per_run: int
    rounds: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "median_seconds": self.median_seconds,
            "iqr_seconds": self.iqr_seconds,
            "number_per_run": self.number_per_run,
            "rounds": self.rounds,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        return cls(
            path=ExecutionPath(str(data["path"])),
            median_seconds=float(data["median_seconds"]),
            iqr_seconds=float(data["iqr_seconds"]),
            number_per_run=int(data["number_per_run"]),
            rounds=int(data["rounds"]),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for empirical AutoRunner path selection."""

    candidate_paths: tuple[ExecutionPath, ...] | None = None
    warmup_runs: int = 3
    min_run_time: float = 0.05


@dataclass(frozen=True)
class ExecutionPlan:
    """A concrete path recommendation for one filter and input signature."""

    path: ExecutionPath
    input_signature: InputSignature
    boundary: BoundaryCondition
    filter_fingerprint: str
    reason: str
    path_version: str = PATH_VERSION
    estimated_cost: float | None = None
    modifiers: tuple[str, ...] = field(default_factory=tuple)
    benchmark_result: BenchmarkResult | None = None
    benchmark_results: tuple[BenchmarkResult, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "input_signature": self.input_signature.to_json_dict(),
            "boundary": self.boundary.to_json_dict(),
            "filter_fingerprint": self.filter_fingerprint,
            "reason": self.reason,
            "path_version": self.path_version,
            "estimated_cost": self.estimated_cost,
            "modifiers": list(self.modifiers),
            "benchmark_result": (
                None if self.benchmark_result is None else self.benchmark_result.to_json_dict()
            ),
            "benchmark_results": [result.to_json_dict() for result in self.benchmark_results],
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        benchmark_result_data = data.get("benchmark_result")
        return cls(
            path=ExecutionPath(str(data["path"])),
            input_signature=InputSignature.from_json_dict(data["input_signature"]),
            boundary=BoundaryCondition.from_json_dict(data["boundary"]),
            filter_fingerprint=str(data["filter_fingerprint"]),
            reason=str(data["reason"]),
            path_version=str(data.get("path_version", PATH_VERSION)),
            estimated_cost=(
                None if data.get("estimated_cost") is None else float(data["estimated_cost"])
            ),
            modifiers=tuple(str(modifier) for modifier in data.get("modifiers", [])),
            benchmark_result=(
                None
                if benchmark_result_data is None
                else BenchmarkResult.from_json_dict(benchmark_result_data)
            ),
            benchmark_results=tuple(
                BenchmarkResult.from_json_dict(result)
                for result in data.get("benchmark_results", [])
            ),
        )


def concrete_path(path: ExecutionPath | ExecutionPlan | str) -> ExecutionPath:
    """Resolve a path argument and reject automatic or unknown values."""
    if isinstance(path, ExecutionPlan):
        return path.path
    return ExecutionPath(path)
