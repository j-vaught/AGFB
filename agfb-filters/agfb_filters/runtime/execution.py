"""Explicit execution types shared by filter definitions and runners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ExecutionPath(StrEnum):
    """Concrete runner paths supported by the filter library."""

    SEPARABLE = "separable"
    SPATIAL_DENSE = "spatial_dense"
    FFT = "fft"
    SPARSE_OFFSETS = "sparse_offsets"
    ANTIPODAL_PAIRS = "antipodal_pairs"
    STENCIL = "stencil"
    BOX_INTEGRAL = "box_integral"
    RECURSIVE = "recursive"
    NONLINEAR_WINDOW = "nonlinear_window"
    ITERATIVE = "iterative"
    ORIENTATION_BANK = "orientation_bank"


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


def concrete_path(path: ExecutionPath | str) -> ExecutionPath:
    """Resolve a path argument and reject unknown values."""
    return path if isinstance(path, ExecutionPath) else ExecutionPath(str(path))
