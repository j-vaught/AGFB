"""Shared filter definitions used by generators and runners."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode

MetadataValue = bool | int | float | str
ParameterValue = MetadataValue | tuple[MetadataValue, ...]
TensorLike1D = torch.Tensor | Sequence[float]
TensorLike2D = torch.Tensor | Sequence[Sequence[float]]
TensorLikeOffsets = torch.Tensor | Sequence[tuple[int, int]] | Sequence[Sequence[int]]
_DEFAULT_CUSTOM_BOUNDARY = BoundaryCondition(BoundaryMode.REPLICATE)
_DEFAULT_REFLECT_BOUNDARY = BoundaryCondition(BoundaryMode.REFLECT)
_DEFAULT_CIRCULAR_BOUNDARY = BoundaryCondition(BoundaryMode.CIRCULAR)


class FilterImplementationKind(StrEnum):
    """Validated execution families carried by a filter definition."""

    FIR = "fir"
    SPARSE_OFFSET = "sparse_offset"
    BOX_GRADIENT = "box_gradient"
    RECURSIVE = "recursive"
    NONLINEAR_WINDOW = "nonlinear_window"
    ITERATIVE = "iterative"
    ORIENTATION_BANK = "orientation_bank"
    RIESZ = "riesz"


@dataclass(frozen=True)
class GradientFilterImplementation:
    """Concrete implementation data for one filter definition."""

    kind: FilterImplementationKind
    kernel_x: torch.Tensor | None = None
    kernel_y: torch.Tensor | None = None
    smooth_kernel_1d: torch.Tensor | None = None
    derivative_kernel_1d: torch.Tensor | None = None
    spatial_padding: tuple[int, int, int, int] | None = None
    sparse_offsets: torch.Tensor | None = None
    sparse_weights_x: torch.Tensor | None = None
    sparse_weights_y: torch.Tensor | None = None
    box_radius: int | None = None
    recursive_sigma: float | None = None
    recursive_method: str | None = None
    nonlinear_radius: int | None = None
    nonlinear_weighting: str | None = None
    nonlinear_range_sigma: float | None = None
    nonlinear_robust_scale: float | None = None
    iterative_method: str | None = None
    iterative_iterations: int | None = None
    iterative_step_size: float | None = None
    iterative_kappa: float | None = None
    iterative_conduction: str | None = None
    iterative_derivative_radius: int | None = None
    orientation_kernels: torch.Tensor | None = None
    angles: torch.Tensor | None = None
    riesz_epsilon: float | None = None

    def __post_init__(self) -> None:
        kind = (
            self.kind
            if isinstance(self.kind, FilterImplementationKind)
            else FilterImplementationKind(str(self.kind))
        )
        object.__setattr__(self, "kind", kind)

        if kind == FilterImplementationKind.FIR:
            _validate_fir_implementation(self)
        elif kind == FilterImplementationKind.SPARSE_OFFSET:
            _validate_sparse_implementation(self)
        elif kind == FilterImplementationKind.BOX_GRADIENT:
            _validate_positive_int(self.box_radius, name="box_radius")
        elif kind == FilterImplementationKind.RECURSIVE:
            _validate_positive_float(self.recursive_sigma, name="recursive_sigma")
            if self.recursive_method not in {"deriche_gaussian_derivative"}:
                raise ValueError("recursive_method must be 'deriche_gaussian_derivative'")
        elif kind == FilterImplementationKind.NONLINEAR_WINDOW:
            _validate_positive_int(self.nonlinear_radius, name="nonlinear_radius")
            if self.nonlinear_weighting not in {"bilateral", "huber", "tukey", "none"}:
                raise ValueError("nonlinear_weighting must be one of bilateral, huber, tukey, none")
            _validate_positive_float(self.nonlinear_range_sigma, name="nonlinear_range_sigma")
            _validate_positive_float(self.nonlinear_robust_scale, name="nonlinear_robust_scale")
        elif kind == FilterImplementationKind.ITERATIVE:
            if self.iterative_method not in {"perona_malik_gradient"}:
                raise ValueError("iterative_method must be 'perona_malik_gradient'")
            _validate_nonnegative_int(self.iterative_iterations, name="iterative_iterations")
            _validate_positive_float(self.iterative_step_size, name="iterative_step_size")
            if _require_float(self.iterative_step_size) > 0.25:
                raise ValueError("iterative_step_size must be <= 0.25 for explicit diffusion")
            _validate_positive_float(self.iterative_kappa, name="iterative_kappa")
            if self.iterative_conduction not in {"exponential", "reciprocal"}:
                raise ValueError("iterative_conduction must be exponential or reciprocal")
            _validate_positive_int(
                self.iterative_derivative_radius,
                name="iterative_derivative_radius",
            )
        elif kind == FilterImplementationKind.ORIENTATION_BANK:
            _validate_orientation_bank_implementation(self)
        elif kind == FilterImplementationKind.RIESZ:
            _validate_positive_float(self.riesz_epsilon, name="riesz_epsilon")


@dataclass(frozen=True)
class GradientFilterDefinition:
    """Math, execution spec, and metadata for one gradient filter."""

    name: str
    default_boundary: BoundaryCondition
    implementation: GradientFilterImplementation | None = None
    kernel_x: torch.Tensor | None = None
    kernel_y: torch.Tensor | None = None
    smooth_kernel_1d: torch.Tensor | None = None
    derivative_kernel_1d: torch.Tensor | None = None
    spatial_padding: tuple[int, int, int, int] | None = None
    support: str = "dense"
    symmetry: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    operator_family: str = "finite_impulse_response"
    linearity: str = "linear"
    stage_count: int = 1
    support_shape: str = "dense"
    orientation_model: str = "cartesian_gradient"
    shape_model: str = "same"
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)
    references: tuple[str, ...] = field(default_factory=tuple)
    supported_boundaries: tuple[BoundaryMode, ...] = field(
        default_factory=lambda: tuple(BoundaryMode)
    )
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.default_boundary, BoundaryCondition):
            object.__setattr__(self, "default_boundary", BoundaryCondition(self.default_boundary))

        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))
        object.__setattr__(self, "parameters", _validated_parameters(self.parameters))
        object.__setattr__(self, "references", _validated_references(self.references))
        supported_boundaries = _validated_supported_boundaries(self.supported_boundaries)
        object.__setattr__(self, "supported_boundaries", supported_boundaries)
        if self.default_boundary.mode not in supported_boundaries:
            raise ValueError("default_boundary must be listed in supported_boundaries")

        implementation = self.implementation
        if implementation is None:
            implementation = _infer_legacy_implementation(self)
        elif not isinstance(implementation, GradientFilterImplementation):
            raise ValueError("implementation must be a GradientFilterImplementation")
        object.__setattr__(self, "implementation", implementation)

        kernel_x = implementation.kernel_x
        kernel_y = implementation.kernel_y
        smooth_kernel = implementation.smooth_kernel_1d
        derivative_kernel = implementation.derivative_kernel_1d

        if implementation.kind == FilterImplementationKind.FIR and kernel_x is None:
            kernel_x, kernel_y = dense_kernels_from_separable(smooth_kernel, derivative_kernel)
            implementation = GradientFilterImplementation(
                kind=FilterImplementationKind.FIR,
                kernel_x=kernel_x,
                kernel_y=kernel_y,
                smooth_kernel_1d=smooth_kernel,
                derivative_kernel_1d=derivative_kernel,
                spatial_padding=implementation.spatial_padding,
            )
            object.__setattr__(self, "implementation", implementation)

        if implementation.spatial_padding is not None:
            object.__setattr__(self, "spatial_padding", implementation.spatial_padding)
        elif self.spatial_padding is not None:
            object.__setattr__(
                self, "spatial_padding", _validated_spatial_padding(self.spatial_padding)
            )

        if implementation.kind == FilterImplementationKind.FIR:
            if implementation.kernel_x is not None:
                _validate_dense_padding(implementation.kernel_x, self.spatial_padding)
            object.__setattr__(self, "kernel_x", implementation.kernel_x)
            object.__setattr__(self, "kernel_y", implementation.kernel_y)
            object.__setattr__(self, "smooth_kernel_1d", implementation.smooth_kernel_1d)
            object.__setattr__(self, "derivative_kernel_1d", implementation.derivative_kernel_1d)

        if self.stage_count < 1:
            raise ValueError("stage_count must be >= 1")
        if not self.name.strip():
            raise ValueError("filter name must not be empty")
        object.__setattr__(self, "_fingerprint", _definition_fingerprint(self))

    @property
    def kind(self) -> FilterImplementationKind:
        return self.require_implementation().kind

    @property
    def has_dense_kernels(self) -> bool:
        return self.kind in {
            FilterImplementationKind.FIR,
            FilterImplementationKind.SPARSE_OFFSET,
            FilterImplementationKind.BOX_GRADIENT,
        }

    @property
    def has_separable_kernels(self) -> bool:
        implementation = self.require_implementation()
        return (
            implementation.kind == FilterImplementationKind.FIR
            and implementation.smooth_kernel_1d is not None
            and implementation.derivative_kernel_1d is not None
        )

    def dense_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        implementation = self.require_implementation()
        if implementation.kind == FilterImplementationKind.FIR:
            if implementation.kernel_x is None or implementation.kernel_y is None:
                raise ValueError(f"{self.name} does not define dense kernels")
            return implementation.kernel_x, implementation.kernel_y
        if implementation.kind == FilterImplementationKind.SPARSE_OFFSET:
            return dense_kernels_from_sparse_offsets(
                implementation.sparse_offsets,
                implementation.sparse_weights_x,
                implementation.sparse_weights_y,
            )
        if implementation.kind == FilterImplementationKind.BOX_GRADIENT:
            return box_gradient_dense_kernels(_require_int(implementation.box_radius))
        raise ValueError(f"{self.name} does not define dense kernels")

    def separable_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        implementation = self.require_implementation()
        if implementation.smooth_kernel_1d is None or implementation.derivative_kernel_1d is None:
            raise ValueError(f"{self.name} does not define separable kernels")
        return implementation.smooth_kernel_1d, implementation.derivative_kernel_1d

    def sparse_offsets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        implementation = self.require_implementation()
        if implementation.kind != FilterImplementationKind.SPARSE_OFFSET:
            raise ValueError(f"{self.name} does not define sparse offsets")
        return (
            _require_tensor(implementation.sparse_offsets),
            _require_tensor(implementation.sparse_weights_x),
            _require_tensor(implementation.sparse_weights_y),
        )

    def supports_boundary(self, boundary: BoundaryCondition) -> bool:
        return boundary.mode in self.supported_boundaries

    def require_implementation(self) -> GradientFilterImplementation:
        """Return the validated implementation spec."""
        if self.implementation is None:
            raise ValueError(f"{self.name} does not have an implementation spec")
        return self.implementation

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the filter structure and weights."""
        return self._fingerprint


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
    operator_family: str = "finite_impulse_response",
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = tuple(BoundaryMode),
) -> GradientFilterDefinition:
    """Create a dense custom gradient filter definition."""
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.FIR,
        kernel_x=_as_tensor(kernel_x, name="kernel_x", ndim=2, dtype=dtype, device=device),
        kernel_y=_as_tensor(kernel_y, name="kernel_y", ndim=2, dtype=dtype, device=device),
        spatial_padding=spatial_padding,
    )
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support=support,
        symmetry=symmetry,
        metadata={} if metadata is None else metadata,
        operator_family=operator_family,
        support_shape=support,
        parameters={} if parameters is None else parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
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
    operator_family: str = "finite_impulse_response",
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = tuple(BoundaryMode),
) -> GradientFilterDefinition:
    """Create a separable custom gradient filter definition."""
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.FIR,
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
    )
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support=support,
        symmetry=symmetry,
        metadata={} if metadata is None else metadata,
        operator_family=operator_family,
        support_shape=support,
        parameters={} if parameters is None else parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
    )


def define_sparse_offset_filter(
    *,
    name: str,
    offsets: TensorLikeOffsets,
    weights_x: TensorLike1D,
    weights_y: TensorLike1D,
    default_boundary: BoundaryCondition = _DEFAULT_CUSTOM_BOUNDARY,
    symmetry: str | None = "odd",
    metadata: Mapping[str, MetadataValue] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = tuple(BoundaryMode),
) -> GradientFilterDefinition:
    """Create a direct sparse-offset gradient filter definition."""
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.SPARSE_OFFSET,
        sparse_offsets=_as_offset_tensor(offsets, device=device),
        sparse_weights_x=_as_tensor(
            weights_x,
            name="weights_x",
            ndim=1,
            dtype=dtype,
            device=device,
        ),
        sparse_weights_y=_as_tensor(
            weights_y,
            name="weights_y",
            ndim=1,
            dtype=dtype,
            device=device,
        ),
    )
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support="sparse_offsets",
        symmetry=symmetry,
        metadata={} if metadata is None else metadata,
        operator_family="finite_impulse_response",
        support_shape="sparse_offsets",
        parameters={} if parameters is None else parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
    )


def define_box_gradient_filter(
    *,
    name: str,
    radius: int,
    default_boundary: BoundaryCondition = _DEFAULT_CUSTOM_BOUNDARY,
    metadata: Mapping[str, MetadataValue] | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = tuple(BoundaryMode),
) -> GradientFilterDefinition:
    """Create a Haar-style rectangular box gradient filter definition."""
    radius = _validate_positive_int(radius, name="radius")
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.BOX_GRADIENT,
        box_radius=radius,
        spatial_padding=(radius, radius, radius, radius),
    )
    merged_parameters: dict[str, ParameterValue] = {"radius": radius}
    if parameters is not None:
        merged_parameters.update(parameters)
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support="box",
        symmetry="odd",
        metadata={} if metadata is None else metadata,
        operator_family="haar_box_gradient",
        support_shape="rectangular_boxes",
        parameters=merged_parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
    )


def define_recursive_filter(
    *,
    name: str,
    sigma: float,
    default_boundary: BoundaryCondition = _DEFAULT_CUSTOM_BOUNDARY,
    metadata: Mapping[str, MetadataValue] | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = (BoundaryMode.REPLICATE,),
) -> GradientFilterDefinition:
    """Create a Deriche-style recursive Gaussian derivative definition."""
    sigma = _validate_positive_float(sigma, name="sigma")
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.RECURSIVE,
        recursive_sigma=sigma,
        recursive_method="deriche_gaussian_derivative",
    )
    merged_parameters: dict[str, ParameterValue] = {"sigma": float(sigma)}
    if parameters is not None:
        merged_parameters.update(parameters)
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support="recursive",
        symmetry="odd",
        metadata={} if metadata is None else metadata,
        operator_family="recursive_gaussian_derivative",
        support_shape="recursive",
        parameters=merged_parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
    )


def define_nonlinear_window_filter(
    *,
    name: str,
    radius: int,
    weighting: str,
    range_sigma: float = 1.0,
    robust_scale: float = 1.0,
    default_boundary: BoundaryCondition = _DEFAULT_REFLECT_BOUNDARY,
    metadata: Mapping[str, MetadataValue] | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = tuple(BoundaryMode),
) -> GradientFilterDefinition:
    """Create a robust local plane gradient definition."""
    radius = _validate_positive_int(radius, name="radius")
    range_sigma = _validate_positive_float(range_sigma, name="range_sigma")
    robust_scale = _validate_positive_float(robust_scale, name="robust_scale")
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.NONLINEAR_WINDOW,
        nonlinear_radius=radius,
        nonlinear_weighting=str(weighting),
        nonlinear_range_sigma=range_sigma,
        nonlinear_robust_scale=robust_scale,
    )
    merged_parameters: dict[str, ParameterValue] = {
        "radius": radius,
        "weighting": str(weighting),
        "range_sigma": float(range_sigma),
        "robust_scale": float(robust_scale),
    }
    if parameters is not None:
        merged_parameters.update(parameters)
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support="local_window",
        symmetry=None,
        metadata={} if metadata is None else metadata,
        operator_family="robust_local_plane",
        linearity="nonlinear",
        support_shape="square",
        parameters=merged_parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
    )


def define_iterative_filter(
    *,
    name: str,
    iterations: int,
    step_size: float,
    kappa: float,
    conduction: str = "exponential",
    derivative_radius: int = 1,
    default_boundary: BoundaryCondition = _DEFAULT_CUSTOM_BOUNDARY,
    metadata: Mapping[str, MetadataValue] | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = tuple(BoundaryMode),
) -> GradientFilterDefinition:
    """Create a named Perona-Malik diffusion plus gradient definition."""
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.ITERATIVE,
        iterative_method="perona_malik_gradient",
        iterative_iterations=iterations,
        iterative_step_size=step_size,
        iterative_kappa=kappa,
        iterative_conduction=conduction,
        iterative_derivative_radius=derivative_radius,
    )
    merged_parameters: dict[str, ParameterValue] = {
        "iterations": int(iterations),
        "step_size": float(step_size),
        "kappa": float(kappa),
        "conduction": str(conduction),
        "derivative_radius": int(derivative_radius),
    }
    if parameters is not None:
        merged_parameters.update(parameters)
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support="iterative",
        symmetry=None,
        metadata={} if metadata is None else metadata,
        operator_family="perona_malik_gradient",
        linearity="nonlinear",
        stage_count=2,
        support_shape="iterative",
        parameters=merged_parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
    )


def define_orientation_bank_filter(
    *,
    name: str,
    angles: TensorLike1D,
    sigma_parallel: float,
    sigma_perpendicular: float,
    truncate: float = 3.0,
    default_boundary: BoundaryCondition = _DEFAULT_CUSTOM_BOUNDARY,
    metadata: Mapping[str, MetadataValue] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = tuple(BoundaryMode),
) -> GradientFilterDefinition:
    """Create rotated anisotropic Gaussian derivative orientation kernels."""
    sigma_parallel = _validate_positive_float(sigma_parallel, name="sigma_parallel")
    sigma_perpendicular = _validate_positive_float(
        sigma_perpendicular,
        name="sigma_perpendicular",
    )
    truncate = _validate_positive_float(truncate, name="truncate")
    angle_tensor = _as_tensor(
        angles,
        name="angles",
        ndim=1,
        dtype=dtype,
        device=device,
    )
    kernels = anisotropic_gaussian_derivative_kernels(
        angle_tensor,
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
        truncate=truncate,
    )
    radius = kernels.shape[-1] // 2
    implementation = GradientFilterImplementation(
        kind=FilterImplementationKind.ORIENTATION_BANK,
        orientation_kernels=kernels,
        angles=angle_tensor,
        spatial_padding=(radius, radius, radius, radius),
    )
    merged_parameters: dict[str, ParameterValue] = {
        "sigma_parallel": float(sigma_parallel),
        "sigma_perpendicular": float(sigma_perpendicular),
        "truncate": float(truncate),
        "angle_count": int(angle_tensor.numel()),
    }
    if parameters is not None:
        merged_parameters.update(parameters)
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=implementation,
        support="orientation_bank",
        symmetry="odd",
        metadata={} if metadata is None else metadata,
        operator_family="anisotropic_gaussian_derivative",
        orientation_model="orientation_bank",
        support_shape="rotated_anisotropic",
        parameters=merged_parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
    )


def define_riesz_filter(
    *,
    name: str,
    default_boundary: BoundaryCondition = _DEFAULT_CIRCULAR_BOUNDARY,
    epsilon: float = 1.0e-12,
    metadata: Mapping[str, MetadataValue] | None = None,
    parameters: Mapping[str, ParameterValue] | None = None,
    references: tuple[str, ...] = (),
    supported_boundaries: tuple[BoundaryMode | str, ...] = (BoundaryMode.CIRCULAR,),
) -> GradientFilterDefinition:
    """Create a first-order Riesz transform vector filter definition."""
    epsilon = _validate_positive_float(epsilon, name="epsilon")
    merged_parameters: dict[str, ParameterValue] = {"epsilon": float(epsilon)}
    if parameters is not None:
        merged_parameters.update(parameters)
    return GradientFilterDefinition(
        name=name,
        default_boundary=default_boundary,
        implementation=GradientFilterImplementation(
            kind=FilterImplementationKind.RIESZ,
            riesz_epsilon=epsilon,
        ),
        support="fft",
        symmetry="odd",
        metadata={} if metadata is None else metadata,
        operator_family="riesz_transform",
        support_shape="global_fft",
        parameters=merged_parameters,
        references=references,
        supported_boundaries=_validated_supported_boundaries(supported_boundaries),
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


def dense_kernels_from_sparse_offsets(
    offsets: torch.Tensor | None,
    weights_x: torch.Tensor | None,
    weights_y: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build centered dense kernels from relative sparse offsets."""
    offsets = _require_tensor(offsets)
    weights_x = _require_tensor(weights_x)
    weights_y = _require_tensor(weights_y)
    if offsets.numel() == 0:
        return torch.zeros(1, 1, dtype=weights_x.dtype), torch.zeros(1, 1, dtype=weights_y.dtype)

    min_row = int(offsets[:, 0].min().item())
    max_row = int(offsets[:, 0].max().item())
    min_column = int(offsets[:, 1].min().item())
    max_column = int(offsets[:, 1].max().item())
    top = -min_row
    left = -min_column
    height = max_row - min_row + 1
    width = max_column - min_column + 1
    kernel_x = torch.zeros(height, width, dtype=weights_x.dtype, device=weights_x.device)
    kernel_y = torch.zeros(height, width, dtype=weights_y.dtype, device=weights_y.device)
    for offset, weight_x, weight_y in zip(offsets, weights_x, weights_y, strict=True):
        row = top + int(offset[0].item())
        column = left + int(offset[1].item())
        kernel_x[row, column] = weight_x
        kernel_y[row, column] = weight_y
    return kernel_x, kernel_y


def sparse_padding(offsets: torch.Tensor) -> tuple[int, int, int, int]:
    """Return same-shape padding for relative sparse offsets."""
    if offsets.numel() == 0:
        return (0, 0, 0, 0)
    min_row = int(offsets[:, 0].min().item())
    max_row = int(offsets[:, 0].max().item())
    min_column = int(offsets[:, 1].min().item())
    max_column = int(offsets[:, 1].max().item())
    return (-min_column, max_column, -min_row, max_row)


def box_gradient_dense_kernels(radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense kernels equivalent to the integral box-gradient path."""
    radius = _validate_positive_int(radius, name="radius")
    size = 2 * radius + 1
    scale = 1.0 / float(size * radius * (radius + 1))
    kernel_x = torch.zeros(size, size, dtype=torch.float32)
    kernel_y = torch.zeros(size, size, dtype=torch.float32)
    center = radius
    kernel_x[:, :center] = -scale
    kernel_x[:, center + 1 :] = scale
    kernel_y[:center, :] = -scale
    kernel_y[center + 1 :, :] = scale
    return kernel_x, kernel_y


def anisotropic_gaussian_derivative_kernels(
    angles: torch.Tensor,
    *,
    sigma_parallel: float,
    sigma_perpendicular: float,
    truncate: float,
) -> torch.Tensor:
    """Build rotated anisotropic Gaussian derivative kernels.

    Angles are radians in `[0, pi)`. `theta=0` differentiates in the positive
    column direction, while `theta=pi/2` differentiates in the positive row
    direction.
    """
    _validate_angles(angles)
    radius = max(1, int(math.ceil(truncate * max(sigma_parallel, sigma_perpendicular))))
    coordinates = torch.arange(-radius, radius + 1, dtype=angles.dtype, device=angles.device)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernels: list[torch.Tensor] = []
    for theta in angles:
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        along = columns * cos_theta + rows * sin_theta
        across = -columns * sin_theta + rows * cos_theta
        gaussian = torch.exp(
            -0.5 * ((along / sigma_parallel) ** 2 + (across / sigma_perpendicular) ** 2)
        )
        basis_x = columns * gaussian
        basis_y = rows * gaussian
        moment_matrix = torch.stack(
            (
                torch.stack((torch.sum(basis_x * columns), torch.sum(basis_y * columns))),
                torch.stack((torch.sum(basis_x * rows), torch.sum(basis_y * rows))),
            )
        )
        target_moments = torch.stack((cos_theta, sin_theta))
        coefficients = torch.linalg.solve(moment_matrix, target_moments)
        kernel = coefficients[0] * basis_x + coefficients[1] * basis_y
        kernel = kernel - kernel.mean()
        kernels.append(kernel)
    return torch.stack(kernels, dim=0)


def _infer_legacy_implementation(
    definition: GradientFilterDefinition,
) -> GradientFilterImplementation:
    has_kernel_x = definition.kernel_x is not None
    has_kernel_y = definition.kernel_y is not None
    has_smooth_kernel = definition.smooth_kernel_1d is not None
    has_derivative_kernel = definition.derivative_kernel_1d is not None

    if has_kernel_x != has_kernel_y:
        raise ValueError("kernel_x and kernel_y must be provided together")
    if has_smooth_kernel != has_derivative_kernel:
        raise ValueError("smooth_kernel_1d and derivative_kernel_1d must be provided together")
    if has_kernel_x:
        return GradientFilterImplementation(
            kind=FilterImplementationKind.FIR,
            kernel_x=definition.kernel_x,
            kernel_y=definition.kernel_y,
            smooth_kernel_1d=definition.smooth_kernel_1d,
            derivative_kernel_1d=definition.derivative_kernel_1d,
            spatial_padding=definition.spatial_padding,
        )
    if has_smooth_kernel:
        return GradientFilterImplementation(
            kind=FilterImplementationKind.FIR,
            smooth_kernel_1d=definition.smooth_kernel_1d,
            derivative_kernel_1d=definition.derivative_kernel_1d,
            spatial_padding=definition.spatial_padding,
        )
    raise ValueError("a filter definition must provide an implementation spec")


def _validate_fir_implementation(implementation: GradientFilterImplementation) -> None:
    has_kernel_x = implementation.kernel_x is not None
    has_kernel_y = implementation.kernel_y is not None
    has_smooth_kernel = implementation.smooth_kernel_1d is not None
    has_derivative_kernel = implementation.derivative_kernel_1d is not None
    if has_kernel_x != has_kernel_y:
        raise ValueError("kernel_x and kernel_y must be provided together")
    if has_smooth_kernel != has_derivative_kernel:
        raise ValueError("smooth_kernel_1d and derivative_kernel_1d must be provided together")
    if has_kernel_x:
        _validate_dense_kernel_pair(implementation.kernel_x, implementation.kernel_y)
    if has_smooth_kernel:
        _validate_separable_kernel_pair(
            implementation.smooth_kernel_1d,
            implementation.derivative_kernel_1d,
        )
    if not has_kernel_x and not has_smooth_kernel:
        raise ValueError("FIR implementation requires dense or separable kernels")
    if implementation.spatial_padding is not None:
        _validated_spatial_padding(implementation.spatial_padding)


def _validate_sparse_implementation(implementation: GradientFilterImplementation) -> None:
    offsets = _validate_offset_tensor(implementation.sparse_offsets)
    weights_x = _validate_tensor(implementation.sparse_weights_x, name="weights_x", ndim=1)
    weights_y = _validate_tensor(implementation.sparse_weights_y, name="weights_y", ndim=1)
    if weights_x.shape != weights_y.shape:
        raise ValueError("weights_x and weights_y must have matching shapes")
    if weights_x.shape[0] != offsets.shape[0]:
        raise ValueError("sparse weights must have one entry per offset")
    if weights_x.device != weights_y.device:
        raise ValueError("weights_x and weights_y must be on the same device")


def _validate_orientation_bank_implementation(
    implementation: GradientFilterImplementation,
) -> None:
    kernels = _validate_tensor(
        implementation.orientation_kernels,
        name="orientation_kernels",
        ndim=3,
    )
    angles = _validate_tensor(implementation.angles, name="angles", ndim=1)
    _validate_angles(angles)
    if int(kernels.shape[0]) != int(angles.shape[0]):
        raise ValueError("orientation_kernels must have one kernel per angle")
    if int(kernels.shape[1]) % 2 == 0 or int(kernels.shape[2]) % 2 == 0:
        raise ValueError("orientation kernels must have odd height and width")
    if implementation.spatial_padding is not None:
        _validated_spatial_padding(implementation.spatial_padding)


def _validate_angles(angles: torch.Tensor) -> None:
    if angles.ndim != 1:
        raise ValueError("angles must be 1-D")
    if angles.numel() == 0:
        raise ValueError("angles must not be empty")
    if not bool(torch.isfinite(angles).all().item()):
        raise ValueError("angles must contain only finite values")
    if bool((angles < 0).any().item()) or bool((angles >= math.pi).any().item()):
        raise ValueError("angles must be in [0, pi)")
    if angles.numel() > 1 and not bool((angles[1:] > angles[:-1]).all().item()):
        raise ValueError("angles must be strictly increasing")


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


def _definition_fingerprint(definition: GradientFilterDefinition) -> str:
    implementation = definition.require_implementation()
    hasher = hashlib.sha256()
    payload = {
        "name": definition.name,
        "default_boundary": definition.default_boundary.to_json_dict(),
        "spatial_padding": definition.spatial_padding,
        "support": definition.support,
        "symmetry": definition.symmetry,
        "metadata": dict(sorted(definition.metadata.items())),
        "operator_family": definition.operator_family,
        "linearity": definition.linearity,
        "stage_count": definition.stage_count,
        "support_shape": definition.support_shape,
        "orientation_model": definition.orientation_model,
        "shape_model": definition.shape_model,
        "parameters": _jsonable_mapping(definition.parameters),
        "references": definition.references,
        "supported_boundaries": tuple(mode.value for mode in definition.supported_boundaries),
        "implementation_kind": implementation.kind.value,
        "box_radius": implementation.box_radius,
        "recursive_sigma": implementation.recursive_sigma,
        "recursive_method": implementation.recursive_method,
        "nonlinear_radius": implementation.nonlinear_radius,
        "nonlinear_weighting": implementation.nonlinear_weighting,
        "nonlinear_range_sigma": implementation.nonlinear_range_sigma,
        "nonlinear_robust_scale": implementation.nonlinear_robust_scale,
        "iterative_method": implementation.iterative_method,
        "iterative_iterations": implementation.iterative_iterations,
        "iterative_step_size": implementation.iterative_step_size,
        "iterative_kappa": implementation.iterative_kappa,
        "iterative_conduction": implementation.iterative_conduction,
        "iterative_derivative_radius": implementation.iterative_derivative_radius,
        "riesz_epsilon": implementation.riesz_epsilon,
    }
    hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for label, tensor in (
        ("kernel_x", implementation.kernel_x),
        ("kernel_y", implementation.kernel_y),
        ("smooth_kernel_1d", implementation.smooth_kernel_1d),
        ("derivative_kernel_1d", implementation.derivative_kernel_1d),
        ("sparse_offsets", implementation.sparse_offsets),
        ("sparse_weights_x", implementation.sparse_weights_x),
        ("sparse_weights_y", implementation.sparse_weights_y),
        ("orientation_kernels", implementation.orientation_kernels),
        ("angles", implementation.angles),
    ):
        if tensor is not None:
            _update_tensor_hash(hasher, label, tensor)
    return hasher.hexdigest()


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


def _as_offset_tensor(
    value: TensorLikeOffsets,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int64, device=device).clone().detach()
    return _validate_offset_tensor(tensor)


def _validated_metadata(metadata: Mapping[str, MetadataValue]) -> dict[str, MetadataValue]:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    validated: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if not isinstance(value, bool | int | float | str):
            raise ValueError(f"metadata value for {key!r} must be bool, int, float, or str")
        validated[key] = value
    return validated


def _validated_parameters(
    parameters: Mapping[str, ParameterValue],
) -> dict[str, ParameterValue]:
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    validated: dict[str, ParameterValue] = {}
    for key, value in parameters.items():
        if not isinstance(key, str):
            raise ValueError("parameter keys must be strings")
        validated[key] = _validated_parameter_value(key, value)
    return validated


def _validated_parameter_value(key: str, value: Any) -> ParameterValue:
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str):
        tuple_value = tuple(value)
        if all(isinstance(item, bool | int | float | str) for item in tuple_value):
            return tuple_value
    raise ValueError(f"parameter value for {key!r} must be scalar or scalar tuple")


def _validated_references(references: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(references, tuple):
        references = tuple(references)
    if not all(isinstance(reference, str) and reference for reference in references):
        raise ValueError("references must be non-empty strings")
    return references


def _validated_supported_boundaries(
    supported_boundaries: tuple[BoundaryMode | str, ...],
) -> tuple[BoundaryMode, ...]:
    if not supported_boundaries:
        raise ValueError("supported_boundaries must not be empty")
    return tuple(
        boundary if isinstance(boundary, BoundaryMode) else BoundaryMode(str(boundary))
        for boundary in supported_boundaries
    )


def _jsonable_mapping(parameters: Mapping[str, ParameterValue]) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in sorted(parameters.items())
    }


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


def _validate_offset_tensor(tensor: torch.Tensor | None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("sparse_offsets must be a torch.Tensor")
    if tensor.ndim != 2 or int(tensor.shape[1]) != 2:
        raise ValueError("sparse_offsets must have shape (count, 2)")
    if tensor.numel() == 0:
        raise ValueError("sparse_offsets must not be empty")
    if tensor.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise ValueError("sparse_offsets must use an integer dtype")
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


def _validate_positive_int(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be provided")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _validate_nonnegative_int(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be provided")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_positive_float(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} must be provided")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _require_tensor(tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        raise ValueError("expected tensor value")
    return tensor


def _require_int(value: int | None) -> int:
    if value is None:
        raise ValueError("expected integer value")
    return int(value)


def _require_float(value: float | None) -> float:
    if value is None:
        raise ValueError("expected float value")
    return float(value)
