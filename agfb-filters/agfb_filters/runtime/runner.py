"""Explicit-path filter execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.fft as torch_fft
import torch.nn.functional as F

from agfb_filters.filters.definitions import (
    FilterImplementationKind,
    GradientFilterDefinition,
    sparse_padding,
)
from agfb_filters.runtime.execution import (
    BoundaryCondition,
    ExecutionPath,
    concrete_path,
)
from agfb_filters.runtime.tensor_ops import check_input, pad_with_boundary, separable_gradient

_SPARSE_STACK_BYTE_LIMIT = 128 * 1024 * 1024

_KernelCacheKey = tuple[str, str, str]
_BoundaryFftCacheKey = tuple[str, str, tuple[int, int], str, float, str, str]
_SparseOffsetCacheValue = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_AntipodalPair = tuple[int, int, torch.Tensor]
_AntipodalCacheKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class OrientationBankResult:
    """Raw orientation-bank response stack."""

    responses: torch.Tensor
    angles: torch.Tensor
    definition_name: str


@dataclass(frozen=True)
class CollapsedOrientationBank:
    """Projected orientation-bank result."""

    gradient_x: torch.Tensor
    gradient_y: torch.Tensor
    response: torch.Tensor
    angle: torch.Tensor


def run_filter(
    definition: GradientFilterDefinition,
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a gradient filter definition against an image batch with one path."""
    image = check_input(image)
    return _DEFAULT_RUNNER.run(definition, image, path=path, boundary=boundary)


def run_orientation_bank(
    definition: GradientFilterDefinition,
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
) -> OrientationBankResult:
    """Run an orientation-bank filter and return raw directional responses."""
    image = check_input(image)
    return _DEFAULT_RUNNER.run_orientation_bank(
        definition,
        image,
        path=path,
        boundary=boundary,
    )


def collapse_orientation_bank(
    result: OrientationBankResult,
    mode: str = "max_projection",
) -> CollapsedOrientationBank:
    """Collapse raw orientation responses to a gradient-like projection."""
    if not isinstance(result, OrientationBankResult):
        raise ValueError("result must be an OrientationBankResult")
    responses = result.responses
    angles = result.angles.to(device=responses.device, dtype=responses.dtype)
    if responses.ndim != 4:
        raise ValueError("orientation responses must have shape (batch, angles, height, width)")
    if int(responses.shape[1]) != int(angles.numel()):
        raise ValueError("response angle dimension must match angles")

    if mode == "max_projection":
        absolute = responses.abs()
        indices = absolute.argmax(dim=1)
        selected = responses.gather(1, indices.unsqueeze(1)).squeeze(1)
        selected_angles = angles[indices]
        gradient_x = selected * torch.cos(selected_angles)
        gradient_y = selected * torch.sin(selected_angles)
        return CollapsedOrientationBank(
            gradient_x=gradient_x.contiguous(),
            gradient_y=gradient_y.contiguous(),
            response=selected.abs().contiguous(),
            angle=torch.remainder(selected_angles, math.pi).contiguous(),
        )

    if mode == "least_squares_projection":
        design = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        pseudo_inverse = torch.linalg.pinv(design)
        flat = responses.permute(1, 0, 2, 3).reshape(angles.numel(), -1)
        projected = pseudo_inverse @ flat
        gradient_x = projected[0].reshape(
            responses.shape[0], responses.shape[2], responses.shape[3]
        )
        gradient_y = projected[1].reshape(
            responses.shape[0], responses.shape[2], responses.shape[3]
        )
        response = torch.sqrt(gradient_x.square() + gradient_y.square())
        angle = torch.remainder(torch.atan2(gradient_y, gradient_x), math.pi)
        return CollapsedOrientationBank(
            gradient_x=gradient_x.contiguous(),
            gradient_y=gradient_y.contiguous(),
            response=response.contiguous(),
            angle=angle.contiguous(),
        )

    raise ValueError("collapse mode must be max_projection or least_squares_projection")


class _FilterRunner:
    def __init__(self) -> None:
        self._dense_kernel_cache: dict[_KernelCacheKey, tuple[torch.Tensor, torch.Tensor]] = {}
        self._spatial_kernel_stack_cache: dict[_KernelCacheKey, torch.Tensor] = {}
        self._sparse_offset_cache: dict[_KernelCacheKey, _SparseOffsetCacheValue] = {}
        self._antipodal_pair_cache: dict[_AntipodalCacheKey, list[_AntipodalPair]] = {}
        self._fft_kernel_cache: dict[_BoundaryFftCacheKey, torch.Tensor] = {}

    def run(
        self,
        definition: GradientFilterDefinition,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_path, selected_boundary = _resolve_execution(definition, path, boundary)
        implementation = definition.require_implementation()
        if implementation.kind == FilterImplementationKind.ORIENTATION_BANK:
            raise ValueError("orientation-bank definitions must be run with run_orientation_bank")

        if selected_path == ExecutionPath.SEPARABLE:
            _require_separable(definition, selected_path)
            smooth_kernel, derivative_kernel = definition.separable_kernels()
            return separable_gradient(
                image,
                smooth_kernel_1d=smooth_kernel.to(device=image.device, dtype=image.dtype),
                derivative_kernel_1d=derivative_kernel.to(device=image.device, dtype=image.dtype),
                boundary=selected_boundary,
            )

        if selected_path == ExecutionPath.BOX_INTEGRAL:
            _require_kind(definition, FilterImplementationKind.BOX_GRADIENT, selected_path)
            return self._box_integral_gradient(image, definition, boundary=selected_boundary)

        if selected_path == ExecutionPath.RECURSIVE:
            _require_kind(definition, FilterImplementationKind.RECURSIVE, selected_path)
            return _recursive_gaussian_derivative(image, definition, boundary=selected_boundary)

        if selected_path == ExecutionPath.NONLINEAR_WINDOW:
            _require_kind(definition, FilterImplementationKind.NONLINEAR_WINDOW, selected_path)
            return _robust_local_plane_gradient(image, definition, boundary=selected_boundary)

        if selected_path == ExecutionPath.ITERATIVE:
            _require_kind(definition, FilterImplementationKind.ITERATIVE, selected_path)
            return _perona_malik_gradient(image, definition, boundary=selected_boundary)

        if selected_path == ExecutionPath.SPARSE_OFFSETS and (
            implementation.kind == FilterImplementationKind.SPARSE_OFFSET
        ):
            offsets, weights_x, weights_y = definition.sparse_offsets()
            return self._direct_sparse_offset_cross_correlation(
                image,
                offsets.to(device=image.device),
                weights_x.to(device=image.device, dtype=image.dtype),
                weights_y.to(device=image.device, dtype=image.dtype),
                boundary=selected_boundary,
            )

        dense_kernels = self._dense_kernels_for_image(definition, image)
        spatial_padding = _spatial_padding(definition, dense_kernels[0])

        if selected_path == ExecutionPath.SPATIAL_DENSE:
            return self._spatial_cross_correlation(
                image,
                dense_kernels,
                boundary=selected_boundary,
                spatial_padding=spatial_padding,
                filter_fingerprint=definition.fingerprint(),
            )
        if selected_path == ExecutionPath.FFT:
            return self._fft_cross_correlation(
                image,
                dense_kernels,
                boundary=selected_boundary,
                spatial_padding=spatial_padding,
                filter_fingerprint=definition.fingerprint(),
            )
        if selected_path == ExecutionPath.SPARSE_OFFSETS:
            return self._offset_cross_correlation(
                image,
                dense_kernels,
                boundary=selected_boundary,
                spatial_padding=spatial_padding,
                filter_fingerprint=definition.fingerprint(),
            )
        if selected_path == ExecutionPath.ANTIPODAL_PAIRS:
            return self._antipodal_cross_correlation(
                image,
                dense_kernels,
                boundary=selected_boundary,
                spatial_padding=spatial_padding,
                filter_fingerprint=definition.fingerprint(),
            )
        if selected_path == ExecutionPath.STENCIL:
            _require_stencil(dense_kernels)
            return self._spatial_cross_correlation(
                image,
                dense_kernels,
                boundary=selected_boundary,
                spatial_padding=spatial_padding,
                filter_fingerprint=definition.fingerprint(),
            )
        raise ValueError(f"unsupported execution path {selected_path}")

    def run_orientation_bank(
        self,
        definition: GradientFilterDefinition,
        image: torch.Tensor,
        *,
        path: ExecutionPath | str,
        boundary: BoundaryCondition | None,
    ) -> OrientationBankResult:
        selected_path, selected_boundary = _resolve_execution(definition, path, boundary)
        _require_kind(definition, FilterImplementationKind.ORIENTATION_BANK, selected_path)
        if selected_path not in {ExecutionPath.ORIENTATION_BANK, ExecutionPath.SPATIAL_DENSE}:
            raise ValueError("orientation banks support orientation_bank and spatial_dense paths")
        implementation = definition.require_implementation()
        kernels = _require_tensor(implementation.orientation_kernels).to(
            device=image.device,
            dtype=image.dtype,
        )
        angles = _require_tensor(implementation.angles).to(device=image.device, dtype=image.dtype)
        spatial_padding = _orientation_padding(definition, kernels)
        image_channels = image.unsqueeze(1)
        padded_image = pad_with_boundary(image_channels, spatial_padding, selected_boundary)
        gradients = F.conv2d(padded_image, kernels.unsqueeze(1))
        return OrientationBankResult(
            responses=gradients.contiguous(),
            angles=angles.contiguous(),
            definition_name=definition.name,
        )

    def _dense_kernels_for_image(
        self,
        definition: GradientFilterDefinition,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = _kernel_cache_key(
            definition.fingerprint(),
            dtype=image.dtype,
            device=image.device,
        )
        cached = self._dense_kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        kernel_x, kernel_y = definition.dense_kernels()
        dense_kernels = (
            kernel_x.to(device=image.device, dtype=image.dtype),
            kernel_y.to(device=image.device, dtype=image.dtype),
        )
        _require_matching_dense_kernels(dense_kernels)
        self._dense_kernel_cache[cache_key] = dense_kernels
        return dense_kernels

    def _fft_cross_correlation(
        self,
        image: torch.Tensor,
        kernels: tuple[torch.Tensor, torch.Tensor],
        *,
        boundary: BoundaryCondition,
        spatial_padding: tuple[int, int, int, int],
        filter_fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_x, kernel_y = kernels
        kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
        batch, image_height, image_width = image.shape
        padded_image = pad_with_boundary(image.unsqueeze(1), spatial_padding, boundary).squeeze(1)
        padded_height = int(padded_image.shape[-2])
        padded_width = int(padded_image.shape[-1])
        _require_output_shape(
            image_height=image_height,
            image_width=image_width,
            padded_height=padded_height,
            padded_width=padded_width,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
        )

        fft_shape = (padded_height + kernel_height - 1, padded_width + kernel_width - 1)
        image_spectrum = torch_fft.rfft2(padded_image, s=fft_shape)
        outputs = []
        for label, kernel in (("x", kernel_x), ("y", kernel_y)):
            kernel_spectrum = self._kernel_spectrum(
                label=label,
                kernel=kernel,
                fft_shape=fft_shape,
                boundary=boundary,
                filter_fingerprint=filter_fingerprint,
            )
            full_output = torch_fft.irfft2(image_spectrum * kernel_spectrum, s=fft_shape)
            outputs.append(
                full_output[
                    :batch,
                    kernel_height - 1 : kernel_height - 1 + image_height,
                    kernel_width - 1 : kernel_width - 1 + image_width,
                ].contiguous()
            )
        return outputs[0], outputs[1]

    def _kernel_spectrum(
        self,
        *,
        label: str,
        kernel: torch.Tensor,
        fft_shape: tuple[int, int],
        boundary: BoundaryCondition,
        filter_fingerprint: str,
    ) -> torch.Tensor:
        cache_key = (
            filter_fingerprint,
            label,
            fft_shape,
            boundary.mode.value,
            boundary.value,
            str(kernel.dtype),
            str(kernel.device),
        )
        cached = self._fft_kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        padded_kernel = torch.zeros(fft_shape, dtype=kernel.dtype, device=kernel.device)
        kernel_height, kernel_width = kernel.shape
        padded_kernel[:kernel_height, :kernel_width] = torch.flip(kernel, dims=(0, 1))
        spectrum = torch_fft.rfft2(padded_kernel, s=fft_shape)
        self._fft_kernel_cache[cache_key] = spectrum
        return spectrum

    def _spatial_cross_correlation(
        self,
        image: torch.Tensor,
        kernels: tuple[torch.Tensor, torch.Tensor],
        *,
        boundary: BoundaryCondition,
        spatial_padding: tuple[int, int, int, int],
        filter_fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
        batch, image_height, image_width = image.shape
        image_channels = image.unsqueeze(1)
        padded_image = pad_with_boundary(image_channels, spatial_padding, boundary)
        _require_output_shape(
            image_height=image_height,
            image_width=image_width,
            padded_height=int(padded_image.shape[-2]),
            padded_width=int(padded_image.shape[-1]),
            kernel_height=kernel_height,
            kernel_width=kernel_width,
        )
        gradients = F.conv2d(
            padded_image,
            self._spatial_kernel_stack(kernels, filter_fingerprint=filter_fingerprint),
        )
        return gradients[:batch, 0].contiguous(), gradients[:batch, 1].contiguous()

    def _spatial_kernel_stack(
        self,
        kernels: tuple[torch.Tensor, torch.Tensor],
        *,
        filter_fingerprint: str,
    ) -> torch.Tensor:
        kernel_x, _ = kernels
        cache_key = _kernel_cache_key(
            filter_fingerprint,
            dtype=kernel_x.dtype,
            device=kernel_x.device,
        )
        cached = self._spatial_kernel_stack_cache.get(cache_key)
        if cached is not None:
            return cached

        kernel_stack = torch.stack(kernels, dim=0).unsqueeze(1)
        self._spatial_kernel_stack_cache[cache_key] = kernel_stack
        return kernel_stack

    def _direct_sparse_offset_cross_correlation(
        self,
        image: torch.Tensor,
        offsets: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
        *,
        boundary: BoundaryCondition,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_height = int(image.shape[-2])
        image_width = int(image.shape[-1])
        padding = sparse_padding(offsets)
        top = padding[2]
        left = padding[0]
        padded_image = pad_with_boundary(image.unsqueeze(1), padding, boundary).squeeze(1)
        gradient_x = torch.zeros_like(image)
        gradient_y = torch.zeros_like(image)
        for offset, weight_x, weight_y in zip(offsets, weights_x, weights_y, strict=True):
            row_start = top + int(offset[0].item())
            column_start = left + int(offset[1].item())
            image_window = padded_image[
                :,
                row_start : row_start + image_height,
                column_start : column_start + image_width,
            ]
            if weight_x != 0:
                gradient_x.add_(image_window * weight_x)
            if weight_y != 0:
                gradient_y.add_(image_window * weight_y)
        return gradient_x.contiguous(), gradient_y.contiguous()

    def _offset_cross_correlation(
        self,
        image: torch.Tensor,
        kernels: tuple[torch.Tensor, torch.Tensor],
        *,
        boundary: BoundaryCondition,
        spatial_padding: tuple[int, int, int, int],
        filter_fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
        image_height = int(image.shape[-2])
        image_width = int(image.shape[-1])
        padded_image = pad_with_boundary(image.unsqueeze(1), spatial_padding, boundary).squeeze(1)
        _require_output_shape(
            image_height=image_height,
            image_width=image_width,
            padded_height=int(padded_image.shape[-2]),
            padded_width=int(padded_image.shape[-1]),
            kernel_height=kernel_height,
            kernel_width=kernel_width,
        )

        gradient_x = torch.zeros_like(image)
        gradient_y = torch.zeros_like(image)
        nonzero_offsets, weights_x, weights_y = self._sparse_offsets(
            kernels,
            filter_fingerprint=filter_fingerprint,
        )
        if nonzero_offsets.numel() == 0:
            return gradient_x, gradient_y
        if _should_stack_sparse_offsets(nonzero_offsets, image):
            return _stacked_offset_cross_correlation(
                padded_image,
                image,
                nonzero_offsets,
                weights_x,
                weights_y,
            )
        for offset_index, row_column in enumerate(nonzero_offsets):
            row = int(row_column[0])
            column = int(row_column[1])
            image_window = padded_image[:, row : row + image_height, column : column + image_width]
            weight_x = weights_x[offset_index]
            weight_y = weights_y[offset_index]
            if weight_x != 0:
                gradient_x.add_(image_window * weight_x)
            if weight_y != 0:
                gradient_y.add_(image_window * weight_y)
        return gradient_x.contiguous(), gradient_y.contiguous()

    def _sparse_offsets(
        self,
        kernels: tuple[torch.Tensor, torch.Tensor],
        *,
        filter_fingerprint: str,
    ) -> _SparseOffsetCacheValue:
        kernel_x, kernel_y = kernels
        cache_key = _kernel_cache_key(
            filter_fingerprint,
            dtype=kernel_x.dtype,
            device=kernel_x.device,
        )
        cached = self._sparse_offset_cache.get(cache_key)
        if cached is not None:
            return cached

        nonzero_offsets = torch.nonzero((kernel_x != 0) | (kernel_y != 0), as_tuple=False)
        if nonzero_offsets.numel() == 0:
            weights_x = torch.empty(0, dtype=kernel_x.dtype, device=kernel_x.device)
            weights_y = torch.empty(0, dtype=kernel_y.dtype, device=kernel_y.device)
        else:
            rows = nonzero_offsets[:, 0]
            columns = nonzero_offsets[:, 1]
            weights_x = kernel_x[rows, columns]
            weights_y = kernel_y[rows, columns]
        cached_value = (nonzero_offsets, weights_x, weights_y)
        self._sparse_offset_cache[cache_key] = cached_value
        return cached_value

    def _antipodal_cross_correlation(
        self,
        image: torch.Tensor,
        kernels: tuple[torch.Tensor, torch.Tensor],
        *,
        boundary: BoundaryCondition,
        spatial_padding: tuple[int, int, int, int],
        filter_fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_x, kernel_y = kernels
        kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("antipodal_pairs path requires odd-sized kernels")
        left, right, top, bottom = spatial_padding
        if left != right or top != bottom:
            raise ValueError("antipodal_pairs path requires symmetric spatial padding")

        pairs_x = self._antipodal_pairs(
            kernel_x,
            label="x",
            filter_fingerprint=filter_fingerprint,
        )
        pairs_y = self._antipodal_pairs(
            kernel_y,
            label="y",
            filter_fingerprint=filter_fingerprint,
        )
        image_height = int(image.shape[-2])
        image_width = int(image.shape[-1])
        padded_image = pad_with_boundary(image.unsqueeze(1), spatial_padding, boundary).squeeze(1)
        _require_output_shape(
            image_height=image_height,
            image_width=image_width,
            padded_height=int(padded_image.shape[-2]),
            padded_width=int(padded_image.shape[-1]),
            kernel_height=kernel_height,
            kernel_width=kernel_width,
        )
        return (
            _apply_antipodal_pairs(padded_image, image, pairs_x),
            _apply_antipodal_pairs(padded_image, image, pairs_y),
        )

    def _antipodal_pairs(
        self,
        kernel: torch.Tensor,
        *,
        label: str,
        filter_fingerprint: str,
    ) -> list[_AntipodalPair]:
        cache_key = (
            filter_fingerprint,
            label,
            str(kernel.dtype),
            str(kernel.device),
        )
        cached = self._antipodal_pair_cache.get(cache_key)
        if cached is not None:
            return cached

        pairs = _antipodal_pairs(kernel)
        self._antipodal_pair_cache[cache_key] = pairs
        return pairs

    def _box_integral_gradient(
        self,
        image: torch.Tensor,
        definition: GradientFilterDefinition,
        *,
        boundary: BoundaryCondition,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        radius = _require_int(definition.require_implementation().box_radius)
        image_height = int(image.shape[-2])
        image_width = int(image.shape[-1])
        padding = (radius, radius, radius, radius)
        padded = pad_with_boundary(image.unsqueeze(1), padding, boundary).squeeze(1)
        integral = F.pad(padded.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))
        size = 2 * radius + 1
        center = radius
        scale = 1.0 / float(size * radius * (radius + 1))
        left_sum = _sliding_rect_sum(
            integral,
            image_height=image_height,
            image_width=image_width,
            row_start=0,
            row_end=size,
            column_start=0,
            column_end=center,
        )
        right_sum = _sliding_rect_sum(
            integral,
            image_height=image_height,
            image_width=image_width,
            row_start=0,
            row_end=size,
            column_start=center + 1,
            column_end=size,
        )
        top_sum = _sliding_rect_sum(
            integral,
            image_height=image_height,
            image_width=image_width,
            row_start=0,
            row_end=center,
            column_start=0,
            column_end=size,
        )
        bottom_sum = _sliding_rect_sum(
            integral,
            image_height=image_height,
            image_width=image_width,
            row_start=center + 1,
            row_end=size,
            column_start=0,
            column_end=size,
        )
        gradient_x = (right_sum - left_sum) * scale
        gradient_y = (bottom_sum - top_sum) * scale
        return gradient_x.contiguous(), gradient_y.contiguous()


def _resolve_execution(
    definition: GradientFilterDefinition,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None,
) -> tuple[ExecutionPath, BoundaryCondition]:
    if boundary is None:
        raise ValueError("boundary must be provided")
    if not isinstance(boundary, BoundaryCondition):
        raise ValueError("boundary must be a BoundaryCondition")
    if not definition.supports_boundary(boundary):
        supported = ", ".join(mode.value for mode in definition.supported_boundaries)
        raise ValueError(
            f"{definition.name} does not support {boundary.mode.value} boundary; "
            f"supported boundaries are {supported}"
        )
    try:
        selected_path = concrete_path(path)
    except ValueError as error:
        raise ValueError(f"unsupported execution path {path!r}") from error
    return selected_path, boundary


def _require_separable(definition: GradientFilterDefinition, path: ExecutionPath) -> None:
    if not definition.has_separable_kernels:
        raise ValueError(f"{path.value} path requires separable kernels for {definition.name}")


def _require_kind(
    definition: GradientFilterDefinition,
    kind: FilterImplementationKind,
    path: ExecutionPath,
) -> None:
    if definition.require_implementation().kind != kind:
        raise ValueError(f"{path.value} path requires {kind.value} implementation")


def _kernel_cache_key(
    filter_fingerprint: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> _KernelCacheKey:
    return (filter_fingerprint, str(dtype), str(device))


def _require_matching_dense_kernels(kernels: tuple[torch.Tensor, torch.Tensor]) -> tuple[int, int]:
    kernel_x, kernel_y = kernels
    if kernel_x.shape != kernel_y.shape:
        raise ValueError(f"kernel shapes must match, got {kernel_x.shape} vs {kernel_y.shape}")
    if kernel_x.ndim != 2:
        raise ValueError(f"dense kernels must be 2-D, got {kernel_x.ndim} dimensions")
    return int(kernel_x.shape[0]), int(kernel_x.shape[1])


def _spatial_padding(
    definition: GradientFilterDefinition,
    kernel: torch.Tensor,
) -> tuple[int, int, int, int]:
    if definition.spatial_padding is not None:
        return definition.spatial_padding
    kernel_height, kernel_width = kernel.shape
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("even-sized dense kernels require explicit spatial padding")
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    return (padding_width, padding_width, padding_height, padding_height)


def _orientation_padding(
    definition: GradientFilterDefinition,
    kernels: torch.Tensor,
) -> tuple[int, int, int, int]:
    if definition.spatial_padding is not None:
        return definition.spatial_padding
    kernel_height = int(kernels.shape[-2])
    kernel_width = int(kernels.shape[-1])
    return (kernel_width // 2, kernel_width // 2, kernel_height // 2, kernel_height // 2)


def _should_stack_sparse_offsets(
    nonzero_offsets: torch.Tensor,
    image: torch.Tensor,
) -> bool:
    required_bytes = int(nonzero_offsets.shape[0]) * int(image.numel()) * image.element_size()
    return required_bytes <= _SPARSE_STACK_BYTE_LIMIT


def _stacked_offset_cross_correlation(
    padded_image: torch.Tensor,
    image: torch.Tensor,
    nonzero_offsets: torch.Tensor,
    weights_x: torch.Tensor,
    weights_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    windows = torch.stack(
        [
            padded_image[
                :,
                int(row) : int(row) + image_height,
                int(column) : int(column) + image_width,
            ]
            for row, column in nonzero_offsets
        ],
        dim=0,
    )
    gradient_x = torch.einsum("kbhw,k->bhw", windows, weights_x)
    gradient_y = torch.einsum("kbhw,k->bhw", windows, weights_y)
    return gradient_x.contiguous(), gradient_y.contiguous()


def _antipodal_pairs(kernel: torch.Tensor) -> list[_AntipodalPair]:
    flipped = torch.flip(kernel, dims=(0, 1))
    scale = max(float(kernel.abs().max()), 1.0)
    tolerance = 1e-5 * scale
    if not torch.allclose(kernel, -flipped, atol=tolerance, rtol=0.0):
        raise ValueError("antipodal_pairs path requires odd symmetry")

    kernel_height, kernel_width = kernel.shape
    pairs: list[_AntipodalPair] = []
    for row in range(kernel_height):
        for column in range(kernel_width):
            antipodal_row = kernel_height - 1 - row
            antipodal_column = kernel_width - 1 - column
            if (row, column) >= (antipodal_row, antipodal_column):
                continue
            weight = kernel[row, column]
            if torch.abs(weight) > tolerance:
                pairs.append((row, column, weight))
    return pairs


def _apply_antipodal_pairs(
    padded_image: torch.Tensor,
    image: torch.Tensor,
    pairs: list[_AntipodalPair],
) -> torch.Tensor:
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    kernel_height = int(padded_image.shape[-2] - image_height + 1)
    kernel_width = int(padded_image.shape[-1] - image_width + 1)
    output = torch.zeros_like(image)
    for row, column, weight in pairs:
        antipodal_row = kernel_height - 1 - row
        antipodal_column = kernel_width - 1 - column
        positive = padded_image[:, row : row + image_height, column : column + image_width]
        negative = padded_image[
            :,
            antipodal_row : antipodal_row + image_height,
            antipodal_column : antipodal_column + image_width,
        ]
        output.add_((positive - negative) * weight)
    return output.contiguous()


def _require_stencil(kernels: tuple[torch.Tensor, torch.Tensor]) -> None:
    kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
    if kernel_height > 3 or kernel_width > 3:
        raise ValueError("stencil path is limited to 2x2 and 3x3 kernels")


def _require_output_shape(
    *,
    image_height: int,
    image_width: int,
    padded_height: int,
    padded_width: int,
    kernel_height: int,
    kernel_width: int,
) -> None:
    output_height = padded_height - kernel_height + 1
    output_width = padded_width - kernel_width + 1
    if output_height != image_height or output_width != image_width:
        raise ValueError(
            "spatial padding must preserve input shape, "
            f"got output {output_height}x{output_width} for input {image_height}x{image_width}"
        )


def _sliding_rect_sum(
    integral: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
) -> torch.Tensor:
    return (
        integral[:, row_end : row_end + image_height, column_end : column_end + image_width]
        - integral[:, row_start : row_start + image_height, column_end : column_end + image_width]
        - integral[:, row_end : row_end + image_height, column_start : column_start + image_width]
        + integral[
            :,
            row_start : row_start + image_height,
            column_start : column_start + image_width,
        ]
    )


def _recursive_gaussian_derivative(
    image: torch.Tensor,
    definition: GradientFilterDefinition,
    *,
    boundary: BoundaryCondition,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = _require_float(definition.require_implementation().recursive_sigma)
    alpha = math.exp(-math.sqrt(2.0) / sigma)
    smoothed = _recursive_smooth_axis(image, alpha=alpha, axis=-1)
    smoothed = _recursive_smooth_axis(smoothed, alpha=alpha, axis=-2)
    padded = pad_with_boundary(smoothed.unsqueeze(1), (1, 1, 1, 1), boundary).squeeze(1)
    gradient_x = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) * 0.5
    gradient_y = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) * 0.5
    return gradient_x.contiguous(), gradient_y.contiguous()


def _recursive_smooth_axis(
    image: torch.Tensor,
    *,
    alpha: float,
    axis: int,
) -> torch.Tensor:
    moved = image.movedim(axis, -1)
    causal = torch.empty_like(moved)
    anti_causal = torch.empty_like(moved)
    causal[..., 0] = moved[..., 0]
    gain = 1.0 - alpha
    for index in range(1, int(moved.shape[-1])):
        causal[..., index] = gain * moved[..., index] + alpha * causal[..., index - 1]
    anti_causal[..., -1] = causal[..., -1]
    for index in range(int(moved.shape[-1]) - 2, -1, -1):
        anti_causal[..., index] = gain * causal[..., index] + alpha * anti_causal[..., index + 1]
    return anti_causal.movedim(-1, axis)


def _robust_local_plane_gradient(
    image: torch.Tensor,
    definition: GradientFilterDefinition,
    *,
    boundary: BoundaryCondition,
) -> tuple[torch.Tensor, torch.Tensor]:
    implementation = definition.require_implementation()
    radius = _require_int(implementation.nonlinear_radius)
    window_size = 2 * radius + 1
    padding = (radius, radius, radius, radius)
    padded = pad_with_boundary(image.unsqueeze(1), padding, boundary)
    patches = F.unfold(padded, kernel_size=(window_size, window_size))
    batch, _, sample_count = patches.shape
    patches = patches.transpose(1, 2).reshape(batch * sample_count, window_size * window_size)
    coordinates = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    design = torch.stack(
        (columns.reshape(-1), rows.reshape(-1), torch.ones_like(rows).reshape(-1)),
        dim=1,
    )
    initial = torch.linalg.pinv(design) @ patches.transpose(0, 1)
    residuals = patches - (design @ initial).transpose(0, 1)
    weights = _local_plane_weights(
        patches,
        residuals,
        center_index=(window_size * window_size) // 2,
        weighting=str(implementation.nonlinear_weighting),
        range_sigma=_require_float(implementation.nonlinear_range_sigma),
        robust_scale=_require_float(implementation.nonlinear_robust_scale),
    )
    weighted_design = design.unsqueeze(0) * weights.unsqueeze(-1)
    normal_matrix = torch.matmul(design.transpose(0, 1).unsqueeze(0), weighted_design)
    right_hand_side = torch.matmul(
        weighted_design.transpose(1, 2),
        patches.unsqueeze(-1),
    )
    ridge = torch.finfo(image.dtype).eps * max(float(window_size * window_size), 1.0) * 10.0
    eye = torch.eye(3, dtype=image.dtype, device=image.device).unsqueeze(0)
    coefficients = torch.linalg.solve(normal_matrix + ridge * eye, right_hand_side).squeeze(-1)
    gradient_x = coefficients[:, 0].reshape(batch, sample_count)
    gradient_y = coefficients[:, 1].reshape(batch, sample_count)
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    return (
        gradient_x.reshape(batch, height, width).contiguous(),
        gradient_y.reshape(batch, height, width).contiguous(),
    )


def _local_plane_weights(
    patches: torch.Tensor,
    residuals: torch.Tensor,
    *,
    center_index: int,
    weighting: str,
    range_sigma: float,
    robust_scale: float,
) -> torch.Tensor:
    if weighting == "none":
        return torch.ones_like(patches)
    if weighting == "bilateral":
        center = patches[:, center_index].unsqueeze(1)
        return torch.exp(-0.5 * ((patches - center) / range_sigma).square())

    absolute = residuals.abs()
    if weighting == "huber":
        return torch.where(
            absolute <= robust_scale,
            torch.ones_like(absolute),
            robust_scale / absolute.clamp_min(torch.finfo(patches.dtype).eps),
        )
    if weighting == "tukey":
        scaled = absolute / robust_scale
        inside = scaled < 1.0
        weights = (1.0 - scaled.square()).clamp_min(0.0).square()
        return torch.where(inside, weights, torch.zeros_like(weights))
    raise ValueError(f"unsupported nonlinear weighting {weighting!r}")


def _perona_malik_gradient(
    image: torch.Tensor,
    definition: GradientFilterDefinition,
    *,
    boundary: BoundaryCondition,
) -> tuple[torch.Tensor, torch.Tensor]:
    implementation = definition.require_implementation()
    state = image.clone()
    iterations = _require_int(implementation.iterative_iterations)
    step_size = _require_float(implementation.iterative_step_size)
    kappa = _require_float(implementation.iterative_kappa)
    conduction = str(implementation.iterative_conduction)
    for _ in range(iterations):
        padded = pad_with_boundary(state.unsqueeze(1), (1, 1, 1, 1), boundary).squeeze(1)
        north = padded[:, :-2, 1:-1] - state
        south = padded[:, 2:, 1:-1] - state
        west = padded[:, 1:-1, :-2] - state
        east = padded[:, 1:-1, 2:] - state
        update = (
            _perona_malik_conduction(north, kappa=kappa, mode=conduction) * north
            + _perona_malik_conduction(south, kappa=kappa, mode=conduction) * south
            + _perona_malik_conduction(west, kappa=kappa, mode=conduction) * west
            + _perona_malik_conduction(east, kappa=kappa, mode=conduction) * east
        )
        state = state + step_size * update

    radius = _require_int(implementation.iterative_derivative_radius)
    padding = (radius, radius, radius, radius)
    padded = pad_with_boundary(state.unsqueeze(1), padding, boundary).squeeze(1)
    scale = 1.0 / float(2 * radius)
    gradient_x = padded[:, radius:-radius, 2 * radius :] - padded[:, radius:-radius, : -2 * radius]
    gradient_y = padded[:, 2 * radius :, radius:-radius] - padded[:, : -2 * radius, radius:-radius]
    return (gradient_x * scale).contiguous(), (gradient_y * scale).contiguous()


def _perona_malik_conduction(
    difference: torch.Tensor,
    *,
    kappa: float,
    mode: str,
) -> torch.Tensor:
    scaled = difference / kappa
    if mode == "exponential":
        return torch.exp(-scaled.square())
    if mode == "reciprocal":
        return 1.0 / (1.0 + scaled.square())
    raise ValueError(f"unsupported Perona-Malik conduction {mode!r}")


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


_DEFAULT_RUNNER = _FilterRunner()
