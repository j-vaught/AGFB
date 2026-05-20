"""Explicit-path filter execution."""

from __future__ import annotations

import torch
import torch.fft as torch_fft
import torch.nn.functional as F

from agfb_filters.base import check_input, separable_gradient
from agfb_filters.definitions import GradientFilterDefinition
from agfb_filters.execution import ExecutionPath, ExecutionPlan, InputSignature, concrete_path


def run_filter(
    definition: GradientFilterDefinition,
    image: torch.Tensor,
    *,
    path: ExecutionPath | ExecutionPlan | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a filter definition against an image batch with one explicit path."""
    image = check_input(image)
    return _DEFAULT_RUNNER.run(definition, image, path=path)


class _FilterRunner:
    def __init__(self) -> None:
        self._fft_kernel_cache: dict[tuple[str, str, tuple[int, int], str, str], torch.Tensor] = {}

    def run(
        self,
        definition: GradientFilterDefinition,
        image: torch.Tensor,
        *,
        path: ExecutionPath | ExecutionPlan | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_path = _resolve_path(definition, image, path)

        if selected_path == ExecutionPath.SEPARABLE:
            _require_separable(definition, selected_path)
            smooth_kernel, derivative_kernel = definition.separable_kernels()
            return separable_gradient(
                image,
                smooth_kernel_1d=smooth_kernel.to(device=image.device, dtype=image.dtype),
                derivative_kernel_1d=derivative_kernel.to(device=image.device, dtype=image.dtype),
                pad_mode=definition.padding_mode,
            )

        dense_kernels = _dense_kernels_for_image(definition, image)
        spatial_padding = _spatial_padding(definition, dense_kernels[0])

        if selected_path == ExecutionPath.SPATIAL_DENSE:
            return _spatial_cross_correlation(
                image,
                dense_kernels,
                padding_mode=definition.padding_mode,
                spatial_padding=spatial_padding,
            )
        if selected_path == ExecutionPath.FFT:
            return self._fft_cross_correlation(
                image,
                dense_kernels,
                padding_mode=definition.padding_mode,
                spatial_padding=spatial_padding,
                filter_fingerprint=definition.fingerprint(),
            )
        if selected_path == ExecutionPath.SPARSE_OFFSETS:
            return _offset_cross_correlation(
                image,
                dense_kernels,
                padding_mode=definition.padding_mode,
                spatial_padding=spatial_padding,
            )
        if selected_path == ExecutionPath.ANTIPODAL_PAIRS:
            return _antipodal_cross_correlation(
                image,
                dense_kernels,
                padding_mode=definition.padding_mode,
                spatial_padding=spatial_padding,
            )
        if selected_path == ExecutionPath.STENCIL:
            _require_stencil(dense_kernels)
            return _offset_cross_correlation(
                image,
                dense_kernels,
                padding_mode=definition.padding_mode,
                spatial_padding=spatial_padding,
            )
        raise ValueError(f"unsupported execution path {selected_path}")

    def _fft_cross_correlation(
        self,
        image: torch.Tensor,
        kernels: tuple[torch.Tensor, torch.Tensor],
        *,
        padding_mode: str,
        spatial_padding: tuple[int, int, int, int],
        filter_fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_x, kernel_y = kernels
        kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
        batch, image_height, image_width = image.shape
        padded_image = F.pad(image.unsqueeze(1), spatial_padding, mode=padding_mode).squeeze(1)
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
        filter_fingerprint: str,
    ) -> torch.Tensor:
        cache_key = (
            filter_fingerprint,
            label,
            fft_shape,
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


def _resolve_path(
    definition: GradientFilterDefinition,
    image: torch.Tensor,
    path: ExecutionPath | ExecutionPlan | str,
) -> ExecutionPath:
    if isinstance(path, ExecutionPlan):
        filter_fingerprint = definition.fingerprint()
        if path.filter_fingerprint != filter_fingerprint:
            raise ValueError("execution plan fingerprint does not match filter definition")
        input_signature = InputSignature.from_tensor(image)
        if path.input_signature != input_signature:
            raise ValueError("execution plan input signature does not match image")
    try:
        selected_path = concrete_path(path)
    except ValueError as error:
        raise ValueError(f"unsupported execution path {path!r}") from error
    if selected_path == ExecutionPath.SEPARABLE:
        _require_separable(definition, selected_path)
    else:
        _require_dense(definition, selected_path)
    return selected_path


def _require_separable(definition: GradientFilterDefinition, path: ExecutionPath) -> None:
    if not definition.has_separable_kernels:
        raise ValueError(f"{path.value} path requires separable kernels for {definition.name}")


def _require_dense(definition: GradientFilterDefinition, path: ExecutionPath) -> None:
    if not definition.has_dense_kernels:
        raise ValueError(f"{path.value} path requires dense kernels for {definition.name}")


def _dense_kernels_for_image(
    definition: GradientFilterDefinition,
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x, kernel_y = definition.dense_kernels()
    dense_kernels = (
        kernel_x.to(device=image.device, dtype=image.dtype),
        kernel_y.to(device=image.device, dtype=image.dtype),
    )
    _require_matching_dense_kernels(dense_kernels)
    return dense_kernels


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


def _spatial_cross_correlation(
    image: torch.Tensor,
    kernels: tuple[torch.Tensor, torch.Tensor],
    *,
    padding_mode: str,
    spatial_padding: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
    batch, image_height, image_width = image.shape
    image_channels = image.unsqueeze(1)
    padded_image = F.pad(image_channels, spatial_padding, mode=padding_mode)
    _require_output_shape(
        image_height=image_height,
        image_width=image_width,
        padded_height=int(padded_image.shape[-2]),
        padded_width=int(padded_image.shape[-1]),
        kernel_height=kernel_height,
        kernel_width=kernel_width,
    )
    kernel_stack = torch.stack(kernels, dim=0).unsqueeze(1)
    gradients = F.conv2d(padded_image, kernel_stack)
    return gradients[:batch, 0].contiguous(), gradients[:batch, 1].contiguous()


def _offset_cross_correlation(
    image: torch.Tensor,
    kernels: tuple[torch.Tensor, torch.Tensor],
    *,
    padding_mode: str,
    spatial_padding: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x, kernel_y = kernels
    kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    padded_image = F.pad(image.unsqueeze(1), spatial_padding, mode=padding_mode).squeeze(1)
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
    nonzero_offsets = torch.nonzero((kernel_x != 0) | (kernel_y != 0), as_tuple=False)
    for row_column in nonzero_offsets:
        row = int(row_column[0])
        column = int(row_column[1])
        image_window = padded_image[:, row : row + image_height, column : column + image_width]
        weight_x = kernel_x[row, column]
        weight_y = kernel_y[row, column]
        if weight_x != 0:
            gradient_x = gradient_x + image_window * weight_x
        if weight_y != 0:
            gradient_y = gradient_y + image_window * weight_y
    return gradient_x.contiguous(), gradient_y.contiguous()


def _antipodal_cross_correlation(
    image: torch.Tensor,
    kernels: tuple[torch.Tensor, torch.Tensor],
    *,
    padding_mode: str,
    spatial_padding: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x, kernel_y = kernels
    kernel_height, kernel_width = _require_matching_dense_kernels(kernels)
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("antipodal_pairs path requires odd-sized kernels")
    left, right, top, bottom = spatial_padding
    if left != right or top != bottom:
        raise ValueError("antipodal_pairs path requires symmetric spatial padding")

    pairs_x = _antipodal_pairs(kernel_x)
    pairs_y = _antipodal_pairs(kernel_y)
    image_height = int(image.shape[-2])
    image_width = int(image.shape[-1])
    padded_image = F.pad(image.unsqueeze(1), spatial_padding, mode=padding_mode).squeeze(1)
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


def _antipodal_pairs(kernel: torch.Tensor) -> list[tuple[int, int, torch.Tensor]]:
    flipped = torch.flip(kernel, dims=(0, 1))
    scale = max(float(kernel.abs().max()), 1.0)
    tolerance = 1e-5 * scale
    if not torch.allclose(kernel, -flipped, atol=tolerance, rtol=0.0):
        raise ValueError("antipodal_pairs path requires odd symmetry")

    kernel_height, kernel_width = kernel.shape
    pairs: list[tuple[int, int, torch.Tensor]] = []
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
    pairs: list[tuple[int, int, torch.Tensor]],
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
        output = output + weight * (positive - negative)
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


_DEFAULT_RUNNER = _FilterRunner()
