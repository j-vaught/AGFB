from __future__ import annotations

import pytest
import torch

from agfb_filters import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    get_filter_definition,
    run_filter,
    shipped_filter_specs,
)


def _step_image() -> tuple[torch.Tensor, int]:
    image_height = 106
    image_width = 160
    edge_column = image_width // 2
    image = torch.zeros(1, image_height, image_width)
    image[:, :, edge_column:] = 1.0
    return image, edge_column


def _gradient_specs():
    return [spec for spec in shipped_filter_specs() if spec.output_api == "gradient"]


def _dense_like_paths(definition) -> tuple[ExecutionPath, ...]:
    paths = [ExecutionPath.SPATIAL_DENSE, ExecutionPath.FFT, ExecutionPath.SPARSE_OFFSETS]
    kernel_x, kernel_y = definition.dense_kernels()
    if max(kernel_x.shape) <= 3:
        paths.append(ExecutionPath.STENCIL)
    if _is_antipodal(kernel_x) and _is_antipodal(kernel_y):
        paths.append(ExecutionPath.ANTIPODAL_PAIRS)
    return tuple(paths)


def test_runner_detects_vertical_step_edge_on_160_by_106_image() -> None:
    image, edge_column = _step_image()

    for spec in _gradient_specs():
        definition = get_filter_definition(spec.name, **dict(spec.smoke_kwargs))
        gradient_x, gradient_y = run_filter(
            definition,
            image,
            path=ExecutionPath(spec.smoke_path),
            boundary=definition.default_boundary,
        )
        horizontal_response = gradient_x.abs().mean(dim=(0, 1))
        peak_column = int(horizontal_response.argmax())

        assert gradient_x.shape == image.shape
        assert gradient_y.shape == image.shape
        assert torch.isfinite(gradient_x).all()
        assert torch.isfinite(gradient_y).all()
        assert peak_column in {edge_column - 1, edge_column}
        assert float(horizontal_response.max()) > 0.01
        assert float(gradient_y.abs().max()) < 1e-4


def test_fir_dense_like_paths_match_spatial_dense_on_random_images() -> None:
    image = torch.randn(1, 18, 19)

    for spec in _gradient_specs():
        definition = get_filter_definition(spec.name, **dict(spec.smoke_kwargs))
        if not definition.has_dense_kernels:
            continue
        reference_x, reference_y = run_filter(
            definition,
            image,
            path=ExecutionPath.SPATIAL_DENSE,
            boundary=definition.default_boundary,
        )
        for path in _dense_like_paths(definition):
            try:
                gradient_x, gradient_y = run_filter(
                    definition,
                    image,
                    path=path,
                    boundary=definition.default_boundary,
                )
            except ValueError as error:
                if path == ExecutionPath.STENCIL:
                    assert "stencil" in str(error)
                    continue
                raise
            assert torch.allclose(reference_x, gradient_x, atol=1e-4)
            assert torch.allclose(reference_y, gradient_y, atol=1e-4)


def test_recursive_filter_rejects_unsupported_boundary() -> None:
    definition = get_filter_definition("deriche_recursive_gaussian_derivative", sigma=1.0)
    image = torch.randn(1, 8, 9)

    with pytest.raises(ValueError, match="does not support reflect boundary"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.RECURSIVE,
            boundary=BoundaryCondition(BoundaryMode.REFLECT),
        )


def _is_antipodal(kernel: torch.Tensor) -> bool:
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        return False
    scale = max(float(kernel.abs().max()), 1.0)
    return bool(
        torch.allclose(
            kernel,
            -torch.flip(kernel, dims=(0, 1)),
            atol=1e-5 * scale,
            rtol=0.0,
        )
    )
