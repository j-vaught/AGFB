from __future__ import annotations

import torch

from agfb_filters import (
    AutoRunner,
    ExecutionPath,
    central_difference_definition,
    cpgf_definition,
    derivative_of_gaussian_definition,
    farid_simoncelli_5_definition,
    freeman_adelson_g1_definition,
    prewitt_3_definition,
    roberts_definition,
    run_filter,
    savitzky_golay_definition,
    scharr_3_definition,
    sobel_definition,
)


def _step_image() -> tuple[torch.Tensor, int]:
    image_height = 106
    image_width = 160
    edge_column = image_width // 2
    image = torch.zeros(1, image_height, image_width)
    image[:, :, edge_column:] = 1.0
    return image, edge_column


def _definitions():
    return [
        central_difference_definition(),
        farid_simoncelli_5_definition(),
        prewitt_3_definition(),
        roberts_definition(),
        scharr_3_definition(),
        sobel_definition(3),
        sobel_definition(5),
        sobel_definition(7),
        cpgf_definition(radius=2, degree=2),
        derivative_of_gaussian_definition(sigma=1.0),
        freeman_adelson_g1_definition(sigma=1.0),
        savitzky_golay_definition(radius=2, degree=2),
    ]


def test_runner_detects_vertical_step_edge_on_160_by_106_image() -> None:
    image, edge_column = _step_image()

    for definition in _definitions():
        gradient_x, gradient_y = run_filter(definition, image, path=ExecutionPath.SPATIAL_DENSE)
        horizontal_response = gradient_x.abs().mean(dim=(0, 1))
        peak_column = int(horizontal_response.argmax())

        assert gradient_x.shape == image.shape
        assert gradient_y.shape == image.shape
        assert torch.isfinite(gradient_x).all()
        assert torch.isfinite(gradient_y).all()
        assert peak_column in {edge_column - 1, edge_column}
        assert float(horizontal_response.max()) > 0.05
        assert float(gradient_y.abs().max()) < 1e-5


def test_valid_paths_match_spatial_dense_on_random_images() -> None:
    image = torch.randn(1, 18, 19)
    auto_runner = AutoRunner()

    for definition in _definitions():
        reference_x, reference_y = run_filter(definition, image, path=ExecutionPath.SPATIAL_DENSE)
        for path in auto_runner.valid_paths(definition):
            gradient_x, gradient_y = run_filter(definition, image, path=path)
            assert torch.allclose(reference_x, gradient_x, atol=1e-4)
            assert torch.allclose(reference_y, gradient_y, atol=1e-4)


def test_valid_paths_match_spatial_dense_on_step_image() -> None:
    image, _ = _step_image()
    auto_runner = AutoRunner()

    for definition in _definitions():
        reference_x, reference_y = run_filter(definition, image, path=ExecutionPath.SPATIAL_DENSE)
        for path in auto_runner.valid_paths(definition):
            gradient_x, gradient_y = run_filter(definition, image, path=path)
            assert torch.allclose(reference_x, gradient_x, atol=1e-4)
            assert torch.allclose(reference_y, gradient_y, atol=1e-4)
