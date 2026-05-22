from __future__ import annotations

import math

import pytest
import torch

from agfb_filters import (
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    collapse_orientation_bank,
    define_box_gradient_filter,
    define_nonlinear_window_filter,
    define_orientation_bank_filter,
    get_filter_definition,
    run_filter,
    run_orientation_bank,
)
from agfb_filters.filters.definitions import box_gradient_dense_kernels


def test_sparse_direct_matches_dense_reference() -> None:
    image = torch.randn(2, 10, 11)
    definition = get_filter_definition("sparse_central_difference", radius=2)

    dense_x, dense_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=definition.default_boundary,
    )
    sparse_x, sparse_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SPARSE_OFFSETS,
        boundary=definition.default_boundary,
    )

    assert torch.allclose(dense_x, sparse_x)
    assert torch.allclose(dense_y, sparse_y)


def test_box_integral_matches_dense_reference() -> None:
    image = torch.randn(2, 12, 13)
    definition = get_filter_definition("haar_box_gradient", radius=2)

    dense_x, dense_y = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=definition.default_boundary,
    )
    box_x, box_y = run_filter(
        definition,
        image,
        path=ExecutionPath.BOX_INTEGRAL,
        boundary=definition.default_boundary,
    )

    assert torch.allclose(dense_x, box_x, atol=1e-6)
    assert torch.allclose(dense_y, box_y, atol=1e-6)


def test_box_gradient_dense_kernel_has_expected_ramp_moment() -> None:
    kernel_x, kernel_y = box_gradient_dense_kernels(radius=3)
    coordinates = torch.arange(-3, 4, dtype=kernel_x.dtype)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")

    assert torch.isclose((kernel_x * columns).sum(), torch.tensor(1.0))
    assert torch.isclose((kernel_y * rows).sum(), torch.tensor(1.0))


def test_constants_produce_zero_gradients_for_new_gradient_filters() -> None:
    image = torch.ones(1, 9, 10)
    cases = (
        ("sparse_central_difference", {"radius": 2}, ExecutionPath.SPARSE_OFFSETS),
        ("haar_box_gradient", {"radius": 2}, ExecutionPath.BOX_INTEGRAL),
        ("deriche_recursive_gaussian_derivative", {"sigma": 1.0}, ExecutionPath.RECURSIVE),
        (
            "robust_local_plane_gradient",
            {"radius": 1, "weighting": "tukey"},
            ExecutionPath.NONLINEAR_WINDOW,
        ),
        (
            "perona_malik_gradient",
            {"iterations": 2, "step_size": 0.15, "kappa": 0.2},
            ExecutionPath.ITERATIVE,
        ),
    )

    for name, kwargs, path in cases:
        definition = get_filter_definition(name, **kwargs)
        gradient_x, gradient_y = run_filter(
            definition,
            image,
            path=path,
            boundary=definition.default_boundary,
        )
        assert torch.allclose(gradient_x, torch.zeros_like(image), atol=1e-5)
        assert torch.allclose(gradient_y, torch.zeros_like(image), atol=1e-5)


def test_robust_local_plane_recovers_exact_plane_slopes() -> None:
    rows = torch.arange(9, dtype=torch.float32).view(1, 9, 1)
    columns = torch.arange(10, dtype=torch.float32).view(1, 1, 10)
    image = 2.5 * columns - 1.25 * rows + 3.0
    definition = get_filter_definition("robust_local_plane_gradient", radius=2, weighting="huber")

    gradient_x, gradient_y = run_filter(
        definition,
        image,
        path=ExecutionPath.NONLINEAR_WINDOW,
        boundary=BoundaryCondition(BoundaryMode.CIRCULAR),
    )

    assert torch.allclose(gradient_x[:, 2:-2, 2:-2], torch.full((1, 5, 6), 2.5), atol=1e-4)
    assert torch.allclose(gradient_y[:, 2:-2, 2:-2], torch.full((1, 5, 6), -1.25), atol=1e-4)


def test_iterative_diffusion_matches_slow_reference_loop() -> None:
    image = torch.linspace(0.0, 1.0, steps=30).view(1, 5, 6)
    definition = get_filter_definition(
        "perona_malik_gradient",
        iterations=3,
        step_size=0.12,
        kappa=0.4,
        conduction="reciprocal",
    )

    gradient_x, gradient_y = run_filter(
        definition,
        image,
        path=ExecutionPath.ITERATIVE,
        boundary=definition.default_boundary,
    )
    expected_x, expected_y = _slow_perona_malik_reference(
        image,
        iterations=3,
        step_size=0.12,
        kappa=0.4,
    )

    assert torch.allclose(gradient_x, expected_x)
    assert torch.allclose(gradient_y, expected_y)


def test_orientation_bank_shape_ordering_and_constant_response() -> None:
    definition = get_filter_definition(
        "anisotropic_gaussian_orientation_bank",
        angle_count=4,
        sigma_parallel=1.0,
        sigma_perpendicular=2.0,
    )
    image = torch.ones(2, 9, 10)

    result = run_orientation_bank(
        definition,
        image,
        path=ExecutionPath.ORIENTATION_BANK,
        boundary=definition.default_boundary,
    )

    assert result.responses.shape == (2, 4, 9, 10)
    assert torch.all(result.angles[1:] > result.angles[:-1])
    assert torch.allclose(result.responses, torch.zeros_like(result.responses), atol=1e-5)


def test_orientation_bank_ramp_response_and_collapse_modes() -> None:
    rows = torch.arange(21, dtype=torch.float32).view(1, 21, 1)
    columns = torch.arange(23, dtype=torch.float32).view(1, 1, 23)
    image = 3.0 * columns + 4.0 * rows
    definition = get_filter_definition(
        "anisotropic_gaussian_orientation_bank",
        angle_count=8,
        sigma_parallel=1.0,
        sigma_perpendicular=2.0,
    )

    result = run_orientation_bank(
        definition,
        image,
        path=ExecutionPath.ORIENTATION_BANK,
        boundary=definition.default_boundary,
    )
    interior = result.responses[:, :, 5:-5, 5:-5]
    expected = 3.0 * torch.cos(result.angles) + 4.0 * torch.sin(result.angles)

    assert torch.allclose(interior.mean(dim=(0, 2, 3)), expected, atol=2e-3)

    least_squares = collapse_orientation_bank(result, mode="least_squares_projection")
    assert torch.allclose(
        least_squares.gradient_x[:, 5:-5, 5:-5],
        torch.full((1, 11, 13), 3.0),
        atol=2e-3,
    )
    assert torch.allclose(
        least_squares.gradient_y[:, 5:-5, 5:-5],
        torch.full((1, 11, 13), 4.0),
        atol=2e-3,
    )

    max_projection = collapse_orientation_bank(result, mode="max_projection")
    assert max_projection.response.shape == image.shape
    assert max_projection.angle.shape == image.shape


def test_orientation_bank_api_rejects_wrong_output_usage() -> None:
    bank_definition = get_filter_definition("anisotropic_gaussian_orientation_bank", angle_count=4)
    gradient_definition = get_filter_definition("sobel_3")
    image = torch.randn(1, 8, 9)

    with pytest.raises(ValueError, match="run_orientation_bank"):
        run_filter(
            bank_definition,
            image,
            path=ExecutionPath.ORIENTATION_BANK,
            boundary=bank_definition.default_boundary,
        )
    with pytest.raises(ValueError, match="orientation_bank"):
        run_orientation_bank(
            gradient_definition,
            image,
            path=ExecutionPath.ORIENTATION_BANK,
            boundary=gradient_definition.default_boundary,
        )


def test_spec_validation_rejects_invalid_bank_angles_and_paths() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        define_orientation_bank_filter(
            name="bad_angles",
            angles=[0.0, 0.0],
            sigma_parallel=1.0,
            sigma_perpendicular=2.0,
        )
    with pytest.raises(ValueError, match=r"\[0, pi\)"):
        define_orientation_bank_filter(
            name="bad_angles",
            angles=[math.pi],
            sigma_parallel=1.0,
            sigma_perpendicular=2.0,
        )

    definition = define_box_gradient_filter(name="box", radius=1)
    image = torch.randn(1, 8, 9)
    with pytest.raises(ValueError, match="requires separable kernels"):
        run_filter(
            definition,
            image,
            path=ExecutionPath.SEPARABLE,
            boundary=definition.default_boundary,
        )


def test_nonlinear_validation_rejects_bad_weighting() -> None:
    with pytest.raises(ValueError, match="nonlinear_weighting"):
        define_nonlinear_window_filter(name="bad", radius=1, weighting="bad")


def _slow_perona_malik_reference(
    image: torch.Tensor,
    *,
    iterations: int,
    step_size: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = image.clone()
    for _ in range(iterations):
        padded = torch.nn.functional.pad(
            state.unsqueeze(1), (1, 1, 1, 1), mode="replicate"
        ).squeeze(1)
        north = padded[:, :-2, 1:-1] - state
        south = padded[:, 2:, 1:-1] - state
        west = padded[:, 1:-1, :-2] - state
        east = padded[:, 1:-1, 2:] - state
        update = sum(
            (1.0 / (1.0 + (difference / kappa).square())) * difference
            for difference in (north, south, west, east)
        )
        state = state + step_size * update
    padded = torch.nn.functional.pad(state.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    gradient_x = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) * 0.5
    gradient_y = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) * 0.5
    return gradient_x, gradient_y
