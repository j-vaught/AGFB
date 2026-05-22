"""Torch-native polynomial gradient kernel construction."""

from __future__ import annotations

from typing import Literal

import torch


def build_polynomial_gradient_kernels(
    radius: int,
    degree: int,
    *,
    support: Literal["disc", "square"],
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build horizontal and vertical least-squares gradient kernels."""
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")

    compute_device = torch.device("cpu") if device is None else device
    offset_values = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float64,
        device=compute_device,
    )
    row_grid, column_grid = torch.meshgrid(offset_values, offset_values, indexing="ij")

    if support == "disc":
        support_mask = row_grid.square() + column_grid.square() <= radius * radius
    elif support == "square":
        support_mask = torch.ones_like(row_grid, dtype=torch.bool)
    else:
        raise ValueError(f"support must be 'disc' or 'square', got {support!r}")

    row_offsets = row_grid[support_mask]
    column_offsets = column_grid[support_mask]
    basis_powers = [
        (row_power, column_power)
        for row_power in range(degree + 1)
        for column_power in range(degree + 1 - row_power)
    ]
    support_count = int(row_offsets.numel())
    basis_count = len(basis_powers)
    if support_count < basis_count:
        raise ValueError(
            f"{support} polynomial fit with radius {radius} and degree {degree} "
            "is underdetermined; "
            f"{support_count} support points cannot fit {basis_count} basis terms"
        )
    row_powers = torch.tensor(
        [row_power for row_power, _ in basis_powers],
        dtype=torch.float64,
        device=compute_device,
    )
    column_powers = torch.tensor(
        [column_power for _, column_power in basis_powers],
        dtype=torch.float64,
        device=compute_device,
    )
    coordinate_scale = float(radius)
    row_fit_offsets = row_offsets / coordinate_scale
    column_fit_offsets = column_offsets / coordinate_scale
    design_matrix = row_fit_offsets[:, None].pow(row_powers) * column_fit_offsets[:, None].pow(
        column_powers
    )
    matrix_rank = int(torch.linalg.matrix_rank(design_matrix).item())
    if matrix_rank < basis_count:
        raise ValueError(
            f"{support} polynomial fit with radius {radius} and degree {degree} is rank deficient; "
            f"rank {matrix_rank} cannot fit {basis_count} basis terms"
        )

    design_matrix_transpose = design_matrix.transpose(0, 1)
    normal_matrix = design_matrix_transpose @ design_matrix
    weights_by_basis = torch.linalg.solve(normal_matrix, design_matrix_transpose)
    column_linear_index = basis_powers.index((0, 1))
    row_linear_index = basis_powers.index((1, 0))

    kernel_size = 2 * radius + 1
    kernel_x = torch.zeros(kernel_size, kernel_size, dtype=torch.float64, device=compute_device)
    kernel_y = torch.zeros(kernel_size, kernel_size, dtype=torch.float64, device=compute_device)
    kernel_x[support_mask] = weights_by_basis[column_linear_index] / coordinate_scale
    kernel_y[support_mask] = weights_by_basis[row_linear_index] / coordinate_scale
    return kernel_x.to(dtype=torch.float32), kernel_y.to(dtype=torch.float32)
