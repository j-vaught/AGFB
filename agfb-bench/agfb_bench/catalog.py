"""Generator catalog.

A :class:`Cell` is one generator with one fully specified parameter dictionary.
The clean field is deterministic, so a cell's identity is ``(generator, params)``
and a seed only matters once noise is added.

Two rules are fixed for every entry:

1. Angle is the outermost loop, so consecutive cells of a generator differ only
   in orientation and the full angle sweep is contiguous.
2. No generator contributes fewer than 16 cells per seed.

Angle values are degrees, passed to the generators as ``math.radians(deg)``.
Generators are called with ``angle_rad`` already in radians inside ``params``;
``angle_deg`` is retained only for labeling. Vessels carry an angle *pair*, so
their ``angle_deg`` records the primary branch angle for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

# -- Angle sweeps (spec 1.2) --------------------------------------------------
# Mirror-symmetric fields fold to the [0, 45]deg fundamental domain; fields
# without that symmetry sweep [0, 90)deg. The explicit per-family values below
# are the source of truth (taken verbatim from the Chapter 1 tables).
ANGLE_5 = (0.0, 11.25, 22.5, 33.75, 45.0)  # symmetric families, 5-step
ANGLE_4_LOW = (0.0, 22.5, 45.0, 67.5)  # asymmetric ridge, chirp, gabor, L/T/Y
ANGLE_X = (0.0, 11.25, 22.5, 33.75)  # X junctions


@dataclass
class Cell:
    """One generator + parameter dictionary (a clean field, before noise)."""

    generator: str
    structure_class: str
    params: dict[str, Any]
    angle_deg: float | None
    truth_kind: str  # "analytic" | "junction" | "vessel_crossing" | "vessel_bifurcation"
    contrast: float  # signal-power reference for the dB noise convention (Ch. 2)
    cell_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.cell_id:
            self.cell_id = _slug(self.generator, self.params, self.angle_deg)


def _slug(generator: str, params: dict[str, Any], angle_deg: float | None) -> str:
    parts = [generator]
    if angle_deg is not None:
        parts.append(f"a{angle_deg:g}")
    for key, value in params.items():
        if key in ("angle_rad", "device", "dtype"):
            continue
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}=tensor{tuple(value.shape)}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    return "__".join(parts)


def _rad(deg: float) -> float:
    return math.radians(deg)


def _contrast(params: dict[str, Any]) -> float:
    return float(params.get("amplitude", 1.0))


# -- Polynomial coefficient surfaces (spec 1.2 "Surfaces") --------------------
def _polynomial_coefficients() -> list[torch.Tensor | None]:
    """Degree-13 default surface (coefficients=None) + 15 random degree-3
    4x4 coefficient tensors, drawn in order from a single seed-0 generator."""
    generator = torch.Generator().manual_seed(0)
    surfaces: list[torch.Tensor | None] = [None]
    for _ in range(15):
        surfaces.append(0.5 * torch.randn((4, 4), generator=generator))
    return surfaces


def _assert_unique_cell_ids(cells: list[Cell]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for cell in cells:
        if cell.cell_id in seen:
            duplicates.append(cell.cell_id)
        seen.add(cell.cell_id)
    if duplicates:
        sample = ", ".join(duplicates[:5])
        raise AssertionError(f"duplicate catalog cell_id values: {sample}")


def build_catalog() -> list[Cell]:
    """Return the full generator catalog (559 unique cells per seed)."""
    cells: list[Cell] = []

    def add(generator, structure_class, angle_deg, truth_kind="analytic", **params):
        cells.append(
            Cell(
                generator=generator,
                structure_class=structure_class,
                params=params,
                angle_deg=angle_deg,
                truth_kind=truth_kind,
                contrast=_contrast(params),
            )
        )

    # -- Edges & steps --------------------------------------------------------
    for deg in ANGLE_5:
        for amplitude in (0.1, 0.5, 1.0):
            for edge_sigma in (0.5, 1.0, 2.0, 4.0):
                add(
                    "smoothed_step",
                    "edge",
                    deg,
                    angle_rad=_rad(deg),
                    amplitude=amplitude,
                    edge_sigma=edge_sigma,
                )
    for deg in ANGLE_5:
        for amplitude in (0.25, 0.75):
            add(
                "smoothed_step",
                "edge",
                deg,
                angle_rad=_rad(deg),
                amplitude=amplitude,
                edge_sigma=0.5,
            )  # hard-step endpoints not already in the smoothed-step grid

    # -- Ridges & blobs -------------------------------------------------------
    for deg in ANGLE_5:
        for width_sigma in (1.0, 2.0, 4.0, 8.0, 16.0):
            add("gaussian_ridge", "ridge", deg, angle_rad=_rad(deg), width_sigma=width_sigma)
    for deg in ANGLE_4_LOW:
        for neg, pos in ((2, 6), (4, 12), (8, 24), (16, 48)):
            add(
                "asymmetric_ridge",
                "ridge",
                deg,
                angle_rad=_rad(deg),
                negative_sigma=float(neg),
                positive_sigma=float(pos),
            )
    for deg in ANGLE_5:
        for width_sigma in (2.0, 8.0):
            for curvature in (0.002, 0.006):
                add(
                    "curved_ridge",
                    "ridge",
                    deg,
                    angle_rad=_rad(deg),
                    width_sigma=width_sigma,
                    curvature=curvature,
                )
    for scale_sigma in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0):
        for amplitude in (0.5, 1.0):
            add("gaussian_blob", "blob", None, scale_sigma=scale_sigma, amplitude=amplitude)
    for deg in ANGLE_5:
        for length, width in ((16, 4), (32, 8), (64, 16), (128, 32)):
            add(
                "anisotropic_blob",
                "blob",
                deg,
                angle_rad=_rad(deg),
                length_sigma=float(length),
                width_sigma=float(width),
            )

    # -- Bars, ramps & roofs --------------------------------------------------
    for deg in ANGLE_5:
        for bar_width in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
            add(
                "smoothed_bar", "bar", deg, angle_rad=_rad(deg), bar_width=bar_width, edge_sigma=1.0
            )
    for generator in ("smoothed_ramp", "finite_ramp", "roof_profile", "mach_band"):
        width_key = {
            "smoothed_ramp": "ramp_width",
            "finite_ramp": "ramp_width",
            "roof_profile": "roof_width",
            "mach_band": "ramp_width",
        }[generator]
        klass = "roof" if generator == "roof_profile" else "ramp"
        for deg in ANGLE_5:
            for width in (16.0, 64.0, 256.0, 1024.0, 4096.0):
                add(generator, klass, deg, angle_rad=_rad(deg), **{width_key: width})

    # -- Curved boundary ------------------------------------------------------
    for radius in (64.0, 256.0, 1024.0, 4096.0):
        for amplitude in (0.5, 1.0):
            for edge_sigma in (0.5, 1.0, 2.0):
                # center_y = -radius so the boundary passes through frame center.
                add(
                    "curved_arc",
                    "curved",
                    None,
                    radius=radius,
                    center_y=-radius,
                    amplitude=amplitude,
                    edge_sigma=edge_sigma,
                )

    # -- Oscillatory ----------------------------------------------------------
    for deg in ANGLE_5:
        for freq in (0.005, 0.02, 0.05, 0.1, 0.2, 0.4):
            add("sinusoid", "oscillatory", deg, angle_rad=_rad(deg), spatial_frequency=freq)
    for deg in ANGLE_4_LOW:
        for base_freq in (0.01, 0.02):
            for slope in (5e-5, 1e-4):
                add(
                    "chirp",
                    "oscillatory",
                    deg,
                    angle_rad=_rad(deg),
                    base_frequency=base_freq,
                    frequency_slope=slope,
                )
    for deg in ANGLE_4_LOW:
        for carrier in (0.02, 0.05):
            for env_len, env_wid in ((64, 32), (128, 64)):
                add(
                    "gabor_packet",
                    "oscillatory",
                    deg,
                    angle_rad=_rad(deg),
                    carrier_frequency=carrier,
                    envelope_length_sigma=float(env_len),
                    envelope_width_sigma=float(env_wid),
                )

    # -- Surfaces -------------------------------------------------------------
    for index, coefficients in enumerate(_polynomial_coefficients()):
        params: dict[str, Any] = {}
        if coefficients is not None:
            params["coefficients"] = coefficients
        cells.append(
            Cell(
                "polynomial",
                "surface",
                params,
                None,
                "analytic",
                1.0,
                cell_id=f"polynomial__i{index:02d}",
            )
        )

    # -- Junctions ------------------------------------------------------------
    junctions = (
        ("smoothed_l_junction", ANGLE_4_LOW),
        ("hard_l_junction", ANGLE_4_LOW),
        ("smoothed_t_junction", ANGLE_4_LOW),
        ("hard_t_junction", ANGLE_4_LOW),
        ("smoothed_x_junction", ANGLE_X),
        ("hard_x_junction", ANGLE_X),
        ("smoothed_y_junction", ANGLE_4_LOW),
        ("hard_y_junction", ANGLE_4_LOW),
    )
    for generator, angles in junctions:
        for deg in angles:
            for arm_width in (8.0, 16.0, 32.0, 64.0):
                add(
                    generator,
                    "junction",
                    deg,
                    truth_kind="junction",
                    angle_rad=_rad(deg),
                    arm_width=arm_width,
                )

    # -- Vessels --------------------------------------------------------------
    # vessel_crossing: (branch-a, branch-b normal angles) x (branch widths).
    for a_deg, b_deg in ((25, 115), (40, 100), (10, 90), (45, 135)):
        for a_w, b_w in ((5, 4), (8, 6), (6, 5), (10, 8)):
            cells.append(
                Cell(
                    "vessel_crossing",
                    "vessel",
                    dict(
                        branch_a_normal_angle_rad=_rad(a_deg),
                        branch_b_normal_angle_rad=_rad(b_deg),
                        branch_a_width_sigma=float(a_w),
                        branch_b_width_sigma=float(b_w),
                    ),
                    float(a_deg),
                    "vessel_crossing",
                    1.0,
                )
            )
    # vessel_bifurcation: (left, right tangent angles) x (trunk/branch widths).
    for left_deg, right_deg in ((35, 145), (20, 160), (50, 130), (35, 160)):
        for trunk, left, right, gate in ((5, 4, 4, 4), (8, 6, 6, 4), (6, 5, 5, 3), (10, 8, 8, 5)):
            cells.append(
                Cell(
                    "vessel_bifurcation",
                    "vessel",
                    dict(
                        left_tangent_angle_rad=_rad(left_deg),
                        right_tangent_angle_rad=_rad(right_deg),
                        trunk_width_sigma=float(trunk),
                        left_width_sigma=float(left),
                        right_width_sigma=float(right),
                        branch_gate_sigma=float(gate),
                    ),
                    float(left_deg),
                    "vessel_bifurcation",
                    1.0,
                )
            )

    _assert_unique_cell_ids(cells)
    return cells


# -- Canonical 24-cell subset (spec 1.3) --------------------------------------
def build_canonical_subset() -> list[Cell]:
    """24 representative cells for the noise-breadth (C) and timing (D) studies."""
    cells: list[Cell] = []

    def add(generator, structure_class, angle_deg, truth_kind="analytic", **params):
        cells.append(
            Cell(generator, structure_class, params, angle_deg, truth_kind, _contrast(params))
        )

    add("smoothed_step", "edge", 45.0, angle_rad=_rad(45), amplitude=1.0, edge_sigma=2.0)
    add("smoothed_step", "edge", 45.0, angle_rad=_rad(45), amplitude=1.0, edge_sigma=0.5)
    add("gaussian_ridge", "ridge", 45.0, angle_rad=_rad(45), width_sigma=2.0)
    add("gaussian_ridge", "ridge", 45.0, angle_rad=_rad(45), width_sigma=8.0)
    add("smoothed_bar", "bar", 45.0, angle_rad=_rad(45), bar_width=16.0, edge_sigma=1.0)
    add("curved_arc", "curved", None, radius=128.0, center_y=-128.0, amplitude=1.0, edge_sigma=2.0)
    add("gaussian_blob", "blob", None, scale_sigma=16.0, amplitude=1.0)
    add("anisotropic_blob", "blob", 45.0, angle_rad=_rad(45), length_sigma=64.0, width_sigma=16.0)
    add("sinusoid", "oscillatory", 45.0, angle_rad=_rad(45), spatial_frequency=0.1)
    add("sinusoid", "oscillatory", 45.0, angle_rad=_rad(45), spatial_frequency=0.3)
    add("chirp", "oscillatory", 45.0, angle_rad=_rad(45), base_frequency=0.02, frequency_slope=1e-4)
    add(
        "gabor_packet",
        "oscillatory",
        45.0,
        angle_rad=_rad(45),
        carrier_frequency=0.05,
        envelope_length_sigma=128.0,
        envelope_width_sigma=64.0,
    )
    add("roof_profile", "roof", 45.0, angle_rad=_rad(45), roof_width=64.0)
    add("mach_band", "ramp", 45.0, angle_rad=_rad(45), ramp_width=128.0)
    add("finite_ramp", "ramp", 45.0, angle_rad=_rad(45), ramp_width=64.0)
    add("smoothed_ramp", "ramp", 45.0, angle_rad=_rad(45), ramp_width=64.0)
    add(
        "asymmetric_ridge",
        "ridge",
        45.0,
        angle_rad=_rad(45),
        negative_sigma=8.0,
        positive_sigma=24.0,
    )
    add("curved_ridge", "ridge", 45.0, angle_rad=_rad(45), width_sigma=4.0, curvature=0.006)
    add(
        "smoothed_l_junction",
        "junction",
        45.0,
        truth_kind="junction",
        angle_rad=_rad(45),
        arm_width=16.0,
    )
    add(
        "smoothed_t_junction",
        "junction",
        45.0,
        truth_kind="junction",
        angle_rad=_rad(45),
        arm_width=18.0,
    )
    add(
        "smoothed_x_junction",
        "junction",
        0.0,
        truth_kind="junction",
        angle_rad=_rad(0),
        arm_width=18.0,
    )
    add(
        "smoothed_y_junction",
        "junction",
        45.0,
        truth_kind="junction",
        angle_rad=_rad(45),
        arm_width=18.0,
    )
    cells.append(Cell("vessel_crossing", "vessel", {}, None, "vessel_crossing", 1.0))
    cells.append(Cell("vessel_bifurcation", "vessel", {}, None, "vessel_bifurcation", 1.0))
    return cells
