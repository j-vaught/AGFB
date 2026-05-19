from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agfb_generators import (  # noqa: E402
    anisotropic_blob,
    asymmetric_ridge,
    chirp,
    curved_arc,
    curved_ridge,
    finite_ramp,
    gabor_packet,
    gaussian_blob,
    gaussian_ridge,
    hard_l_junction,
    hard_step,
    hard_t_junction,
    hard_x_junction,
    hard_y_junction,
    mach_band,
    polynomial,
    roof_profile,
    sinusoid,
    smoothed_bar,
    smoothed_l_junction,
    smoothed_ramp,
    smoothed_step,
    smoothed_t_junction,
    smoothed_x_junction,
    smoothed_y_junction,
    vessel_bifurcation,
    vessel_crossing,
)
from agfb_generators.base import Frame  # noqa: E402

HEIGHT = 193
WIDTH = 193
PLOT_WIDTH = 4.4
PLOT_HEIGHT = 1.35
PLOT_PAD = 0.08


@dataclass(frozen=True)
class GeneratorSpec:
    family: str
    name: str
    title: str
    render: Callable[[], Frame]
    row_offset: int = 0


def main() -> None:
    groups: list[dict[str, object]] = []
    group_lookup: dict[str, list[dict[str, object]]] = {}

    with torch.no_grad():
        for spec in _specs():
            row = HEIGHT // 2 + spec.row_offset
            if row < 0 or row >= HEIGHT:
                raise ValueError(f"{spec.name} requested row {row}, outside height {HEIGHT}")
            frame = spec.render()
            plot = _plot_from_frame(spec, frame, row)
            group_lookup.setdefault(spec.family, []).append(plot)

    for family, plots in group_lookup.items():
        groups.append({"family": family, "plots": plots})

    output = {
        "height": HEIGHT,
        "width": WIDTH,
        "plot_width": PLOT_WIDTH,
        "plot_height": PLOT_HEIGHT,
        "note": (
            "Each panel plots a horizontal image cross section. Intensity and gradient "
            "magnitude are normalized separately within the panel."
        ),
        "groups": groups,
    }

    out_path = Path(__file__).with_name("cross_sections.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def _specs() -> list[GeneratorSpec]:
    return [
        GeneratorSpec(
            family="Polynomial Fields",
            name="polynomial",
            title="Polynomial Surface Generator",
            render=_polynomial_frame,
        ),
        GeneratorSpec(
            family="Edges And Transitions",
            name="smoothed_step",
            title="Smoothed Step Generator",
            render=lambda: smoothed_step(HEIGHT, WIDTH, theta_rad=0.0, sigma_e=5.0),
        ),
        GeneratorSpec(
            family="Edges And Transitions",
            name="hard_step",
            title="Hard Step Generator",
            render=lambda: hard_step(HEIGHT, WIDTH, theta_rad=0.0),
        ),
        GeneratorSpec(
            family="Edges And Transitions",
            name="finite_ramp",
            title="Finite Ramp Generator",
            render=lambda: finite_ramp(HEIGHT, WIDTH, width_px=80.0, theta_rad=0.0),
        ),
        GeneratorSpec(
            family="Edges And Transitions",
            name="smoothed_ramp",
            title="Smoothed Ramp Generator",
            render=lambda: smoothed_ramp(
                HEIGHT,
                WIDTH,
                width_px=80.0,
                theta_rad=0.0,
                sigma_e=5.0,
            ),
        ),
        GeneratorSpec(
            family="Edges And Transitions",
            name="roof_profile",
            title="Roof Profile Generator",
            render=lambda: roof_profile(HEIGHT, WIDTH, width_px=88.0, theta_rad=0.0),
        ),
        GeneratorSpec(
            family="Edges And Transitions",
            name="mach_band",
            title="Mach Band Generator",
            render=lambda: mach_band(
                HEIGHT,
                WIDTH,
                width_px=88.0,
                theta_rad=0.0,
                sigma_e=5.0,
                band_strength=0.12,
                band_sigma=5.0,
            ),
        ),
        GeneratorSpec(
            family="Blobs And Scale",
            name="gaussian_blob",
            title="Gaussian Blob Generator",
            render=lambda: gaussian_blob(HEIGHT, WIDTH, sigma=18.0),
        ),
        GeneratorSpec(
            family="Blobs And Scale",
            name="anisotropic_blob",
            title="Anisotropic Blob Generator",
            render=lambda: anisotropic_blob(
                HEIGHT,
                WIDTH,
                length_sigma=30.0,
                width_sigma=10.0,
                angle_rad=0.0,
            ),
        ),
        GeneratorSpec(
            family="Ridges And Bars",
            name="gaussian_ridge",
            title="Gaussian Ridge Generator",
            render=lambda: gaussian_ridge(HEIGHT, WIDTH, sigma=10.0, theta_rad=0.0),
        ),
        GeneratorSpec(
            family="Ridges And Bars",
            name="smoothed_bar",
            title="Smoothed Bar Generator",
            render=lambda: smoothed_bar(
                HEIGHT,
                WIDTH,
                width_px=50.0,
                theta_rad=0.0,
                sigma_e=5.0,
            ),
        ),
        GeneratorSpec(
            family="Ridges And Bars",
            name="asymmetric_ridge",
            title="Asymmetric Ridge Generator",
            render=lambda: asymmetric_ridge(
                HEIGHT,
                WIDTH,
                negative_sigma=6.0,
                positive_sigma=18.0,
                angle_rad=0.0,
            ),
        ),
        GeneratorSpec(
            family="Ridges And Bars",
            name="curved_ridge",
            title="Curved Ridge Generator",
            render=lambda: curved_ridge(
                HEIGHT,
                WIDTH,
                sigma=6.0,
                theta_rad=0.0,
                curvature=0.004,
            ),
            row_offset=48,
        ),
        GeneratorSpec(
            family="Curved Boundaries",
            name="curved_arc",
            title="Curved Arc Generator",
            render=lambda: curved_arc(HEIGHT, WIDTH, r0=55.0, sigma_e=5.0),
        ),
        GeneratorSpec(
            family="Frequency Fields",
            name="sinusoid",
            title="Sinusoid Generator",
            render=lambda: sinusoid(HEIGHT, WIDTH, freq=0.035, theta_rad=0.0),
        ),
        GeneratorSpec(
            family="Frequency Fields",
            name="chirp",
            title="Chirp Generator",
            render=lambda: chirp(
                HEIGHT,
                WIDTH,
                base_frequency=0.008,
                frequency_slope=0.00035,
                angle_rad=0.0,
            ),
        ),
        GeneratorSpec(
            family="Frequency Fields",
            name="gabor_packet",
            title="Gabor Packet Generator",
            render=lambda: gabor_packet(
                HEIGHT,
                WIDTH,
                freq=0.045,
                theta_rad=0.0,
                sigma_u=45.0,
                sigma_v=18.0,
            ),
        ),
        GeneratorSpec(
            family="Junctions",
            name="smoothed_l_junction",
            title="Smoothed L Junction Generator",
            render=lambda: smoothed_l_junction(
                HEIGHT,
                WIDTH,
                arm_width_px=18.0,
                theta_rad=0.0,
                sigma_e=3.0,
            ),
        ),
        GeneratorSpec(
            family="Junctions",
            name="hard_l_junction",
            title="Hard L Junction Generator",
            render=lambda: hard_l_junction(HEIGHT, WIDTH, arm_width_px=18.0, theta_rad=0.0),
        ),
        GeneratorSpec(
            family="Junctions",
            name="smoothed_t_junction",
            title="Smoothed T Junction Generator",
            render=lambda: smoothed_t_junction(
                HEIGHT,
                WIDTH,
                arm_width_px=18.0,
                theta_rad=0.0,
                sigma_e=3.0,
            ),
        ),
        GeneratorSpec(
            family="Junctions",
            name="hard_t_junction",
            title="Hard T Junction Generator",
            render=lambda: hard_t_junction(HEIGHT, WIDTH, arm_width_px=18.0, theta_rad=0.0),
        ),
        GeneratorSpec(
            family="Junctions",
            name="smoothed_y_junction",
            title="Smoothed Y Junction Generator",
            render=lambda: smoothed_y_junction(
                HEIGHT,
                WIDTH,
                arm_width_px=16.0,
                theta_rad=-math.pi / 2.0,
                sigma_e=3.0,
            ),
        ),
        GeneratorSpec(
            family="Junctions",
            name="hard_y_junction",
            title="Hard Y Junction Generator",
            render=lambda: hard_y_junction(
                HEIGHT,
                WIDTH,
                arm_width_px=16.0,
                theta_rad=-math.pi / 2.0,
            ),
        ),
        GeneratorSpec(
            family="Junctions",
            name="smoothed_x_junction",
            title="Smoothed X Junction Generator",
            render=lambda: smoothed_x_junction(
                HEIGHT,
                WIDTH,
                arm_width_px=15.0,
                theta_rad=math.pi / 4.0,
                sigma_e=3.0,
            ),
        ),
        GeneratorSpec(
            family="Junctions",
            name="hard_x_junction",
            title="Hard X Junction Generator",
            render=lambda: hard_x_junction(
                HEIGHT,
                WIDTH,
                arm_width_px=15.0,
                theta_rad=math.pi / 4.0,
            ),
        ),
        GeneratorSpec(
            family="Vessels",
            name="vessel_crossing",
            title="Vessel Crossing Generator",
            render=lambda: vessel_crossing(
                HEIGHT,
                WIDTH,
                sigma_a=5.0,
                sigma_b=4.0,
                theta_a_rad=math.radians(25.0),
                theta_b_rad=math.radians(115.0),
                contrast_a=0.85,
                contrast_b=0.90,
            ),
        ),
        GeneratorSpec(
            family="Vessels",
            name="vessel_bifurcation",
            title="Vessel Bifurcation Generator",
            render=lambda: vessel_bifurcation(
                HEIGHT,
                WIDTH,
                sigma_trunk=5.0,
                sigma_left=4.0,
                sigma_right=4.0,
                theta_trunk_rad=-math.pi / 2.0,
                theta_left_rad=math.radians(35.0),
                theta_right_rad=math.radians(145.0),
                gate_sigma=4.0,
            ),
        ),
    ]


def _polynomial_frame() -> Frame:
    coeffs = torch.zeros(1, 4, 4)
    coeffs[0, 0, 0] = 0.20
    coeffs[0, 1, 0] = 0.35
    coeffs[0, 0, 1] = -0.25
    coeffs[0, 2, 0] = 0.12
    coeffs[0, 0, 2] = -0.08
    coeffs[0, 1, 1] = 0.18
    coeffs[0, 2, 1] = 0.04
    coeffs[0, 1, 2] = -0.03
    return polynomial(HEIGHT, WIDTH, coeffs=coeffs, scale=64.0)


def _plot_from_frame(spec: GeneratorSpec, frame: Frame, row: int) -> dict[str, object]:
    intensity = frame.I[0, row, :].detach().cpu().float()
    gx = frame.gx[0, row, :].detach().cpu().float()
    gy = frame.gy[0, row, :].detach().cpu().float()
    grad = torch.sqrt(gx * gx + gy * gy)

    i_min = float(intensity.min())
    i_max = float(intensity.max())
    g_max = float(grad.max())
    row_centered = row - HEIGHT // 2

    return {
        "name": spec.name,
        "title": spec.title,
        "caption": (
            f"Horizontal row y = {row_centered:+d} px. "
            f"I range {i_min:.4g} to {i_max:.4g}. "
            f"g max {g_max:.4g}."
        ),
        "intensity": _points(_normalize(intensity)),
        "gradient": _points(_normalize(grad, lower=0.0, upper=g_max)),
    }


def _normalize(
    values: torch.Tensor,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> torch.Tensor:
    lo = float(values.min()) if lower is None else lower
    hi = float(values.max()) if upper is None else upper
    if abs(hi - lo) < 1e-12:
        return torch.full_like(values, PLOT_HEIGHT / 2.0)
    scaled = (values - lo) / (hi - lo)
    return PLOT_PAD + scaled * (PLOT_HEIGHT - 2.0 * PLOT_PAD)


def _points(values: torch.Tensor) -> list[list[float]]:
    n = int(values.numel())
    return [[round(i * PLOT_WIDTH / (n - 1), 5), round(float(values[i]), 5)] for i in range(n)]


if __name__ == "__main__":
    main()
