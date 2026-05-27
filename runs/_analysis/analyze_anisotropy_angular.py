"""Recompute the circular-vs-square anisotropy of the polynomial gradient filter.

This reproduces the Sec. 5 anisotropy experiment analytically from the filter
kernels and emits two quantities per (support, radius):

    eta_minus_1   magnitude anisotropy  max_theta|G| / min_theta|G| - 1
    delta_star    worst-case orientation error in degrees over Theta_36

The smoothed-step stimulus and the Theta_36 = {0, 10, ..., 350} degree
orientation set match the definitions in the paper's theory section. The eta
column reproduces the stored figure CSV (fig_sec05_anisotropy_values.csv) to
five decimals, which validates the delta_star column emitted alongside it.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from agfb_filters.filters.polynomial import build_polynomial_gradient_kernels

# Output lands next to the other Sec. 5 figure CSVs in the paper repo.
OUT_CSV = (
    Path(__file__).resolve().parents[2].parent
    / "PGF_paper"
    / "figures"
    / "cetz_src"
    / "main"
    / "fig_sec05_anisotropy_angular.csv"
)

DEGREE = 3
EDGE_WIDTH = 1.0  # one-pixel smoothed step, matching Sec. 5
RADII = list(range(3, 64))
THETA_36 = [math.radians(10.0 * k) for k in range(36)]


def anisotropy(radius: int, support: str) -> tuple[float, float]:
    """Return (eta_minus_1, delta_star_deg) for one support at one radius."""
    kx, ky = build_polynomial_gradient_kernels(radius=radius, degree=DEGREE, support=support)
    kx = kx.double()
    ky = ky.double()
    axis = torch.arange(-radius, radius + 1).double()
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")  # rows = y, cols = x

    magnitudes: list[float] = []
    angular_errors: list[float] = []
    for theta in THETA_36:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        field = 0.5 * (1.0 + torch.tanh((cols * cos_t + rows * sin_t) / EDGE_WIDTH))
        gx = float(torch.sum(kx * field))
        gy = float(torch.sum(ky * field))
        magnitudes.append(math.hypot(gx, gy))
        wrapped = (math.atan2(gy, gx) - theta + math.pi) % (2.0 * math.pi) - math.pi
        angular_errors.append(abs(math.degrees(wrapped)))

    mags = torch.tensor(magnitudes)
    eta_minus_1 = float(mags.max() / mags.min()) - 1.0
    return eta_minus_1, max(angular_errors)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["support", "radius", "eta_minus_1", "delta_star_deg"])
        for support, label in (("disc", "circle"), ("square", "square")):
            for radius in RADII:
                eta_minus_1, delta_star = anisotropy(radius, support)
                writer.writerow([label, radius, f"{eta_minus_1:.6f}", f"{delta_star:.4f}"])

    # Headline cross-check at the radius the paper quotes.
    for label, support in (("circle", "disc"), ("square", "square")):
        eta_minus_1, delta_star = anisotropy(21, support)
        print(f"r=21 d=3 {label:6s}: eta-1={eta_minus_1:.5f}  delta*={delta_star:.4f} deg")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
