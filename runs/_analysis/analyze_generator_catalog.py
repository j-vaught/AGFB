"""Synthetic image-generator catalog table for the benchmark appendix.

Emits a reproducibility table listing every synthetic image generator in the
AGFB catalog with its structure class, orientation sweep, parameter grid, and
the number of cells it contributes. The class, orientation set, and cell counts
are read directly from `build_catalog()` so they cannot drift from the code that
the benchmark actually runs; only the human-readable parameter-grid text is
authored here. Run from anywhere -- paths are resolved relative to this file.
"""

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]  # runs/_analysis -> runs -> AGFB
OUT = HERE.parents[2] / "PGF_paper" / "figures" / "tables"  # repo-root/PGF_paper/figures/tables
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORKSPACE / "agfb-bench"))

from agfb_bench.catalog import build_catalog  # noqa: E402

# Display order of structure classes, conceptual rather than alphabetical.
CLASS_ORDER = [
    "edge",
    "ridge",
    "blob",
    "bar",
    "ramp",
    "roof",
    "curved",
    "oscillatory",
    "surface",
    "junction",
    "vessel",
]

# Known orientation sweeps, matched against the realized angle set per generator.
ANGLE_SETS = {
    (0.0, 11.25, 22.5, 33.75, 45.0): "0, 11.25, 22.5, 33.75, 45",
    (0.0, 22.5, 45.0, 67.5): "0, 22.5, 45, 67.5",
    (0.0, 11.25, 22.5, 33.75): "0, 11.25, 22.5, 33.75",
}

# Hand-authored parameter-grid descriptions, keyed by generator. Values verified
# against build_catalog(); Typst math markup is used for symbols (e.g. $sigma$).
# Lengths are in pixels; frequencies in cycles per pixel.
PARAMS = {
    "smoothed_step": "amplitude 0.1, 0.25, 0.5, 0.75, 1.0; edge $sigma$ 0.5, 1, 2, 4",
    "gaussian_ridge": "width $sigma$ 1, 2, 4, 8, 16",
    "asymmetric_ridge": "(neg., pos.) $sigma$ (2, 6), (4, 12), (8, 24), (16, 48)",
    "curved_ridge": "width $sigma$ 2, 8; curvature 0.002, 0.006",
    "gaussian_blob": "scale $sigma$ 1, 2, 4, 8, 16, 32, 48, 64; amplitude 0.5, 1.0",
    "anisotropic_blob": "(length, width) $sigma$ (16, 4), (32, 8), (64, 16), (128, 32)",
    "smoothed_bar": "bar width 4, 8, 16, 32, 64, 128; edge $sigma$ 1",
    "smoothed_ramp": "ramp width 16, 64, 256, 1024, 4096",
    "finite_ramp": "ramp width 16, 64, 256, 1024, 4096",
    "mach_band": "ramp width 16, 64, 256, 1024, 4096",
    "roof_profile": "roof width 16, 64, 256, 1024, 4096",
    "curved_arc": "radius 64, 256, 1024, 4096; amplitude 0.5, 1.0; edge $sigma$ 0.5, 1, 2",
    "sinusoid": "frequency 0.005, 0.02, 0.05, 0.1, 0.2, 0.4",
    "chirp": "base freq. 0.01, 0.02; freq. slope $5 times 10^(-5)$, $10^(-4)$",
    "gabor_packet": (
        "carrier freq. 0.02, 0.05; envelope (length, width) $sigma$ (64, 32), (128, 64)"
    ),
    "polynomial": (
        "one degree-13 reference field; 15 random degree-3 $4 times 4$ "
        "coefficient tensors (torch seed 0)"
    ),
    "smoothed_l_junction": "arm width 8, 16, 32, 64",
    "hard_l_junction": "arm width 8, 16, 32, 64",
    "smoothed_t_junction": "arm width 8, 16, 32, 64",
    "hard_t_junction": "arm width 8, 16, 32, 64",
    "smoothed_x_junction": "arm width 8, 16, 32, 64",
    "hard_x_junction": "arm width 8, 16, 32, 64",
    "smoothed_y_junction": "arm width 8, 16, 32, 64",
    "hard_y_junction": "arm width 8, 16, 32, 64",
    "vessel_crossing": (
        "branch-angle pairs (deg) (25, 115), (40, 100), (10, 90), (45, 135); "
        "width $sigma$ pairs (5, 4), (8, 6), (6, 5), (10, 8)"
    ),
    "vessel_bifurcation": (
        "tangent-angle pairs (deg) (35, 145), (20, 160), (50, 130), (35, 160); "
        "(trunk, left, right, gate) $sigma$ (5, 4, 4, 4), (8, 6, 6, 4), (6, 5, 5, 3), (10, 8, 8, 5)"
    ),
}


# Generators whose orientation is encoded in the per-branch geometry rather than
# a shared orientation sweep; the angles live in their parameter grid instead.
GEOMETRY_ORIENTED = {"vessel_crossing", "vessel_bifurcation"}


def angles_label(gen, cells) -> str:
    if gen in GEOMETRY_ORIENTED:
        return "--"
    seen = sorted({c.angle_deg for c in cells if c.angle_deg is not None})
    if not seen:
        return "--"
    key = tuple(seen)
    if key in ANGLE_SETS:
        return ANGLE_SETS[key]
    return ", ".join(f"{a:g}" for a in seen)


cells = build_catalog()
groups = defaultdict(list)
order = []
for c in cells:
    if c.generator not in groups:
        order.append(c.generator)
    groups[c.generator].append(c)

missing = [g for g in order if g not in PARAMS]
assert not missing, f"no parameter description for: {missing}"

rows = []
for gen in sorted(order, key=lambda g: (CLASS_ORDER.index(groups[g][0].structure_class), g)):
    cs = groups[gen]
    rows.append(
        [
            cs[0].structure_class.capitalize(),
            f"`{gen}`",
            angles_label(gen, cs),
            PARAMS[gen],
            str(len(cs)),
        ]
    )

total = sum(len(groups[g]) for g in order)
assert total == len(cells), f"cell count mismatch: {total} != {len(cells)}"

header = ["Class", "Generator", "Orientations (deg)", "Parameter grid", "Cells"]
import csv  # noqa: E402

path = OUT / "appendix_generator_catalog.csv"
with path.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    w.writerows(rows)
print(f"wrote {path.name} ({len(rows)} generators, {total} cells)")
