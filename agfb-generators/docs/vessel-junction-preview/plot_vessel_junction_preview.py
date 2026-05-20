from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agfb_generators import vessel_bifurcation, vessel_crossing  # noqa: E402


def frame_panels(frame):
    intensity = frame.I[0].cpu()
    gradient_magnitude = torch.sqrt(frame.gx[0] ** 2 + frame.gy[0] ** 2).cpu()
    gradient_angle = torch.atan2(frame.gy[0], frame.gx[0]).cpu()
    gradient_angle[gradient_magnitude < 0.02 * gradient_magnitude.max()] = torch.nan
    return intensity, gradient_angle, gradient_magnitude


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    crossing = vessel_crossing(
        160,
        160,
        branch_a_width_sigma=5,
        branch_b_width_sigma=4,
        branch_a_normal_angle_rad=math.radians(25),
        branch_b_normal_angle_rad=math.radians(115),
        branch_a_amplitude=0.85,
        branch_b_amplitude=0.9,
        device=device,
    )
    bifurcation = vessel_bifurcation(
        160,
        160,
        trunk_width_sigma=5,
        left_width_sigma=4,
        right_width_sigma=4,
        trunk_tangent_angle_rad=math.radians(-90),
        left_tangent_angle_rad=math.radians(35),
        right_tangent_angle_rad=math.radians(145),
        branch_gate_sigma=4,
        device=device,
    )

intensity_cmap = LinearSegmentedColormap.from_list(
    "intensity",
    ["#000000", "#73000A", "#FFFFFF"],
)
angle_cmap = LinearSegmentedColormap.from_list(
    "angle",
    ["#466A9F", "#FFFFFF", "#CC2E40"],
)
magnitude_cmap = LinearSegmentedColormap.from_list(
    "magnitude",
    ["#000000", "#A49137", "#FFFFFF"],
)
angle_cmap.set_bad("#000000")

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), dpi=180)
for row_index, frame in enumerate((crossing, bifurcation)):
    intensity, gradient_angle, gradient_magnitude = frame_panels(frame)
    axes[row_index, 0].imshow(intensity, cmap=intensity_cmap)
    axes[row_index, 1].imshow(gradient_angle, cmap=angle_cmap, vmin=-math.pi, vmax=math.pi)
    axes[row_index, 2].imshow(gradient_magnitude, cmap=magnitude_cmap)

for axis in axes.ravel():
    axis.axis("off")

plt.subplots_adjust(0, 0, 1, 1, 0, 0)
plt.savefig(
    Path(__file__).with_name("vessel_junction_preview.png"), bbox_inches="tight", pad_inches=0
)
plt.show(block=False)
