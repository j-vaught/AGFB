from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agfb_generators import curved_arc  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    frame = curved_arc(
        160,
        160,
        radius=92,
        center_x=-60,
        center_y=18,
        edge_sigma=4,
        amplitude=1,
        device=device,
    )
    intensity = frame.I[0].cpu()
    gradient_magnitude = torch.sqrt(frame.gx[0] ** 2 + frame.gy[0] ** 2).cpu()
    gradient_angle = torch.atan2(frame.gy[0], frame.gx[0]).cpu()

gradient_angle[gradient_magnitude < 0.02 * gradient_magnitude.max()] = torch.nan

intensity_cmap = LinearSegmentedColormap.from_list("intensity", ["#000000", "#73000A", "#FFFFFF"])
angle_cmap = LinearSegmentedColormap.from_list("angle", ["#466A9F", "#FFFFFF", "#CC2E40"])
magnitude_cmap = LinearSegmentedColormap.from_list("magnitude", ["#000000", "#A49137", "#FFFFFF"])
angle_cmap.set_bad("#000000")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), dpi=180)
axes[0].imshow(intensity, cmap=intensity_cmap)
axes[1].imshow(gradient_angle, cmap=angle_cmap, vmin=-torch.pi, vmax=torch.pi)
axes[2].imshow(gradient_magnitude, cmap=magnitude_cmap)

for axis in axes:
    axis.axis("off")

plt.subplots_adjust(0, 0, 1, 1, 0, 0)
plt.savefig(Path(__file__).with_name("curved_arc_preview.png"), bbox_inches="tight", pad_inches=0)
plt.show(block=False)
