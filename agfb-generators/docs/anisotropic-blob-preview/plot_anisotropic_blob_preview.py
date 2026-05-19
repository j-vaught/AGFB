from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agfb_generators import anisotropic_blob  # noqa: E402

BRAND = {
    "garnet": "#73000A",
    "black": "#000000",
    "white": "#FFFFFF",
    "gray90": "#363636",
    "gray70": "#5C5C5C",
    "gray30": "#C7C7C7",
    "honeycomb": "#A49137",
}


@dataclass(frozen=True)
class PreviewSpec:
    group: str
    label: str
    angle_deg: float
    length_sigma: float
    width_sigma: float
    amplitude: float


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)

    specs = _preview_specs()
    frame = anisotropic_blob(
        args.height,
        args.width,
        length_sigma=torch.tensor([spec.length_sigma for spec in specs], device=device),
        width_sigma=torch.tensor([spec.width_sigma for spec in specs], device=device),
        angle_rad=torch.tensor([math.radians(spec.angle_deg) for spec in specs], device=device),
        amplitude=torch.tensor([spec.amplitude for spec in specs], device=device),
        device=device,
    )

    intensity = frame.I.detach().cpu()
    gradient_magnitude = torch.sqrt(frame.gx * frame.gx + frame.gy * frame.gy).detach().cpu()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _plot_preview(
        specs=specs,
        intensity=intensity,
        gradient_magnitude=gradient_magnitude,
        output=output,
    )

    print(f"saved {output}")
    print(f"device {device}")
    print(f"intensity max {float(intensity.max()):.6g}")
    print(f"gradient magnitude max {float(gradient_magnitude.max()):.6g}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot anisotropic_blob angle and magnitude previews."
    )
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use cuda when available by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("anisotropic_blob_preview.png"),
    )
    return parser.parse_args()


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _preview_specs() -> list[PreviewSpec]:
    return [
        PreviewSpec("Angle", "angle 0 deg", 0.0, 24.0, 7.0, 1.0),
        PreviewSpec("Angle", "angle 30 deg", 30.0, 24.0, 7.0, 1.0),
        PreviewSpec("Angle", "angle 60 deg", 60.0, 24.0, 7.0, 1.0),
        PreviewSpec("Angle", "angle 90 deg", 90.0, 24.0, 7.0, 1.0),
        PreviewSpec("Magnitude", "amplitude 0.5", 35.0, 24.0, 7.0, 0.5),
        PreviewSpec("Magnitude", "amplitude 1.0", 35.0, 24.0, 7.0, 1.0),
        PreviewSpec("Magnitude", "amplitude 1.5", 35.0, 24.0, 7.0, 1.5),
    ]


def _plot_preview(
    *,
    specs: list[PreviewSpec],
    intensity: torch.Tensor,
    gradient_magnitude: torch.Tensor,
    output: Path,
) -> None:
    intensity_map = LinearSegmentedColormap.from_list(
        "agfb_intensity",
        [(0.0, BRAND["black"]), (0.55, BRAND["garnet"]), (1.0, BRAND["white"])],
    )
    magnitude_map = LinearSegmentedColormap.from_list(
        "agfb_magnitude",
        [(0.0, BRAND["black"]), (0.72, BRAND["honeycomb"]), (1.0, BRAND["white"])],
    )
    plt.rcParams.update(
        {
            "figure.facecolor": BRAND["white"],
            "axes.facecolor": BRAND["white"],
            "axes.edgecolor": BRAND["gray90"],
            "axes.linewidth": 0.8,
            "font.size": 9,
            "savefig.facecolor": BRAND["white"],
        }
    )

    fig, axes = plt.subplots(4, 4, figsize=(11.0, 8.2), dpi=180, constrained_layout=True)
    title = (
        "Anisotropic Blob Angle And Magnitude Preview\n"
        "All intensity panels share one scale. All g magnitude panels share one scale."
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    intensity_vmax = float(intensity.max())
    magnitude_vmax = float(gradient_magnitude.max())
    image_shape = (int(intensity.shape[-2]), int(intensity.shape[-1]))
    angle_indices = [index for index, spec in enumerate(specs) if spec.group == "Angle"]
    magnitude_indices = [index for index, spec in enumerate(specs) if spec.group == "Magnitude"]

    for col, index in enumerate(angle_indices):
        spec = specs[index]
        axes[0, col].imshow(
            intensity[index].numpy(), cmap=intensity_map, vmin=0.0, vmax=intensity_vmax
        )
        axes[1, col].imshow(
            gradient_magnitude[index].numpy(),
            cmap=magnitude_map,
            vmin=0.0,
            vmax=magnitude_vmax,
        )
        axes[0, col].set_title(spec.label)
        axes[1, col].set_title(spec.label)
        _draw_angle_marker(axes[0, col], spec.angle_deg, image_shape)
        _draw_angle_marker(axes[1, col], spec.angle_deg, image_shape)

    for col, index in enumerate(magnitude_indices):
        spec = specs[index]
        axes[2, col].imshow(
            intensity[index].numpy(), cmap=intensity_map, vmin=0.0, vmax=intensity_vmax
        )
        axes[3, col].imshow(
            gradient_magnitude[index].numpy(),
            cmap=magnitude_map,
            vmin=0.0,
            vmax=magnitude_vmax,
        )
        axes[2, col].set_title(spec.label)
        axes[3, col].set_title(spec.label)
        _draw_angle_marker(axes[2, col], spec.angle_deg, image_shape)
        _draw_angle_marker(axes[3, col], spec.angle_deg, image_shape)

    axes[2, 3].axis("off")
    axes[3, 3].axis("off")
    row_labels = (
        "angle sweep\nintensity",
        "angle sweep\ng magnitude",
        "amplitude sweep\nintensity",
        "amplitude sweep\ng magnitude",
    )
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=34)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(BRAND["gray90"])

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _draw_angle_marker(axis: plt.Axes, angle_deg: float, shape: tuple[int, int]) -> None:
    height, width = shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = min(height, width) * 0.32
    angle = math.radians(angle_deg)
    dx = math.cos(angle) * radius
    dy = math.sin(angle) * radius
    x_points = [center_x - dx, center_x + dx]
    y_points = [center_y - dy, center_y + dy]
    axis.plot(x_points, y_points, color=BRAND["gray90"], linewidth=2.0)
    axis.plot(x_points, y_points, color=BRAND["white"], linewidth=1.0)


if __name__ == "__main__":
    main()
