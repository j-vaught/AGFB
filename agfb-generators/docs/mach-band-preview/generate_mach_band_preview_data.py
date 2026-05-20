from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agfb_generators import mach_band, smoothed_ramp  # noqa: E402

SIZE = 80
CELL_SIZE = 0.014
PLOT_WIDTH = 1.95
PLOT_HEIGHT = 0.78
PLOT_PAD = 0.06
PROFILE_MIN = -0.20
PROFILE_MAX = 1.20

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GARNET = (115, 0, 10)
ATLANTIC = (70, 106, 159)
HONEYCOMB = (164, 145, 55)


@dataclass(frozen=True)
class PreviewCase:
    title: str
    ramp_width: float
    angle_deg: float
    edge_sigma: float
    shoulder_amplitude: float
    shoulder_sigma: float


def main() -> None:
    palettes = {
        "intensity": [_color(level / 63.0, palette="intensity") for level in range(64)],
        "gradient": [_color(level / 63.0, palette="gradient") for level in range(64)],
    }
    panels = []
    for case in _cases():
        angle_rad = math.radians(case.angle_deg)
        frame = mach_band(
            SIZE,
            SIZE,
            ramp_width=case.ramp_width,
            angle_rad=angle_rad,
            edge_sigma=case.edge_sigma,
            shoulder_amplitude=case.shoulder_amplitude,
            shoulder_sigma=case.shoulder_sigma,
        )
        base = smoothed_ramp(
            SIZE,
            SIZE,
            width_px=case.ramp_width,
            theta_rad=angle_rad,
            sigma_e=case.edge_sigma,
        )
        intensity = frame.I[0].detach().cpu().float()
        base_intensity = base.I[0].detach().cpu().float()
        gradient = torch.sqrt(frame.gx[0] * frame.gx[0] + frame.gy[0] * frame.gy[0])
        gradient = gradient.detach().cpu().float()
        row = SIZE // 2
        panels.append(
            {
                "title": case.title,
                "caption": (
                    f"width {case.ramp_width:g} px, sigma {case.edge_sigma:g}, "
                    f"shoulder {case.shoulder_amplitude:g}, shoulder sigma "
                    f"{case.shoulder_sigma:g}, angle {case.angle_deg:g} deg"
                ),
                "intensity": _runs(_scale_intensity(intensity)),
                "gradient": _runs(gradient / gradient.max().clamp_min(1e-12)),
                "profile": _points(intensity[row], base_intensity[row]),
            }
        )

    payload = {
        "size": SIZE,
        "cell_size": CELL_SIZE,
        "plot_width": PLOT_WIDTH,
        "plot_height": PLOT_HEIGHT,
        "palettes": palettes,
        "panels": panels,
    }
    out_path = Path(__file__).with_name("mach_band_preview.json")
    out_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def _cases() -> list[PreviewCase]:
    return [
        PreviewCase("Default Shoulders", 52.0, 0.0, 3.0, 0.08, 4.0),
        PreviewCase("Sharper Edge", 52.0, 0.0, 1.0, 0.08, 4.0),
        PreviewCase("Stronger Shoulders", 52.0, 0.0, 3.0, 0.16, 4.0),
        PreviewCase("Rotated Profile", 52.0, 25.0, 3.0, 0.08, 4.0),
    ]


def _scale_intensity(values: torch.Tensor) -> torch.Tensor:
    return ((values - PROFILE_MIN) / (PROFILE_MAX - PROFILE_MIN)).clamp(0.0, 1.0)


def _runs(values: torch.Tensor) -> list[list[int]]:
    quantized = torch.round(values.clamp(0.0, 1.0) * 63.0).to(torch.int16)
    rows: list[list[int]] = []
    for y in range(int(quantized.shape[0])):
        x = 0
        while x < int(quantized.shape[1]):
            level = int(quantized[y, x])
            run_width = 1
            while x + run_width < int(quantized.shape[1]):
                if int(quantized[y, x + run_width]) != level:
                    break
                run_width += 1
            rows.append([x, y, run_width, level])
            x += run_width
    return rows


def _points(intensity: torch.Tensor, base_intensity: torch.Tensor) -> dict[str, list[list[float]]]:
    return {
        "intensity": _profile_points(intensity),
        "base": _profile_points(base_intensity),
    }


def _profile_points(values: torch.Tensor) -> list[list[float]]:
    n = int(values.numel())
    scaled = ((values - PROFILE_MIN) / (PROFILE_MAX - PROFILE_MIN)).clamp(0.0, 1.0)
    return [
        [
            round(index * PLOT_WIDTH / (n - 1), 5),
            round(PLOT_PAD + float(scaled[index]) * (PLOT_HEIGHT - 2.0 * PLOT_PAD), 5),
        ]
        for index in range(n)
    ]


def _color(value: float, *, palette: str) -> str:
    if palette == "gradient":
        return _blend_three(value, BLACK, HONEYCOMB, WHITE, midpoint=0.55)
    return _blend_three(value, BLACK, GARNET, WHITE, midpoint=0.68)


def _blend_three(
    value: float,
    low: tuple[int, int, int],
    mid: tuple[int, int, int],
    high: tuple[int, int, int],
    *,
    midpoint: float,
) -> str:
    value = max(0.0, min(1.0, value))
    if value <= midpoint:
        return _hex(_lerp_rgb(low, mid, value / midpoint))
    return _hex(_lerp_rgb(mid, high, (value - midpoint) / (1.0 - midpoint)))


def _lerp_rgb(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    red = round(start[0] + (end[0] - start[0]) * amount)
    green = round(start[1] + (end[1] - start[1]) * amount)
    blue = round(start[2] + (end[2] - start[2]) * amount)
    return red, green, blue


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


if __name__ == "__main__":
    main()
