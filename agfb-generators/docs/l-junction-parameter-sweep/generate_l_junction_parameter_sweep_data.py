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

from agfb_generators import smoothed_l_junction  # noqa: E402

SIZE = 72
CENTER_X = -13.0
CENTER_Y = -13.0
CELL_SIZE = 0.018

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GARNET = (115, 0, 10)
HONEYCOMB = (164, 145, 55)


@dataclass(frozen=True)
class SweepCase:
    family: str
    title: str
    arm_width: float
    angle_deg: float
    edge_sigma: float


def main() -> None:
    cases = _cases()
    palettes = {
        "intensity": [_color(level / 63.0, palette="intensity") for level in range(64)],
        "gradient": [_color(level / 63.0, palette="gradient") for level in range(64)],
    }
    panels = []
    for case in cases:
        frame = smoothed_l_junction(
            SIZE,
            SIZE,
            arm_width=case.arm_width,
            angle_rad=math.radians(case.angle_deg),
            center_x=CENTER_X,
            center_y=CENTER_Y,
            edge_sigma=case.edge_sigma,
        )
        intensity = frame.I[0].detach().cpu().float()
        gradient = torch.sqrt(frame.gx[0] * frame.gx[0] + frame.gy[0] * frame.gy[0])
        gradient = gradient.detach().cpu().float()
        panels.append(
            {
                "family": case.family,
                "title": case.title,
                "caption": (
                    f"sigma {case.edge_sigma:g}, width {case.arm_width:g} px, "
                    f"angle {case.angle_deg:g} deg, g max {float(gradient.max()):.4g}"
                ),
                "intensity": _runs(intensity),
                "gradient": _runs(gradient / gradient.max().clamp_min(1e-12)),
            }
        )

    out_path = Path(__file__).with_name("l_junction_parameter_sweep.json")
    payload = {
        "size": SIZE,
        "cell_size": CELL_SIZE,
        "palettes": palettes,
        "panels": panels,
    }
    out_path.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _cases() -> list[SweepCase]:
    return [
        SweepCase("Edge Sigma", "sigma = 0.5 px", 18.0, 0.0, 0.5),
        SweepCase("Edge Sigma", "sigma = 1 px", 18.0, 0.0, 1.0),
        SweepCase("Edge Sigma", "sigma = 3 px", 18.0, 0.0, 3.0),
        SweepCase("Edge Sigma", "sigma = 6 px", 18.0, 0.0, 6.0),
        SweepCase("Arm Width", "width = 10 px", 10.0, 0.0, 3.0),
        SweepCase("Arm Width", "width = 18 px", 18.0, 0.0, 3.0),
        SweepCase("Arm Width", "width = 28 px", 28.0, 0.0, 3.0),
        SweepCase("Arm Width", "width = 36 px", 36.0, 0.0, 3.0),
        SweepCase("Rotation", "angle = 0 deg", 18.0, 0.0, 3.0),
        SweepCase("Rotation", "angle = 22.5 deg", 18.0, 22.5, 3.0),
        SweepCase("Rotation", "angle = 45 deg", 18.0, 45.0, 3.0),
        SweepCase("Rotation", "angle = 67.5 deg", 18.0, 67.5, 3.0),
    ]


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


def _color(value: float, *, palette: str) -> str:
    if palette == "gradient":
        return _blend_three(value, BLACK, HONEYCOMB, WHITE, midpoint=0.5)
    return _blend_three(value, BLACK, GARNET, WHITE, midpoint=0.62)


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
