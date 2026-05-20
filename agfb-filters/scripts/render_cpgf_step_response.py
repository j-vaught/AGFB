#!/usr/bin/env python3
"""Render a CPGF step-edge response PNG with Python."""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agfb_filters import cpgf_definition, run_filter

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GARNET = (115, 0, 10)
ROSE = (204, 46, 64)
BORDER = (54, 54, 54)


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw_rows = []
    for row in pixels:
        raw_rows.append(b"\x00" + b"".join(bytes(pixel) for pixel in row))
    raw = b"".join(raw_rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def draw_rect(
    pixels: list[list[tuple[int, int, int]]],
    left: int,
    top: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    for row in range(top, top + height):
        pixels[row][left : left + width] = [color] * width


def draw_border(
    pixels: list[list[tuple[int, int, int]]],
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    draw_rect(pixels, left, top, width, 1, BORDER)
    draw_rect(pixels, left, top + height - 1, width, 1, BORDER)
    draw_rect(pixels, left, top, 1, height, BORDER)
    draw_rect(pixels, left + width - 1, top, 1, height, BORDER)


def response_color(normalized_response: float) -> tuple[int, int, int]:
    if normalized_response < 0.001:
        return WHITE
    if normalized_response < 0.5:
        return ROSE
    return GARNET


def render_step_response(output_path: Path) -> None:
    image_height = 106
    image_width = 160
    edge_column = image_width // 2
    scale = 4
    margin = 24
    gap = 32
    panel_width = image_width * scale
    panel_height = image_height * scale
    canvas_width = 2 * panel_width + gap + 2 * margin
    canvas_height = panel_height + 2 * margin

    image = torch.zeros(1, image_height, image_width)
    image[:, :, edge_column:] = 1.0
    gradient_x, _ = run_filter(cpgf_definition(radius=2, degree=2), image)
    response_by_column = gradient_x.abs().mean(dim=(0, 1))
    response_max = float(response_by_column.max())

    pixels = [[WHITE for _ in range(canvas_width)] for _ in range(canvas_height)]
    left_panel_x = margin
    right_panel_x = margin + panel_width + gap
    panel_y = margin

    draw_rect(pixels, left_panel_x, panel_y, panel_width // 2, panel_height, BLACK)
    draw_rect(
        pixels,
        left_panel_x + panel_width // 2,
        panel_y,
        panel_width // 2,
        panel_height,
        WHITE,
    )

    for column in range(image_width):
        value = float(response_by_column[column])
        normalized = value / response_max if response_max else 0.0
        color = response_color(normalized)
        draw_rect(
            pixels,
            right_panel_x + column * scale,
            panel_y,
            scale,
            panel_height,
            color,
        )

    draw_border(pixels, left_panel_x, panel_y, panel_width, panel_height)
    draw_border(pixels, right_panel_x, panel_y, panel_width, panel_height)
    write_png(output_path, pixels)
    print(f"wrote {output_path}")
    print(f"image_shape=(1, {image_height}, {image_width}) edge_column={edge_column}")
    print(f"peak_response={response_max:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cpgf_step_response.png"),
    )
    args = parser.parse_args()
    render_step_response(args.output)


if __name__ == "__main__":
    main()
