#!/usr/bin/env python3
"""Render a CPGF step-edge response figure through Typst and CeTZ."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agfb_filters import ExecutionPath, cpgf_definition, run_filter


def render_step_response(output_path: Path) -> None:
    image_height = 106
    image_width = 160
    edge_column = image_width // 2
    image = torch.zeros(1, image_height, image_width)
    image[:, :, edge_column:] = 1.0

    definition = cpgf_definition(radius=2, degree=2)
    gradient_x, _ = run_filter(
        definition,
        image,
        path=ExecutionPath.SPATIAL_DENSE,
        boundary=definition.default_boundary,
    )
    response_by_column = gradient_x.abs().mean(dim=(0, 1))
    response_max = float(response_by_column.max())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_suffix(".csv")
    typst_path = output_path.with_suffix(".typ")
    _write_response_data(data_path, response_by_column, response_max)
    _write_typst_source(
        typst_path,
        data_path=data_path,
        image_width=image_width,
        image_height=image_height,
    )
    subprocess.run(
        ["typst", "compile", typst_path.name, output_path.name],
        cwd=typst_path.parent,
        check=True,
    )
    print(f"wrote {output_path}")
    print(f"wrote {data_path}")
    print(f"wrote {typst_path}")
    print(f"image_shape=(1, {image_height}, {image_width}) edge_column={edge_column}")
    print(f"peak_response={response_max:.9f}")


def _write_response_data(
    data_path: Path,
    response_by_column: torch.Tensor,
    response_max: float,
) -> None:
    with data_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        for column, value in enumerate(response_by_column):
            normalized = float(value) / response_max if response_max else 0.0
            writer.writerow([column, f"{normalized:.9f}"])


def _write_typst_source(
    typst_path: Path,
    *,
    data_path: Path,
    image_width: int,
    image_height: int,
) -> None:
    scale = 4
    margin = 24
    gap = 32
    panel_width = image_width * scale
    panel_height = image_height * scale
    canvas_width = 2 * panel_width + gap + 2 * margin
    canvas_height = panel_height + 2 * margin
    data_name = data_path.name
    typst_path.write_text(
        f"""#import "@preview/cetz:0.3.4": canvas, draw
#set page(width: {canvas_width}pt, height: {canvas_height}pt, margin: 0pt)

#let black = rgb("#000000")
#let white = rgb("#FFFFFF")
#let garnet = rgb("#73000A")
#let rose = rgb("#CC2E40")
#let border = rgb("#363636")
#let data = csv("{data_name}")
#let scale = {scale}
#let margin = {margin}
#let gap = {gap}
#let panel-width = {panel_width}
#let panel-height = {panel_height}
#let left-panel-x = margin
#let right-panel-x = margin + panel-width + gap
#let panel-y = margin

#canvas(length: 1pt, {{
  draw.rect((0, 0), ({canvas_width}, {canvas_height}), fill: white, stroke: none)
  draw.rect(
    (left-panel-x, panel-y),
    (left-panel-x + panel-width / 2, panel-y + panel-height),
    fill: black,
    stroke: none,
  )
  draw.rect(
    (left-panel-x + panel-width / 2, panel-y),
    (left-panel-x + panel-width, panel-y + panel-height),
    fill: white,
    stroke: none,
  )
  for row in data {{
    let column = int(row.at(0))
    let normalized = float(row.at(1))
    let color = if normalized < 0.001 {{
      white
    }} else if normalized < 0.5 {{
      rose
    }} else {{
      garnet
    }}
    draw.rect(
      (right-panel-x + column * scale, panel-y),
      (right-panel-x + (column + 1) * scale, panel-y + panel-height),
      fill: color,
      stroke: none,
    )
  }}
  draw.rect(
    (left-panel-x, panel-y),
    (left-panel-x + panel-width, panel-y + panel-height),
    fill: none,
    stroke: border,
  )
  draw.rect(
    (right-panel-x, panel-y),
    (right-panel-x + panel-width, panel-y + panel-height),
    fill: none,
    stroke: border,
  )
}})
""",
        encoding="utf-8",
    )


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
