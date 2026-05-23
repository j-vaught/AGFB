"""Small display helpers for AGFB metric notebooks.

These helpers render inline images and tables in a Jupyter environment without
any plotting dependency. Images are encoded as PNG data URIs and tables are
rendered as styled HTML, so a notebook only needs `agfb_metrics` and IPython.
"""

from __future__ import annotations

import base64
import html
import os
import platform
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

ColorStop = tuple[float, str]
ColorScheme = dict[str, list[ColorStop]]


BRAND = {
    "garnet": "#73000A",
    "black": "#000000",
    "white": "#FFFFFF",
    "gray90": "#363636",
    "gray30": "#C7C7C7",
    "gray10": "#ECECEC",
    "rose": "#CC2E40",
    "atlantic": "#466A9F",
    "grass": "#CED318",
    "honeycomb": "#A49137",
}

_COLOR_SCHEME: ColorScheme = {
    "intensity": [(0.0, BRAND["black"]), (0.55, BRAND["garnet"]), (1.0, BRAND["white"])],
    "magnitude": [(0.0, BRAND["black"]), (0.72, BRAND["honeycomb"]), (1.0, BRAND["white"])],
    "signed": [(0.0, BRAND["rose"]), (0.5, BRAND["white"]), (1.0, BRAND["atlantic"])],
    "mask": [(0.0, BRAND["black"]), (1.0, BRAND["grass"])],
}


def setup_notebook(*, height: int, width: int) -> None:
    """Install the notebook stylesheet and show runtime information."""

    _display_html(_style_html())
    _display_html(_runtime_html(height=height, width=width))


def set_color_scheme(color_scheme: ColorScheme) -> None:
    """Set the color scheme used by subsequent `show_image` calls."""

    global _COLOR_SCHEME
    _COLOR_SCHEME = _normalize_color_scheme(color_scheme)


def show_color_scheme(color_scheme: ColorScheme | None = None) -> None:
    """Show a legend for the active or supplied color scheme."""

    scheme = _normalize_color_scheme(color_scheme or _COLOR_SCHEME)
    _display_html(_color_scheme_html(scheme))


def show_image(
    values: torch.Tensor | np.ndarray,
    title: str = "image",
    *,
    kind: str = "intensity",
    signed: bool = False,
    color_scheme: ColorScheme | None = None,
) -> None:
    """Display one image tensor or array inline."""

    show_images([(values, title)], kind=kind, signed=signed, color_scheme=color_scheme)


def show_images(
    items: Sequence[tuple[torch.Tensor | np.ndarray, str]],
    *,
    title: str | None = None,
    kind: str = "intensity",
    signed: bool = False,
    color_scheme: ColorScheme | None = None,
) -> None:
    """Display a row of image tensors under one section heading."""

    scheme = _normalize_color_scheme(color_scheme or _COLOR_SCHEME)
    palette = _palette_for(kind=kind, signed=signed, color_scheme=scheme)
    figures = []
    for values, caption in items:
        if isinstance(values, torch.Tensor):
            image = values.detach().cpu().numpy()
        else:
            image = np.asarray(values)
        figures.append(_figure(image.astype(np.float32), caption, palette, symmetric=signed))
    heading = title if title is not None else (items[0][1] if len(items) == 1 else "images")
    _display_html(_section_html(heading, "".join(figures)))


def show_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    title: str = "table",
    highlight_first_column: bool = True,
) -> None:
    """Render a labelled result table as styled HTML."""

    head = "".join(f"<th>{html.escape(str(name))}</th>" for name in headers)
    body_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            tag = "th" if (highlight_first_column and index == 0) else "td"
            cells.append(f"<{tag}>{html.escape(_format_cell(value))}</{tag}>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    table = (
        "<table class='agfb-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    _display_html(_section_html(title, table))


def metric_table(
    results: Mapping[str, Mapping[str, float]],
    *,
    title: str = "metrics",
    decimals: int = 4,
) -> None:
    """Render a predictor-by-metric table from a nested result mapping.

    `results` maps a predictor label to a mapping of metric name to score. One
    row is emitted per predictor and one column per metric, in first-seen order.
    """

    metric_names: list[str] = []
    for scores in results.values():
        for name in scores:
            if name not in metric_names:
                metric_names.append(name)
    headers = ["predictor", *metric_names]
    rows = []
    for label, scores in results.items():
        row: list[object] = [label]
        for name in metric_names:
            value = scores.get(name)
            row.append(value if value is None else round(float(value), decimals))
        rows.append(row)
    show_table(headers, rows, title=title)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
    return str(value)


def _palette_for(*, kind: str, signed: bool, color_scheme: ColorScheme) -> list[ColorStop]:
    if signed:
        return color_scheme["signed"]
    if kind not in color_scheme:
        options = ", ".join(color_scheme)
        raise KeyError(f"unknown color scheme kind {kind!r}; available kinds: {options}")
    return color_scheme[kind]


def _runtime_html(*, height: int, width: int) -> str:
    rows = [
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("cpu cores", str(os.cpu_count() or "unknown")),
        ("torch", torch.__version__),
        ("cuda available", str(torch.cuda.is_available())),
        ("cuda device", _cuda_name()),
        ("render size", f"{height}x{width}"),
        ("helper path", str(Path(__file__).resolve())),
    ]
    body = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>" for key, value in rows
    )
    return f"""
    <section class="agfb-block">
      <h3>setup</h3>
      <table class="agfb-table"><tbody>{body}</tbody></table>
    </section>
    """


def _cuda_name() -> str:
    if not torch.cuda.is_available():
        return "not available"
    return torch.cuda.get_device_name(None)


def _normalize_color_scheme(color_scheme: ColorScheme) -> ColorScheme:
    required = ("intensity", "magnitude", "signed", "mask")
    missing = [key for key in required if key not in color_scheme]
    if missing:
        raise KeyError(f"color scheme missing keys: {', '.join(missing)}")
    return {
        key: [(float(position), str(color)) for position, color in color_scheme[key]]
        for key in required
    }


def _color_scheme_html(color_scheme: ColorScheme) -> str:
    descriptions = {
        "intensity": "Scalar image values.",
        "magnitude": "Nonnegative gradient magnitude values.",
        "signed": "Signed fields such as g_x or g_y.",
        "mask": "Boolean or binary masks.",
    }
    rows = []
    for name, palette in color_scheme.items():
        gradient = ", ".join(f"{color} {100.0 * position:.1f}%" for position, color in palette)
        stops = ", ".join(f"{position:.2f}: {color}" for position, color in palette)
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            "<td>"
            f"<div class='agfb-swatch' style='background: linear-gradient(to right, {gradient});'>"
            "</div>"
            "</td>"
            f"<td>{html.escape(stops)}</td>"
            f"<td>{html.escape(descriptions[name])}</td>"
            "</tr>"
        )
    return f"""
    <section class="agfb-block">
      <h3>color scheme</h3>
      <table class="agfb-table">
        <thead><tr><th>role</th><th>legend</th><th>stops</th><th>used for</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """


def _section_html(title: str, body: str) -> str:
    return (
        "<section class='agfb-block'>"
        f"<h3>{html.escape(title)}</h3>"
        "<div class='agfb-grid'>"
        f"{body}"
        "</div></section>"
    )


def _figure(
    values: np.ndarray,
    title: str,
    palette: list[ColorStop],
    *,
    symmetric: bool,
) -> str:
    uri = _image_uri(values, palette, symmetric=symmetric)
    caption = html.escape(title)
    return (
        "<figure class='agfb-figure'>"
        f"<figcaption>{caption}</figcaption>"
        f"<img src='{uri}' alt='{caption}' />"
        "</figure>"
    )


def _image_uri(values: np.ndarray, palette: list[ColorStop], *, symmetric: bool) -> str:
    normalized = _normalize_values(values, symmetric=symmetric)
    rgb = _apply_palette(normalized, palette)
    encoded = base64.b64encode(_png_rgb(rgb)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize_values(values: np.ndarray, *, symmetric: bool) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(data)
    if not mask.any():
        return np.zeros_like(data, dtype=np.float32)
    if symmetric:
        limit = float(np.max(np.abs(data[mask])))
        if limit <= 1e-12:
            return np.full_like(data, 0.5, dtype=np.float32)
        return np.clip(0.5 + 0.5 * data / limit, 0.0, 1.0)
    lo = float(np.min(data[mask]))
    hi = float(np.max(data[mask]))
    if hi - lo <= 1e-12:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def _apply_palette(values: np.ndarray, palette: list[ColorStop]) -> np.ndarray:
    positions = np.array([position for position, _ in palette], dtype=np.float32)
    colors = np.stack([_hex_to_rgb(color) for _, color in palette], axis=0)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    channels = [np.interp(flat, positions, colors[:, channel]) for channel in range(3)]
    rgb = np.stack(channels, axis=1).reshape(*values.shape, 3)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[index : index + 2], 16) for index in (0, 2, 4)], dtype=np.float32)


def _png_rgb(rgb: np.ndarray) -> bytes:
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"expected RGB image, got shape {rgb.shape}")
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    payload = tag + data
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", checksum)


def _display_html(markup: str) -> None:
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise RuntimeError(
            "agfb_metrics.notebook helpers require an IPython or Jupyter environment"
        ) from exc
    display(HTML(markup))


def _style_html() -> str:
    return f"""
    <style>
    .agfb-block {{ border-top: 2px solid {BRAND["garnet"]}; padding: 12px 0 18px; }}
    .agfb-block h3 {{ margin: 0 0 8px; font-size: 16px; color: {BRAND["black"]}; }}
    .agfb-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 10px;
    }}
    .agfb-figure {{
        margin: 0;
        border: 1px solid {BRAND["gray30"]};
        background: {BRAND["white"]};
    }}
    .agfb-figure figcaption {{
        padding: 5px 6px;
        font: 12px sans-serif;
        background: {BRAND["gray10"]};
        color: {BRAND["black"]};
    }}
    .agfb-figure img {{ display: block; width: 100%; image-rendering: pixelated; }}
    .agfb-swatch {{ height: 22px; min-width: 140px; border: 1px solid {BRAND["gray30"]}; }}
    .agfb-table {{ border-collapse: collapse; width: 100%; font: 13px sans-serif; }}
    .agfb-table th {{
        background: {BRAND["black"]};
        color: {BRAND["white"]};
        text-align: left;
    }}
    .agfb-table th,
    .agfb-table td {{ border: 1px solid {BRAND["gray30"]}; padding: 6px 8px; }}
    code {{ color: {BRAND["garnet"]}; }}
    </style>
    """
