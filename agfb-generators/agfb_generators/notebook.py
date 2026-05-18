"""Notebook helpers for visual AGFB generator checks."""

from __future__ import annotations

import base64
import html
import math
import os
import platform
import struct
import sys
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agfb_generators import (
    CompositeRect,
    Frame,
    composite,
    curved_arc,
    gaussian_blob,
    gaussian_ridge,
    hard_step,
    polynomial,
    sinusoid,
    smoothed_bar,
    smoothed_step,
)

BrandColor = tuple[float, str]
Renderer = Callable[["NotebookContext"], Frame]

BRAND = {
    "garnet": "#73000A",
    "black": "#000000",
    "white": "#FFFFFF",
    "gray90": "#363636",
    "gray70": "#5C5C5C",
    "gray50": "#A2A2A2",
    "gray30": "#C7C7C7",
    "gray10": "#ECECEC",
    "rose": "#CC2E40",
    "atlantic": "#466A9F",
    "congaree": "#1F414D",
    "grass": "#CED318",
    "honeycomb": "#A49137",
}

PALETTE_INTENSITY = [
    (0.0, BRAND["black"]),
    (0.55, BRAND["garnet"]),
    (1.0, BRAND["white"]),
]
PALETTE_MAGNITUDE = [
    (0.0, BRAND["black"]),
    (0.72, BRAND["honeycomb"]),
    (1.0, BRAND["white"]),
]
PALETTE_SIGNED = [
    (0.0, BRAND["rose"]),
    (0.5, BRAND["white"]),
    (1.0, BRAND["atlantic"]),
]
PALETTE_MASK = [
    (0.0, BRAND["black"]),
    (1.0, BRAND["grass"]),
]


@dataclass(frozen=True)
class GeneratorCase:
    """Describe one visual notebook case and its gradient tolerance."""

    name: str
    description: str
    render: Renderer
    rel_tol: float


@dataclass(frozen=True)
class NotebookContext:
    """Hold shared notebook render settings and available generator cases."""

    height: int
    width: int
    device: torch.device
    dtype: torch.dtype
    cases: tuple[GeneratorCase, ...]


def setup_notebook(
    *,
    height: int = 160,
    width: int = 160,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> NotebookContext:
    """Prepare the notebook, show environment info, and return shared context."""

    ctx = make_context(height=height, width=width, device=device, dtype=dtype)
    _display_html(_style_html())
    _display_html(_environment_html(ctx))
    _display_html(_case_menu_html(ctx))
    return ctx


def make_context(
    *,
    height: int = 160,
    width: int = 160,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> NotebookContext:
    """Create shared notebook settings without displaying anything."""

    return NotebookContext(
        height=height,
        width=width,
        device=_resolve_device(device),
        dtype=dtype,
        cases=_default_cases(),
    )


def generator_names(ctx: NotebookContext) -> list[str]:
    """Return generator names available to `show_case`."""

    return [case.name for case in ctx.cases]


def show_image(
    values: torch.Tensor | np.ndarray,
    title: str = "image",
    *,
    signed: bool = False,
    kind: str = "intensity",
) -> None:
    """Display one tensor or array as a small inline notebook image."""

    if isinstance(values, torch.Tensor):
        image = values.detach().cpu().numpy()
    else:
        image = np.asarray(values)

    palette = PALETTE_INTENSITY
    if kind == "magnitude":
        palette = PALETTE_MAGNITUDE
    elif kind == "mask":
        palette = PALETTE_MASK
    elif signed:
        palette = PALETTE_SIGNED

    _display_html(
        _section_html(
            title,
            "",
            _figure(image.astype(np.float32), title, palette, symmetric=signed),
        )
    )


def render_case(ctx: NotebookContext, name: str) -> Frame:
    """Render one named generator case without displaying it."""

    return _get_case(ctx, name).render(ctx)


def show_case(ctx: NotebookContext, name: str) -> Frame:
    """Render, validate, and display one named generator case."""

    case = _get_case(ctx, name)
    frame = case.render(ctx)
    row = check_frame(name, frame, rel_tol=case.rel_tol)
    _display_html(_metrics_table_html([row]))
    _display_html(_frame_html(name, frame, description=case.description))
    return frame


def show_composite(ctx: NotebookContext) -> tuple[Frame, torch.Tensor]:
    """Render, validate, and display the rectangular composite example."""

    frame, junction = render_composite(ctx)
    assert frame.I.shape == (1, ctx.height, ctx.width)
    assert frame.g.shape == (1, 2, ctx.height, ctx.width)
    assert junction.shape == (ctx.height, ctx.width)
    assert torch.isfinite(frame.I).all()
    assert torch.isfinite(frame.g).all()
    row: dict[str, object] = {
        "name": "composite",
        "batch": frame.batch_size,
        "shape": f"{frame.height}x{frame.width}",
        "max_grad": float(torch.sqrt(frame.gx[0] ** 2 + frame.gy[0] ** 2).max()),
        "signal_px": int(junction.sum()),
        "nrmse": float("nan"),
        "tol": float("nan"),
        "status": "pass",
    }
    _display_html(_metrics_table_html([row]))
    _display_html(_frame_html("composite", frame, description="Rectangular component assembly."))
    _display_html(_mask_html("composite junction mask", junction))
    return frame, junction


def show_batch_demo(ctx: NotebookContext) -> Frame:
    """Render and display a three-frame batched smoothed-step example."""

    batch = smoothed_step(
        ctx.height,
        ctx.width,
        theta_rad=torch.tensor([0.0, math.radians(30.0), math.radians(60.0)], device=ctx.device),
        sigma_e=torch.tensor([2.0, 4.0, 6.0], device=ctx.device),
        device=ctx.device,
        dtype=ctx.dtype,
    )
    assert batch.batch_size == 3
    assert batch.I.shape == (3, ctx.height, ctx.width)
    assert batch.g.shape == (3, 2, ctx.height, ctx.width)
    figures = []
    for idx in range(batch.batch_size):
        image = batch.I[idx].detach().cpu().numpy()
        figures.append(_figure(image, f"batch {idx}", PALETTE_INTENSITY))
    _display_html(
        _section_html(
            "batched smoothed_step",
            "Same generator call, three parameter rows.",
            "".join(figures),
        )
    )
    return batch


def show_all(ctx: NotebookContext) -> None:
    """Run every generator case plus the composite and batch examples."""

    rows: list[dict[str, object]] = []
    for case in ctx.cases:
        frame = case.render(ctx)
        rows.append(check_frame(case.name, frame, rel_tol=case.rel_tol))
        _display_html(_frame_html(case.name, frame, description=case.description))
    _display_html(_metrics_table_html(rows))
    show_composite(ctx)
    show_batch_demo(ctx)


def render_composite(ctx: NotebookContext) -> tuple[Frame, torch.Tensor]:
    """Render the notebook's composite example without displaying it."""

    rects = [
        CompositeRect(0, ctx.height // 2, 0, ctx.width // 2, render_case(ctx, "smoothed_step")),
        CompositeRect(
            0, ctx.height // 2, ctx.width // 2, ctx.width, render_case(ctx, "gaussian_blob")
        ),
        CompositeRect(
            ctx.height // 2, ctx.height, 0, ctx.width // 2, render_case(ctx, "gaussian_ridge")
        ),
        CompositeRect(
            ctx.height // 2, ctx.height, ctx.width // 2, ctx.width, render_case(ctx, "sinusoid")
        ),
    ]
    return composite(
        ctx.height,
        ctx.width,
        rects,
        junction_radius=3,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def check_frame(name: str, frame: Frame, *, rel_tol: float) -> dict[str, object]:
    """Validate one frame and return summary metrics for display."""

    assert frame.I.ndim == 3, f"{name}: expected I shape (B, H, W), got {tuple(frame.I.shape)}"
    assert frame.g.ndim == 4, f"{name}: expected g shape (B, 2, H, W), got {tuple(frame.g.shape)}"
    assert frame.g.shape[1] == 2, f"{name}: gradient channel count must be 2"
    assert frame.I.shape[0] == frame.g.shape[0], f"{name}: batch mismatch"
    assert frame.I.shape[1:] == frame.g.shape[2:], f"{name}: spatial shape mismatch"
    assert torch.isfinite(frame.I).all(), f"{name}: non-finite intensity"
    assert torch.isfinite(frame.g).all(), f"{name}: non-finite gradient"

    image = frame.I[0]
    fd_gx, fd_gy = _fd4(image)
    inner = torch.zeros_like(image, dtype=torch.bool)
    inner[3:-3, 3:-3] = True
    mag = torch.sqrt(frame.gx[0] ** 2 + frame.gy[0] ** 2)
    signal = (mag > 1e-2 * float(mag.max())) & inner
    signal_count = int(signal.sum())
    assert signal_count > 50, f"{name}: signal mask too small ({signal_count})"

    diff_x = (fd_gx - frame.gx[0])[signal]
    diff_y = (fd_gy - frame.gy[0])[signal]
    numerator = torch.mean(diff_x * diff_x + diff_y * diff_y)
    denominator = torch.mean(frame.gx[0][signal] ** 2 + frame.gy[0][signal] ** 2)
    nrmse = float(torch.sqrt(numerator / denominator))
    assert nrmse < rel_tol, f"{name}: NRMSE={nrmse:.3e} >= {rel_tol:.3e}"

    return {
        "name": name,
        "batch": frame.batch_size,
        "shape": f"{frame.height}x{frame.width}",
        "max_grad": float(mag.max()),
        "signal_px": signal_count,
        "nrmse": nrmse,
        "tol": rel_tol,
        "status": "pass",
    }


def _default_cases() -> tuple[GeneratorCase, ...]:
    return (
        GeneratorCase(
            "smoothed_step", "Gaussian-smoothed straight edge.", _case_smoothed_step, 1e-3
        ),
        GeneratorCase("hard_step", "Sharpest band-limited straight edge.", _case_hard_step, 3e-1),
        GeneratorCase("curved_arc", "Radially smoothed circular boundary.", _case_curved_arc, 1e-3),
        GeneratorCase("sinusoid", "Single-frequency oriented grating.", _case_sinusoid, 1e-2),
        GeneratorCase(
            "gaussian_blob", "Isotropic two-dimensional Gaussian peak.", _case_blob, 1e-3
        ),
        GeneratorCase(
            "gaussian_ridge", "Oriented one-dimensional Gaussian ridge.", _case_ridge, 1e-3
        ),
        GeneratorCase("smoothed_bar", "Finite-width bar from two smoothed edges.", _case_bar, 1e-3),
        GeneratorCase("polynomial", "Low-order polynomial scalar field.", _case_polynomial, 1e-3),
    )


def _case_smoothed_step(ctx: NotebookContext) -> Frame:
    return smoothed_step(
        ctx.height,
        ctx.width,
        theta_rad=math.radians(30.0),
        sigma_e=4.0,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _case_hard_step(ctx: NotebookContext) -> Frame:
    return hard_step(
        ctx.height,
        ctx.width,
        theta_rad=math.radians(15.0),
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _case_curved_arc(ctx: NotebookContext) -> Frame:
    return curved_arc(
        ctx.height,
        ctx.width,
        r0=42.0,
        sigma_e=4.0,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _case_sinusoid(ctx: NotebookContext) -> Frame:
    return sinusoid(
        ctx.height,
        ctx.width,
        freq=0.05,
        theta_rad=math.radians(30.0),
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _case_blob(ctx: NotebookContext) -> Frame:
    return gaussian_blob(
        ctx.height,
        ctx.width,
        sigma=8.0,
        x0=-10.0,
        y0=8.0,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _case_ridge(ctx: NotebookContext) -> Frame:
    return gaussian_ridge(
        ctx.height,
        ctx.width,
        sigma=4.0,
        theta_rad=math.radians(20.0),
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _case_bar(ctx: NotebookContext) -> Frame:
    return smoothed_bar(
        ctx.height,
        ctx.width,
        width_px=32.0,
        theta_rad=math.radians(15.0),
        sigma_e=4.0,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _case_polynomial(ctx: NotebookContext) -> Frame:
    coeffs = torch.zeros(1, 4, 4, device=ctx.device, dtype=ctx.dtype)
    coeffs[0, 1, 0] = 0.3
    coeffs[0, 0, 1] = -0.2
    coeffs[0, 2, 1] = 0.05
    coeffs[0, 1, 2] = -0.04
    return polynomial(
        ctx.height,
        ctx.width,
        coeffs=coeffs,
        scale=64.0,
        device=ctx.device,
        dtype=ctx.dtype,
    )


def _get_case(ctx: NotebookContext, name: str) -> GeneratorCase:
    for case in ctx.cases:
        if case.name == name:
            return case
    names = ", ".join(generator_names(ctx))
    raise KeyError(f"unknown generator case {name!r}; available cases: {names}")


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fd4(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(image)
    gy = torch.zeros_like(image)
    gx[:, 2:-2] = (-image[:, 4:] + 8 * image[:, 3:-1] - 8 * image[:, 1:-3] + image[:, :-4]) / 12.0
    gy[2:-2, :] = (-image[4:, :] + 8 * image[3:-1, :] - 8 * image[1:-3, :] + image[:-4, :]) / 12.0
    return gx, gy


def _display_html(markup: str) -> None:
    try:
        from IPython.display import HTML, display
    except ImportError:
        print(markup)
        return
    display(HTML(markup))


def _environment_html(ctx: NotebookContext) -> str:
    cuda_name = "not available"
    if torch.cuda.is_available():
        device_index = ctx.device.index if ctx.device.type == "cuda" else None
        cuda_name = torch.cuda.get_device_name(device_index)
    rows = [
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("cpu cores", str(os.cpu_count() or "unknown")),
        ("torch", torch.__version__),
        ("cuda available", str(torch.cuda.is_available())),
        ("cuda device", cuda_name),
        ("active device", str(ctx.device)),
        ("dtype", str(ctx.dtype).replace("torch.", "")),
        ("render size", f"{ctx.height}x{ctx.width}"),
        ("package path", str(Path(__file__).resolve().parent)),
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


def _case_menu_html(ctx: NotebookContext) -> str:
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(case.name)}</code></td>"
        f"<td>{html.escape(case.description)}</td>"
        f"<td>{case.rel_tol:.1e}</td>"
        "</tr>"
        for case in ctx.cases
    )
    return f"""
    <section class="agfb-block">
      <h3>available generator cells</h3>
      <table class="agfb-table">
        <thead><tr><th>case</th><th>description</th><th>gradient tolerance</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def _metrics_table_html(rows: list[dict[str, object]]) -> str:
    headers = ["name", "batch", "shape", "max_grad", "signal_px", "nrmse", "tol", "status"]
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.3e}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table class='agfb-table'><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def _frame_html(name: str, frame: Frame, *, description: str) -> str:
    image = frame.I[0].detach().cpu().numpy()
    gx = frame.gx[0].detach().cpu().numpy()
    gy = frame.gy[0].detach().cpu().numpy()
    grad_mag = np.sqrt(gx * gx + gy * gy)
    figures = (
        _figure(image, "intensity", PALETTE_INTENSITY)
        + _figure(grad_mag, "gradient magnitude", PALETTE_MAGNITUDE)
        + _figure(gx, "g_x", PALETTE_SIGNED, symmetric=True)
        + _figure(gy, "g_y", PALETTE_SIGNED, symmetric=True)
    )
    return _section_html(name, description, figures)


def _mask_html(title: str, mask: torch.Tensor) -> str:
    image = mask.detach().cpu().numpy().astype(np.float32)
    return _section_html(
        title,
        "Boundary pixels around component ownership changes.",
        _figure(image, "mask", PALETTE_MASK),
    )


def _section_html(title: str, description: str, body: str) -> str:
    description_html = f"<p>{html.escape(description)}</p>" if description else ""
    return (
        "<section class='agfb-block'>"
        f"<h3>{html.escape(title)}</h3>"
        f"{description_html}"
        "<div class='agfb-grid'>" + body + "</div></section>"
    )


def _figure(
    values: np.ndarray,
    title: str,
    palette: list[BrandColor],
    *,
    symmetric: bool = False,
) -> str:
    uri = _image_uri(values, palette, symmetric=symmetric)
    caption = html.escape(title)
    return (
        "<figure class='agfb-figure'>"
        f"<figcaption>{caption}</figcaption>"
        f"<img src='{uri}' alt='{caption}' />"
        "</figure>"
    )


def _image_uri(
    values: np.ndarray,
    palette: list[BrandColor],
    *,
    symmetric: bool = False,
) -> str:
    normalized = _normalize(values, symmetric=symmetric)
    rgb = _apply_palette(normalized, palette)
    encoded = base64.b64encode(_png_rgb(rgb)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _normalize(values: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
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


def _apply_palette(values: np.ndarray, palette: list[BrandColor]) -> np.ndarray:
    positions = np.array([p for p, _ in palette], dtype=np.float32)
    colors = np.stack([_hex_to_rgb(c) for _, c in palette], axis=0)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    channels = [np.interp(flat, positions, colors[:, k]) for k in range(3)]
    rgb = np.stack(channels, axis=1).reshape(*values.shape, 3)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


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


def _style_html() -> str:
    return f"""
    <style>
    .agfb-block {{ border-top: 2px solid {BRAND["garnet"]}; padding: 12px 0 18px; }}
    .agfb-block h3 {{ margin: 0 0 8px; font-size: 16px; color: {BRAND["black"]}; }}
    .agfb-block p {{ margin: 0 0 8px; color: {BRAND["gray90"]}; }}
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
