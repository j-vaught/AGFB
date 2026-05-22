"""Notebook helpers for AGFB noise visual checks."""

from __future__ import annotations

import base64
import html
import importlib
import io
import math
import struct
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

_GARNET = torch.tensor([115.0, 0.0, 10.0])
_ATLANTIC = torch.tensor([70.0, 106.0, 159.0])
_WHITE = torch.tensor([255.0, 255.0, 255.0])
_BLACK = "#000000"
_NEUTRAL = "#363636"


def synthetic_1024_image(
    *,
    height: int = 1024,
    width: int = 1024,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return one batched synthetic AGFB image with shape `(1, height, width)`."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    ys = torch.linspace(-1.0, 1.0, steps=height, device=resolved_device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=resolved_device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    smooth_edge = 0.28 * torch.sigmoid((xx + 0.32) / 0.035)
    blob = 0.30 * torch.exp(-(((xx - 0.32) / 0.23) ** 2 + ((yy + 0.22) / 0.17) ** 2) / 2.0)
    ridge_axis = xx * math.cos(0.72) + yy * math.sin(0.72) + 0.18
    ridge = 0.22 * torch.exp(-(ridge_axis**2) / (2.0 * 0.045**2))
    grating = 0.07 * torch.sin(2.0 * math.pi * (5.0 * xx - 3.0 * yy))
    ramp = 0.18 * (0.55 * xx + 0.45 * yy + 1.0) / 2.0
    image = (0.18 + smooth_edge + blob + ridge + grating + ramp).clamp(0.0, 1.0)
    return image.unsqueeze(0).contiguous()


def summarize_tensors(tensors: Mapping[str, torch.Tensor]) -> list[dict[str, float | str]]:
    """Return compact scalar summaries for notebook display."""
    rows: list[dict[str, float | str]] = []
    for name, tensor in tensors.items():
        values = tensor.detach().to(torch.float32).reshape(-1).cpu()
        quantiles = torch.quantile(values, torch.tensor([0.01, 0.5, 0.99]))
        rows.append(
            {
                "name": name,
                "min": float(values.min().item()),
                "q01": float(quantiles[0].item()),
                "mean": float(values.mean().item()),
                "q50": float(quantiles[1].item()),
                "q99": float(quantiles[2].item()),
                "max": float(values.max().item()),
                "std": float(values.std(unbiased=False).item()),
            }
        )
    return rows


def show_noise_preview(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    *,
    title: str,
    max_size: int = 384,
) -> Any:
    """Display clean, noisy, and residual tensors as inline notebook bitmaps."""
    clean_2d = _as_2d(clean)
    noisy_2d = _as_2d(noisy)
    if clean_2d.shape != noisy_2d.shape:
        raise ValueError("clean and noisy images must have matching shapes")
    residual = noisy_2d - clean_2d
    html_text = _preview_html(
        title=title,
        panels=(
            ("Clean 1024 image", _bmp_data_url(_to_grayscale_rgb(clean_2d, max_size=max_size))),
            ("Noisy image", _bmp_data_url(_to_grayscale_rgb(noisy_2d, max_size=max_size))),
            ("Noisy - clean", _bmp_data_url(_to_diverging_rgb(residual, max_size=max_size))),
        ),
    )
    try:
        display_module = importlib.import_module("IPython.display")
    except ModuleNotFoundError:
        return html_text
    html_obj = display_module.HTML(html_text)
    display_module.display(html_obj)
    return html_obj


def _as_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError("batched tensors must contain exactly one image for preview")
        return tensor[0].detach()
    if tensor.ndim == 2:
        return tensor.detach()
    raise ValueError(f"preview tensors must be `(H, W)` or `(1, H, W)`, got {tuple(tensor.shape)}")


def _downsample(tensor: torch.Tensor, *, max_size: int) -> torch.Tensor:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    height, width = tensor.shape[-2:]
    scale = min(1.0, float(max_size) / float(max(height, width)))
    if scale == 1.0:
        return tensor.to(torch.float32).cpu()
    out_height = max(1, int(round(height * scale)))
    out_width = max(1, int(round(width * scale)))
    sampled = F.interpolate(
        tensor.to(torch.float32).view(1, 1, height, width),
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
    )
    return sampled[0, 0].cpu()


def _to_grayscale_rgb(tensor: torch.Tensor, *, max_size: int) -> torch.Tensor:
    sampled = _downsample(tensor, max_size=max_size)
    low = torch.quantile(sampled.reshape(-1), 0.005)
    high = torch.quantile(sampled.reshape(-1), 0.995)
    normalized = ((sampled - low) / (high - low).clamp_min(1e-12)).clamp(0.0, 1.0)
    channel = (normalized * 255.0).round().to(torch.uint8)
    return channel.unsqueeze(-1).expand(*channel.shape, 3).contiguous()


def _to_diverging_rgb(tensor: torch.Tensor, *, max_size: int) -> torch.Tensor:
    sampled = _downsample(tensor, max_size=max_size)
    limit = torch.quantile(sampled.abs().reshape(-1), 0.995).clamp_min(1e-12)
    values = (sampled / limit).clamp(-1.0, 1.0)
    negative = values < 0
    positive = values > 0
    magnitude = values.abs().unsqueeze(-1)
    white = _WHITE.to(dtype=torch.float32)
    atlantic = _ATLANTIC.to(dtype=torch.float32)
    garnet = _GARNET.to(dtype=torch.float32)
    rgb = white.expand(*values.shape, 3).clone()
    rgb = torch.where(negative.unsqueeze(-1), white * (1.0 - magnitude) + atlantic * magnitude, rgb)
    rgb = torch.where(positive.unsqueeze(-1), white * (1.0 - magnitude) + garnet * magnitude, rgb)
    return rgb.round().clamp(0, 255).to(torch.uint8).contiguous()


def _bmp_data_url(rgb: torch.Tensor) -> str:
    data = _bmp_bytes(rgb)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/bmp;base64,{encoded}"


def _bmp_bytes(rgb: torch.Tensor) -> bytes:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb image must have shape `(H, W, 3)`")
    rgb_cpu = rgb.detach().cpu().contiguous()
    height, width, _ = rgb_cpu.shape
    row_bytes = width * 3
    padding = (4 - row_bytes % 4) % 4
    pixel_bytes = (row_bytes + padding) * height
    file_size = 14 + 40 + pixel_bytes
    buffer = io.BytesIO()
    buffer.write(b"BM")
    buffer.write(struct.pack("<IHHI", file_size, 0, 0, 54))
    buffer.write(struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_bytes, 0, 0, 0, 0))
    padding_bytes = b"\x00" * padding
    array = rgb_cpu.numpy()
    for row_index in range(height - 1, -1, -1):
        row = array[row_index, :, ::-1].tobytes()
        buffer.write(row)
        buffer.write(padding_bytes)
    return buffer.getvalue()


def _preview_html(*, title: str, panels: tuple[tuple[str, str], ...]) -> str:
    panel_html = "\n".join(
        f"""
        <figure style="margin:0; min-width:0;">
          <figcaption style="font:600 13px system-ui; color:{_NEUTRAL}; margin-bottom:6px;">
            {html.escape(label)}
          </figcaption>
          <img
            src="{url}"
            style="display:block; width:100%; image-rendering:auto; border:1px solid #000000;"
          />
        </figure>
        """
        for label, url in panels
    )
    return f"""
    <section style="font-family:system-ui, sans-serif;">
      <h2 style="font-size:18px; margin:0 0 10px 0; color:{_BLACK};">{html.escape(title)}</h2>
      <div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px;">
        {panel_html}
      </div>
    </section>
    """
