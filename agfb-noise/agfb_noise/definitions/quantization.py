"""Uniform scalar quantization noise."""

from __future__ import annotations

import torch

from agfb_noise.base import ClampRange, Numeric, apply_clamp, check_image

NOISE_SPECS = (
    {
        "name": "quantization",
        "function": "add_quantization",
        "description": "uniform scalar quantization over a finite intensity range",
        "aliases": ("quantize",),
    },
)


def add_quantization(
    image: torch.Tensor,
    *,
    levels: int,
    min_value: Numeric = 0.0,
    max_value: Numeric = 1.0,
    clamp: ClampRange = None,
) -> torch.Tensor:
    """Quantize image values to `levels` uniformly spaced values."""
    image = check_image(image)
    levels_int = int(levels)
    if levels_int < 2:
        raise ValueError("levels must be at least two")
    low = torch.as_tensor(min_value, dtype=image.dtype, device=image.device)
    high = torch.as_tensor(max_value, dtype=image.dtype, device=image.device)
    if bool((high <= low).any().item()):
        raise ValueError("max_value must be greater than min_value")
    scaled = ((image - low) / (high - low)).clamp(0.0, 1.0)
    quantized = torch.round(scaled * float(levels_int - 1)) / float(levels_int - 1)
    out = quantized * (high - low) + low
    return apply_clamp(out, clamp)
