"""Render a catalog :class:`Cell` into an ``agfb-generators`` Frame.

The Frame carries ``.I`` (1, H, W intensity) and the analytic gradient
``.gx`` / ``.gy`` (each 1, H, W), which the metrics score against.
"""

from __future__ import annotations

import torch

from agfb_bench.catalog import Cell


def render_cell(cell: Cell, image_size: int, device: torch.device):
    """Render a clean field. Returns the generator ``Frame``."""
    import agfb_generators

    generator_fn = getattr(agfb_generators, cell.generator)
    params = dict(cell.params)
    coefficients = params.get("coefficients")
    if isinstance(coefficients, torch.Tensor):
        params["coefficients"] = coefficients.to(device)
    return generator_fn(image_size, image_size, device=device, **params)
