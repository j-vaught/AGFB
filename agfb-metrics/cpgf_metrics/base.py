"""Shared helpers for the CPGF benchmark metrics.

All metrics consume the same five inputs:
    g_x, g_y         : (B, H, W) float32 filter output
    g_x_t, g_y_t     : (B, H, W) float32 ground-truth gradient field
    signal_mask      : (B, H, W) bool, true edge pixels (axis A and B)
    flat_mask        : (B, H, W) bool, flat background pixels (axis C)

`masks(gx_t, gy_t)` constructs both masks from the truth field in one call,
matching the §1.1 protocol used by the existing PGF_paper prototype.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def check_grad_pair(g_x: torch.Tensor, g_y: torch.Tensor, name: str = "gradient") -> None:
    if g_x.shape != g_y.shape:
        raise ValueError(f"{name} g_x and g_y must share shape; got {g_x.shape} vs {g_y.shape}")
    if g_x.ndim != 3:
        raise ValueError(f"{name} tensors must be (B, H, W); got {tuple(g_x.shape)}")
    if g_x.dtype != torch.float32 or g_y.dtype != torch.float32:
        raise ValueError(f"{name} tensors must be float32; got {g_x.dtype} and {g_y.dtype}")
    if g_x.device != g_y.device:
        raise ValueError(f"{name} g_x and g_y must be on same device")


def magnitude(g_x: torch.Tensor, g_y: torch.Tensor) -> torch.Tensor:
    """Vector magnitude `sqrt(g_x^2 + g_y^2)` element-wise; same shape as inputs."""
    return torch.sqrt(g_x * g_x + g_y * g_y)


def masks(
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    *,
    dilate_px: int = 8,
    rel_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Build the per-image (B, H, W) signal and flat-region masks from truth.

    * Signal mask: `|grad_true|(p) > rel_eps * max(|grad_true|)` per image.
    * Flat mask: complement of the signal mask, eroded inward by `dilate_px`
      pixels so that no flat pixel sits within the spatial support of an edge
      pixel (matches the §1.1 protocol).

    Bit-identical, per image, to the prototype `mini.masks(gx_t, gy_t)` (which
    operates on a single image at a time).
    """
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    mag = magnitude(g_x_t, g_y_t)
    max_per_image = mag.reshape(mag.shape[0], -1).amax(dim=1).clamp_min(1e-30)
    eps = (rel_eps * max_per_image).view(-1, 1, 1)
    signal = mag > eps

    k = 2 * dilate_px + 1
    not_sig = (~signal).to(torch.float32).unsqueeze(1)  # (B, 1, H, W)
    eroded = -F.max_pool2d(-not_sig, kernel_size=k, stride=1, padding=dilate_px)
    flat = eroded.squeeze(1) > 0.5
    return {"signal": signal, "flat": flat}


def unit_normal_from_truth(
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit normal `n̂ = grad_true / |grad_true|` on the truth field.

    Returned as `(n_x, n_y)`, each `(B, H, W)`. Pixels with magnitude below
    `eps` are returned as `(0, 0)` rather than left as NaN — callers must
    restrict to the signal mask before using the result.
    """
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    mag = magnitude(g_x_t, g_y_t)
    safe = mag.clamp_min(eps)
    n_x = g_x_t / safe
    n_y = g_y_t / safe
    n_x = torch.where(mag > eps, n_x, torch.zeros_like(n_x))
    n_y = torch.where(mag > eps, n_y, torch.zeros_like(n_y))
    return n_x, n_y


def masked_reduce_per_image(
    values: torch.Tensor,
    mask: torch.Tensor,
    reducer,
) -> torch.Tensor:
    """Apply `reducer(masked_values_1d)` per image; return (B,) float32 tensor.

    `reducer` is called once per image with a flat 1-D tensor of the masked
    values. Images whose mask is empty yield NaN so the sweep aggregator can
    detect them rather than silently producing 0/0.
    """
    if values.shape != mask.shape:
        raise ValueError(f"values {values.shape} and mask {mask.shape} must match")
    out = torch.empty(values.shape[0], dtype=torch.float32, device=values.device)
    for i in range(values.shape[0]):
        m = mask[i]
        if not bool(m.any()):
            out[i] = float("nan")
            continue
        out[i] = float(reducer(values[i][m]))
    return out
