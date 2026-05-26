"""Shared helpers for the Analytical Gradient Filter Benchmark metrics.

All metrics consume the same five inputs:
    g_x, g_y         : (B, H, W) float32 filter output
    g_x_t, g_y_t     : (B, H, W) float32 ground-truth gradient field
    signal_mask      : (B, H, W) bool, true-gradient signal pixels
    flat_mask        : (B, H, W) bool, flat background pixels

`masks(gx_t, gy_t)` constructs both masks from the truth field in one call,
matching the Section 1.1 protocol used by the existing PGF_paper prototype.
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


def masked_count_per_image(mask: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor:
    """Count true mask values per image as `(B,)` float32 on the mask device."""
    if mask is None:
        pixels_per_image = values.shape[1] * values.shape[2]
        return torch.full(
            (values.shape[0],),
            float(pixels_per_image),
            dtype=torch.float32,
            device=values.device,
        )
    if mask.ndim != 3:
        raise ValueError(f"mask must be (B, H, W); got {tuple(mask.shape)}")
    return mask.reshape(mask.shape[0], -1).sum(dim=1).to(torch.float32)


def masked_sum_per_image(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Sum masked values per image as `(B,)` float32 on the values device."""
    if mask is None:
        return values.reshape(values.shape[0], -1).sum(dim=1)
    if values.shape != mask.shape:
        raise ValueError(f"values {values.shape} and mask {mask.shape} must match")
    masked = torch.where(mask, values, torch.zeros_like(values))
    return masked.reshape(values.shape[0], -1).sum(dim=1)


def masked_mean_per_image(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean of masked values per image; empty masks yield NaN."""
    if mask is None:
        return values.reshape(values.shape[0], -1).mean(dim=1)
    count = masked_count_per_image(mask, values)
    total = masked_sum_per_image(values, mask)
    mean = total / count.clamp_min(1.0)
    return torch.where(count > 0, mean, torch.full_like(mean, float("nan")))


def masked_std_per_image(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    min_count: int = 1,
) -> torch.Tensor:
    """Population std of masked values per image; small masks yield NaN."""
    if mask is None:
        flat = values.reshape(values.shape[0], -1)
        count = torch.full(
            (values.shape[0],),
            float(flat.shape[1]),
            dtype=torch.float32,
            device=values.device,
        )
        std = torch.std(flat, dim=1, unbiased=False)
        return torch.where(count >= float(min_count), std, torch.full_like(std, float("nan")))
    count = masked_count_per_image(mask, values)
    mean = masked_sum_per_image(values, mask) / count.clamp_min(1.0)
    centered_sq = torch.where(mask, (values - mean.view(-1, 1, 1)) ** 2, torch.zeros_like(values))
    var = centered_sq.reshape(values.shape[0], -1).sum(dim=1) / count.clamp_min(1.0)
    std = torch.sqrt(var)
    return torch.where(count >= float(min_count), std, torch.full_like(std, float("nan")))


def masked_quantile_per_image(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    q: float,
) -> torch.Tensor:
    """Linear-interpolated quantile of masked values per image; empty masks yield NaN."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")
    if mask is None:
        return torch.quantile(values.reshape(values.shape[0], -1), q, dim=1)
    if values.shape != mask.shape:
        raise ValueError(f"values {values.shape} and mask {mask.shape} must match")

    B = values.shape[0]
    flat_values = values.reshape(B, -1)
    flat_mask = mask.reshape(B, -1)
    count = flat_mask.sum(dim=1)
    sorted_values = flat_values.masked_fill(~flat_mask, float("inf")).sort(dim=1).values

    safe_count = count.clamp_min(1).to(torch.float32)
    pos = q * (safe_count - 1.0)
    lower = torch.floor(pos).to(torch.long)
    upper = torch.ceil(pos).to(torch.long)
    frac = (pos - lower.to(torch.float32)).to(values.dtype)
    rows = torch.arange(B, device=values.device)

    lower_values = sorted_values[rows, lower]
    upper_values = sorted_values[rows, upper]
    quantile = lower_values + (upper_values - lower_values) * frac
    return torch.where(count > 0, quantile, torch.full_like(quantile, float("nan")))


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
      pixels so that no flat pixel sits within the spatial support of a
      signal pixel (matches the Section 1.1 protocol).

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
    """Unit normal `n_hat = grad_true / |grad_true|` on the truth field.

    Returned as `(n_x, n_y)`, each `(B, H, W)`. Pixels with magnitude below
    `eps` are returned as `(0, 0)` rather than left as NaN - callers must
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


def ridge_mask_from_truth(
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    step: float = 1.0,
) -> torch.Tensor:
    """Non-max-suppress `|grad_true|` along its own normal direction.

    A pixel `p` is a ridge pixel iff
        |grad_true|(p) >= |grad_true|(p + step*n_hat_p)  and
        |grad_true|(p) >= |grad_true|(p - step*n_hat_p)
    where `n_hat_p` is the unit normal at `p`. The two neighbour values are
    bilinear-sampled via `grid_sample`, so the test is independent of the
    normal's quadrant.

    Restricted to `signal_mask`; returns `(B, H, W)` bool on the same device.
    Pixels with `|grad_true| == 0` are never ridges.
    """
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if signal_mask.shape != g_x_t.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match (B, H, W) {g_x_t.shape}")
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}")

    B, H, W = g_x_t.shape
    device = g_x_t.device
    mag = magnitude(g_x_t, g_y_t)
    n_x, n_y = unit_normal_from_truth(g_x_t, g_y_t)

    ys = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)

    def _sample_neighbor(sign: float) -> torch.Tensor:
        sample_y = ys + sign * step * n_y
        sample_x = xs + sign * step * n_x
        norm_x = 2.0 * sample_x / max(W - 1, 1) - 1.0
        norm_y = 2.0 * sample_y / max(H - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H, W, 2)
        sampled = F.grid_sample(
            mag.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(1)

    mag_plus = _sample_neighbor(+1.0)
    mag_minus = _sample_neighbor(-1.0)
    is_local_max = (mag >= mag_plus) & (mag >= mag_minus) & (mag > 0)
    return signal_mask & is_local_max
