"""Shared cross-edge profile sampler for edge-profile metrics.

For each true edge pixel `p` with unit normal `n̂_p`, bilinearly sample a
scalar field along `p + t * n̂_p` for `t in [-r_max, r_max]` with sample
step `step`. Returns one `(N_edge, K)` profile tensor per image.

Important: the signal mask is the §1.1 band mask (all pixels where
`|grad_true|` exceeds threshold), not a thin ridge. For an off-ridge edge
pixel `p`, the profile of `|grad_truth|` peaks at the *signed distance
from p to the true edge crest along the normal*, not at `t = 0`. Localization
offset therefore measures `argmax(filter_profile) - argmax(truth_profile)`;
edge FWHM and side-lobe ratio measure intrinsic shape properties of the filter
profile that do not depend on edge-pixel anchoring.

Edge pixels near the image boundary fall back to `padding_mode='border'`
rather than being dropped — §1.1 generators centre edges in the field of
view, so this is rarely exercised.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from agfb_metrics.base import unit_normal_from_truth


def cross_edge_profile(
    field: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    r_max: float = 16.0,
    step: float = 0.5,
) -> tuple[list[torch.Tensor], torch.Tensor, int]:
    """Sample `field` along the truth-field normal at every edge pixel.

    Args:
        field: (B, H, W) float32 scalar field to sample (typically
            `|grad_filter|` or `|grad_truth|`).
        g_x_t, g_y_t: (B, H, W) truth gradient components; used only to
            define the per-pixel unit normal direction `n̂_p`.
        signal_mask: (B, H, W) bool — true edge pixels.
        r_max, step: cross-edge window half-width (pixels) and sample step.

    Returns `(profiles_per_image, t, t0_index)`:
        * `profiles_per_image[i]` is `(N_edge_i, K)` float32, on `field.device`.
        * `t` is `(K,)` float32, the t-axis from `-r_max` to `+r_max` step `step`.
        * `t0_index` is the integer index in `t` that equals zero (= K // 2).
    """
    if field.ndim != 3:
        raise ValueError(f"field must be (B, H, W); got {tuple(field.shape)}")
    if field.dtype != torch.float32:
        raise ValueError(f"field must be float32; got {field.dtype}")
    if g_x_t.shape != field.shape or g_y_t.shape != field.shape:
        raise ValueError("g_x_t and g_y_t must match field shape")
    if signal_mask.shape != field.shape:
        raise ValueError(f"signal_mask {signal_mask.shape} must match field {field.shape}")
    if r_max <= 0 or step <= 0:
        raise ValueError(f"r_max and step must be positive; got r_max={r_max}, step={step}")

    B, H, W = field.shape
    device = field.device
    n_x, n_y = unit_normal_from_truth(g_x_t, g_y_t)

    n_samples_one_side = int(round(r_max / step))
    t = torch.arange(
        -n_samples_one_side, n_samples_one_side + 1, device=device, dtype=torch.float32
    )
    t = t * step
    K = t.shape[0]
    t0_index = n_samples_one_side

    profiles_per_image: list[torch.Tensor] = []
    for i in range(B):
        m = signal_mask[i]
        idx = m.nonzero(as_tuple=False)  # (N, 2) -> (row, col) = (y, x)
        if idx.shape[0] == 0:
            profiles_per_image.append(torch.empty((0, K), dtype=torch.float32, device=device))
            continue
        p_y = idx[:, 0].to(torch.float32)
        p_x = idx[:, 1].to(torch.float32)
        n_y_p = n_y[i][idx[:, 0], idx[:, 1]]
        n_x_p = n_x[i][idx[:, 0], idx[:, 1]]

        sample_y = p_y.unsqueeze(1) + t.unsqueeze(0) * n_y_p.unsqueeze(1)
        sample_x = p_x.unsqueeze(1) + t.unsqueeze(0) * n_x_p.unsqueeze(1)

        norm_x = 2.0 * sample_x / max(W - 1, 1) - 1.0
        norm_y = 2.0 * sample_y / max(H - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, N, K, 2)

        field_i = field[i].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        sampled = F.grid_sample(
            field_i,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, 1, N, K)
        profiles_per_image.append(sampled.squeeze(0).squeeze(0))
    return profiles_per_image, t, t0_index
