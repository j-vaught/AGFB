"""Shared softened half-bar junction renderer."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from agfb_generators.base import (
    Frame,
    Numeric,
    as_batch,
    coord_grid,
    gauss_Phi,
    gauss_phi,
    infer_batch_size,
    pack,
)


def junction_frame(
    height: int,
    width: int,
    *,
    angles_rad: Sequence[Numeric],
    arm_width_px: Numeric,
    x0: Numeric = 0.0,
    y0: Numeric = 0.0,
    contrast: Numeric = 1.0,
    sigma_e: Numeric = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Frame:
    """Render a smooth union of half-infinite softened bar arms.

    The arm endpoint gates fade to 0.5 exactly at the junction point. A
    softened circular cap with radius `arm_width_px / 2` is included in the
    union so connected junctions do not develop a low-intensity notch at the
    shared endpoint.
    """
    device = device or torch.device("cpu")
    B = infer_batch_size(*angles_rad, arm_width_px, x0, y0, contrast, sigma_e)
    xx, yy = coord_grid(height, width, device, dtype)

    alphas = [as_batch(alpha, B, device, dtype) for alpha in angles_rad]
    w = as_batch(arm_width_px, B, device, dtype)
    x0_b = as_batch(x0, B, device, dtype)
    y0_b = as_batch(y0, B, device, dtype)
    c = as_batch(contrast, B, device, dtype)
    s = as_batch(sigma_e, B, device, dtype)

    dx = xx - x0_b
    dy = yy - y0_b
    half_w = w / 2.0

    arm_masks: list[torch.Tensor] = []
    arm_gx: list[torch.Tensor] = []
    arm_gy: list[torch.Tensor] = []

    for alpha in alphas:
        ex = torch.cos(alpha)
        ey = torch.sin(alpha)
        mx = -ey
        my = ex

        t = dx * ex + dy * ey
        q = dx * mx + dy * my
        qp = (q + half_w) / s
        qm = (q - half_w) / s
        ts = t / s

        R = gauss_Phi(qp) - gauss_Phi(qm)
        H = gauss_Phi(ts)
        A = R * H

        dR_dq = (gauss_phi(qp) - gauss_phi(qm)) / s
        dH_dt = gauss_phi(ts) / s
        gradA_x = H * dR_dq * mx + R * dH_dt * ex
        gradA_y = H * dR_dq * my + R * dH_dt * ey

        arm_masks.append(A)
        arm_gx.append(gradA_x)
        arm_gy.append(gradA_y)

    cap_radius = half_w
    cap_distance = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-12)
    cap_arg = (cap_radius - cap_distance) / s
    cap_mask = gauss_Phi(cap_arg)
    cap_derivative = -(gauss_phi(cap_arg) / s)
    cap_gx = cap_derivative * (dx / cap_distance)
    cap_gy = cap_derivative * (dy / cap_distance)

    arm_masks.append(cap_mask)
    arm_gx.append(cap_gx)
    arm_gy.append(cap_gy)

    one_minus = [1.0 - A for A in arm_masks]
    prod_all = torch.ones_like(arm_masks[0])
    for term in one_minus:
        prod_all = prod_all * term

    gradU_x = torch.zeros_like(prod_all)
    gradU_y = torch.zeros_like(prod_all)
    for i in range(len(arm_masks)):
        prod_except = torch.ones_like(prod_all)
        for j, term in enumerate(one_minus):
            if i != j:
                prod_except = prod_except * term
        gradU_x = gradU_x + arm_gx[i] * prod_except
        gradU_y = gradU_y + arm_gy[i] * prod_except

    U = 1.0 - prod_all
    I = c * U
    gx = c * gradU_x
    gy = c * gradU_y
    return pack(I, gx, gy)
