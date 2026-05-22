"""Triton evaluator for full-image pixel metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch

from agfb_metrics.metrics.base import check_grad_pair
from agfb_metrics.metrics.evaluator import PIXEL_METRICS, MetricName

_TRITON_ERROR = "TritonPixelEvaluator requires Triton and CUDA tensors"
_STATS = 11

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ModuleNotFoundError:
    triton = None
    tl = None
    libdevice = None


if triton is not None and tl is not None and libdevice is not None:

    @triton.jit
    def _pixel_partials_kernel(
        g_x,
        g_y,
        g_x_t,
        g_y_t,
        partials,
        pixels_per_image: tl.constexpr,
        partial_blocks: tl.constexpr,
        block_size: tl.constexpr,
        need_count: tl.constexpr,
        need_err_sum: tl.constexpr,
        need_mag_t_sum: tl.constexpr,
        need_angle: tl.constexpr,
        need_tangent: tl.constexpr,
        need_mag_f_sum: tl.constexpr,
        need_mag_f_sq_sum: tl.constexpr,
        need_max_err: tl.constexpr,
        need_max_mag: tl.constexpr,
    ):
        image_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block_size + tl.arange(0, block_size)
        mask = offsets < pixels_per_image
        base = image_id * pixels_per_image + offsets

        gx = tl.load(g_x + base, mask=mask, other=0.0)
        gy = tl.load(g_y + base, mask=mask, other=0.0)

        out_base = (image_id * partial_blocks + block_id) * 11

        if need_count:
            active = tl.where(mask, 1.0, 0.0)
            tl.store(partials + out_base + 0, tl.sum(active, axis=0))

        if need_err_sum or need_mag_t_sum or need_angle or need_tangent or need_max_err:
            gxt = tl.load(g_x_t + base, mask=mask, other=0.0)
            gyt = tl.load(g_y_t + base, mask=mask, other=0.0)

        if need_err_sum or need_max_err:
            err_x = gx - gxt
            err_y = gy - gyt
            err_sq = err_x * err_x + err_y * err_y
            if need_err_sum:
                tl.store(partials + out_base + 1, tl.sum(tl.where(mask, err_sq, 0.0), axis=0))
            if need_max_err:
                tl.store(
                    partials + out_base + 9, tl.max(tl.where(mask, tl.sqrt(err_sq), 0.0), axis=0)
                )

        if need_mag_t_sum or need_angle or need_tangent:
            mag_t_sq = gxt * gxt + gyt * gyt
            mag_t = tl.sqrt(mag_t_sq)
            if need_mag_t_sum:
                tl.store(partials + out_base + 2, tl.sum(tl.where(mask, mag_t, 0.0), axis=0))

        if need_angle or need_mag_f_sum or need_mag_f_sq_sum or need_max_mag:
            mag_f_sq = gx * gx + gy * gy
            if need_mag_f_sq_sum:
                tl.store(partials + out_base + 8, tl.sum(tl.where(mask, mag_f_sq, 0.0), axis=0))

        if need_angle or need_mag_f_sum or need_max_mag:
            mag_f = tl.sqrt(mag_f_sq)
            if need_mag_f_sum:
                tl.store(partials + out_base + 7, tl.sum(tl.where(mask, mag_f, 0.0), axis=0))
            if need_max_mag:
                tl.store(partials + out_base + 10, tl.max(tl.where(mask, mag_f, 0.0), axis=0))

        if need_angle or need_tangent:
            dot = gx * gxt + gy * gyt

        if need_angle:
            valid_angle = mask & (mag_f > 1.0e-12) & (mag_t > 1.0e-12)
            denom = tl.maximum(mag_f * mag_t, 1.0e-12)
            cos_theta = dot / denom
            cos_theta = tl.minimum(tl.maximum(cos_theta, -1.0), 1.0)
            theta_deg = libdevice.acos(cos_theta) * 57.29577951308232
            theta_deg = tl.where(valid_angle, theta_deg, 0.0)
            tl.store(partials + out_base + 3, tl.sum(theta_deg, axis=0))
            tl.store(partials + out_base + 4, tl.sum(tl.where(valid_angle, 1.0, 0.0), axis=0))

        if need_tangent:
            inv_mag_t = tl.where(mag_t > 1.0e-12, 1.0 / tl.maximum(mag_t, 1.0e-12), 0.0)
            g_n_sq = (dot * inv_mag_t) * (dot * inv_mag_t)
            cross = gy * gxt - gx * gyt
            g_t_sq = (cross * inv_mag_t) * (cross * inv_mag_t)
            tl.store(partials + out_base + 5, tl.sum(tl.where(mask, g_n_sq, 0.0), axis=0))
            tl.store(partials + out_base + 6, tl.sum(tl.where(mask, g_t_sq, 0.0), axis=0))

    @triton.jit
    def _tail_histogram_kernel(
        g_x,
        g_y,
        g_x_t,
        g_y_t,
        err_hist,
        mag_hist,
        max_err,
        max_mag,
        pixels_per_image: tl.constexpr,
        bins: tl.constexpr,
        block_size: tl.constexpr,
        use_err: tl.constexpr,
        use_mag: tl.constexpr,
    ):
        image_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offsets = block_id * block_size + tl.arange(0, block_size)
        mask = offsets < pixels_per_image
        base = image_id * pixels_per_image + offsets

        gx = tl.load(g_x + base, mask=mask, other=0.0)
        gy = tl.load(g_y + base, mask=mask, other=0.0)
        hist_base = image_id * bins

        if use_err:
            gxt = tl.load(g_x_t + base, mask=mask, other=0.0)
            gyt = tl.load(g_y_t + base, mask=mask, other=0.0)
            err_x = gx - gxt
            err_y = gy - gyt
            err_mag = tl.sqrt(err_x * err_x + err_y * err_y)
            err_max = tl.maximum(tl.load(max_err + image_id), 1.0e-30)
            err_bin = tl.minimum((err_mag * (bins / err_max)).to(tl.int32), bins - 1)
            tl.atomic_add(err_hist + hist_base + err_bin, 1, sem="relaxed", mask=mask)

        if use_mag:
            mag = tl.sqrt(gx * gx + gy * gy)
            mag_max = tl.maximum(tl.load(max_mag + image_id), 1.0e-30)
            mag_bin = tl.minimum((mag * (bins / mag_max)).to(tl.int32), bins - 1)
            tl.atomic_add(mag_hist + hist_base + mag_bin, 1, sem="relaxed", mask=mask)


class TritonPixelEvaluator:
    """Full-image selected-metric evaluator for repeated CUDA batches."""

    def __init__(
        self,
        *,
        metrics: Sequence[MetricName] = PIXEL_METRICS,
        sigma_n: float | None = None,
        tail_vector_q: float = 0.95,
        tail_spurious_q: float = 0.99,
        tail_mode: Literal["exact", "histogram"] = "exact",
        tail_bins: int = 4096,
        block_size: int = 1024,
    ) -> None:
        selected = tuple(metrics)
        unknown = sorted(set(selected) - set(PIXEL_METRICS))
        if unknown:
            raise ValueError(f"TritonPixelEvaluator requires pixel metrics; got {unknown}")
        if "noise_gain" in selected and sigma_n is None:
            raise ValueError("sigma_n is required for noise_gain")
        if not 0.0 < tail_vector_q < 1.0:
            raise ValueError(f"tail_vector_q must be in (0, 1); got {tail_vector_q}")
        if not 0.0 < tail_spurious_q < 1.0:
            raise ValueError(f"tail_spurious_q must be in (0, 1); got {tail_spurious_q}")
        if tail_mode not in {"exact", "histogram"}:
            raise ValueError(f"tail_mode must be 'exact' or 'histogram'; got {tail_mode}")
        if tail_bins <= 1 or tail_bins & (tail_bins - 1) != 0:
            raise ValueError(f"tail_bins must be a power of two greater than one; got {tail_bins}")
        if block_size < 512 or block_size & (block_size - 1) != 0:
            raise ValueError(f"block_size must be a power of two at least 512; got {block_size}")

        self.metrics = selected
        self.sigma_n = float(sigma_n) if sigma_n is not None else None
        self.tail_vector_q = tail_vector_q
        self.tail_spurious_q = tail_spurious_q
        self.tail_mode = tail_mode
        self.tail_bins = tail_bins
        self.block_size = block_size

    def __call__(
        self,
        g_x: torch.Tensor,
        g_y: torch.Tensor,
        g_x_t: torch.Tensor,
        g_y_t: torch.Tensor,
        *,
        signal_mask: torch.Tensor | None,
        flat_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        check_grad_pair(g_x, g_y, name="filter gradient")
        check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
        if g_x_t.shape != g_x.shape:
            raise ValueError(f"ground-truth tensors {g_x_t.shape} must match filter {g_x.shape}")
        if signal_mask is not None or flat_mask is not None:
            raise ValueError("TritonPixelEvaluator evaluates full-image pixel metrics")
        if not g_x.is_cuda:
            raise RuntimeError(_TRITON_ERROR)
        if triton is None or _pixel_partials_kernel is None:
            raise RuntimeError(_TRITON_ERROR)

        g_x = g_x.contiguous()
        g_y = g_y.contiguous()
        g_x_t = g_x_t.contiguous()
        g_y_t = g_y_t.contiguous()

        B, H, W = g_x.shape
        pixels_per_image = H * W
        values: dict[str, torch.Tensor] = {}
        selected_set = set(self.metrics)
        needs_tail_vector_hist = (
            self.tail_mode == "histogram" and "tail_vector_error" in selected_set
        )
        needs_tail_spurious_hist = (
            self.tail_mode == "histogram" and "tail_spurious_grad" in selected_set
        )
        needs_reduce = bool(
            selected_set
            & {
                "nrmse",
                "angular_mae",
                "tangential_normal_leak",
                "magnitude_bias",
                "noise_gain",
            }
        )
        needs_tail_hist = self.tail_mode == "histogram" and bool(
            selected_set & {"tail_vector_error", "tail_spurious_grad"}
        )
        needs_partials = needs_reduce or needs_tail_hist
        partial_blocks = triton.cdiv(pixels_per_image, self.block_size)
        partials = None
        sums = None
        if needs_partials:
            need_count = bool(selected_set & {"nrmse", "tangential_normal_leak", "noise_gain"})
            need_err_sum = "nrmse" in selected_set
            need_mag_t_sum = bool(selected_set & {"nrmse", "magnitude_bias"})
            need_angle = "angular_mae" in selected_set
            need_tangent = "tangential_normal_leak" in selected_set
            need_mag_f_sum = bool(selected_set & {"magnitude_bias", "noise_gain"})
            need_mag_f_sq_sum = "noise_gain" in selected_set
            partials = torch.empty(
                (B, partial_blocks, _STATS),
                dtype=torch.float32,
                device=g_x.device,
            )

            _pixel_partials_kernel[(B, partial_blocks)](
                g_x,
                g_y,
                g_x_t,
                g_y_t,
                partials,
                pixels_per_image,
                partial_blocks,
                self.block_size,
                need_count,
                need_err_sum,
                need_mag_t_sum,
                need_angle,
                need_tangent,
                need_mag_f_sum,
                need_mag_f_sq_sum,
                needs_tail_vector_hist,
                needs_tail_spurious_hist,
                num_warps=8,
            )

            sums = partials[:, :, :9].sum(dim=1)

        err_hist = None
        mag_hist = None
        if needs_tail_hist:
            assert partials is not None
            max_err = (
                partials[:, :, 9].amax(dim=1)
                if needs_tail_vector_hist
                else torch.empty((B,), dtype=torch.float32, device=g_x.device)
            )
            max_mag = (
                partials[:, :, 10].amax(dim=1)
                if needs_tail_spurious_hist
                else torch.empty((B,), dtype=torch.float32, device=g_x.device)
            )
            err_hist = torch.zeros((B, self.tail_bins), dtype=torch.int32, device=g_x.device)
            mag_hist = torch.zeros((B, self.tail_bins), dtype=torch.int32, device=g_x.device)
            _tail_histogram_kernel[(B, partial_blocks)](
                g_x,
                g_y,
                g_x_t,
                g_y_t,
                err_hist,
                mag_hist,
                max_err,
                max_mag,
                pixels_per_image,
                self.tail_bins,
                self.block_size,
                needs_tail_vector_hist,
                needs_tail_spurious_hist,
                num_warps=8,
            )

        if "nrmse" in selected_set:
            assert sums is not None
            count = sums[:, 0].clamp_min(1.0)
            sum_err_sq = sums[:, 1]
            sum_mag_t = sums[:, 2]
            values["nrmse"] = torch.sqrt(sum_err_sq / count) / (sum_mag_t / count).clamp_min(1e-30)

        if "angular_mae" in selected_set:
            assert sums is not None
            sum_theta = sums[:, 3]
            count_theta = sums[:, 4]
            angular = sum_theta / count_theta.clamp_min(1.0)
            values["angular_mae"] = torch.where(
                count_theta > 0, angular, torch.full_like(angular, float("nan"))
            )

        if "tail_vector_error" in selected_set:
            if err_hist is None:
                err_mag = torch.sqrt((g_x - g_x_t) ** 2 + (g_y - g_y_t) ** 2)
                values["tail_vector_error"] = torch.quantile(
                    err_mag.reshape(B, -1), self.tail_vector_q, dim=1
                )
            else:
                assert partials is not None
                max_err = partials[:, :, 9].amax(dim=1)
                values["tail_vector_error"] = _histogram_quantile(
                    err_hist, max_err, self.tail_vector_q
                )

        if "tangential_normal_leak" in selected_set:
            assert sums is not None
            count = sums[:, 0].clamp_min(1.0)
            sum_gn_sq = sums[:, 5]
            sum_gt_sq = sums[:, 6]
            e_n = sum_gn_sq / count
            e_t = sum_gt_sq / count
            finite = 10.0 * torch.log10(e_t / e_n)
            value = torch.where(
                e_n < 1e-30, torch.where(e_t < 1e-30, -torch.inf, torch.inf), finite
            )
            values["tangential_normal_leak"] = torch.where(
                (e_n >= 1e-30) & (e_t < 1e-30), -torch.inf, value
            )

        if "magnitude_bias" in selected_set:
            assert sums is not None
            sum_mag_t = sums[:, 2]
            sum_mag_f = sums[:, 7]
            values["magnitude_bias"] = sum_mag_f / sum_mag_t.clamp_min(1e-30) - 1.0

        if "noise_gain" in selected_set:
            assert self.sigma_n is not None
            assert sums is not None
            count = sums[:, 0].clamp_min(1.0)
            sum_mag_f = sums[:, 7]
            sum_mag_f_sq = sums[:, 8]
            mean = sum_mag_f / count
            var = (sum_mag_f_sq / count - mean * mean).clamp_min(0.0)
            std = torch.sqrt(var)
            values["noise_gain"] = torch.where(
                count >= 2.0,
                std / self.sigma_n,
                torch.full_like(std, float("nan")),
            )

        if "tail_spurious_grad" in selected_set:
            if mag_hist is None:
                mag_f = torch.sqrt(g_x * g_x + g_y * g_y)
                values["tail_spurious_grad"] = torch.quantile(
                    mag_f.reshape(B, -1), self.tail_spurious_q, dim=1
                )
            else:
                assert partials is not None
                max_mag = partials[:, :, 10].amax(dim=1)
                values["tail_spurious_grad"] = _histogram_quantile(
                    mag_hist, max_mag, self.tail_spurious_q
                )

        return {name: values[name] for name in self.metrics}


def is_triton_pixel_available() -> bool:
    return bool(triton is not None and torch.cuda.is_available())


def _histogram_quantile(hist: torch.Tensor, max_values: torch.Tensor, q: float) -> torch.Tensor:
    bins = hist.shape[1]
    cumulative = hist.to(torch.float32).cumsum(dim=1)
    total = cumulative[:, -1].clamp_min(1.0)
    target = q * (total - 1.0) + 1.0
    reached = cumulative >= target.view(-1, 1)
    bin_idx = reached.to(torch.int64).argmax(dim=1)
    rows = torch.arange(hist.shape[0], device=hist.device)
    prev_idx = (bin_idx - 1).clamp_min(0)
    previous = torch.where(
        bin_idx > 0,
        cumulative[rows, prev_idx],
        torch.zeros_like(target),
    )
    bin_count = hist[rows, bin_idx].to(torch.float32).clamp_min(1.0)
    frac = ((target - previous) / bin_count).clamp(0.0, 1.0)
    width = max_values / float(bins)
    value = (bin_idx.to(torch.float32) + frac) * width
    return torch.where(max_values > 0, value, torch.zeros_like(value))


def _triton_state_for_tests() -> dict[str, Any]:
    return {"triton": triton, "kernel": globals().get("_pixel_partials_kernel")}
