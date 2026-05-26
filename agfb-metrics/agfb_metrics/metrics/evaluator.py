"""Evaluate selected metrics for one gradient batch with shared intermediates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch

from agfb_metrics.metrics._cross_edge_profile import cross_edge_profile
from agfb_metrics.metrics.base import (
    check_grad_pair,
    magnitude,
    masked_count_per_image,
    masked_mean_per_image,
    masked_quantile_per_image,
    masked_std_per_image,
    masked_sum_per_image,
)
from agfb_metrics.metrics.edge_fwhm import _fwhm_profiles
from agfb_metrics.metrics.localization_offset import _peak_positions
from agfb_metrics.metrics.sidelobe_ratio import _sidelobe_ratios

MetricName = Literal[
    "nrmse",
    "angular_mae",
    "tail_vector_error",
    "localization_offset",
    "tangential_normal_leak",
    "magnitude_bias",
    "edge_fwhm",
    "sidelobe_ratio",
    "noise_gain",
    "tail_spurious_grad",
]

ALL_METRICS: tuple[MetricName, ...] = (
    "nrmse",
    "angular_mae",
    "tail_vector_error",
    "localization_offset",
    "tangential_normal_leak",
    "magnitude_bias",
    "edge_fwhm",
    "sidelobe_ratio",
    "noise_gain",
    "tail_spurious_grad",
)

PIXEL_METRICS: tuple[MetricName, ...] = (
    "nrmse",
    "angular_mae",
    "tail_vector_error",
    "tangential_normal_leak",
    "magnitude_bias",
    "noise_gain",
    "tail_spurious_grad",
)

_EDGE_PROFILE_METRICS = {"localization_offset", "edge_fwhm", "sidelobe_ratio"}


class MetricEvaluator:
    """Reusable selected-metric evaluator for repeated same-shape batches."""

    def __init__(
        self,
        *,
        metrics: Sequence[MetricName] = ALL_METRICS,
        sigma_n: float | None = None,
        r_max: float = 16.0,
        step: float = 0.5,
        tail_vector_q: float = 0.95,
        tail_spurious_q: float = 0.99,
        use_compile: bool = False,
        compile_mode: str | None = "reduce-overhead",
    ) -> None:
        selected = tuple(metrics)
        unknown = sorted(set(selected) - set(ALL_METRICS))
        if unknown:
            raise ValueError(f"unknown metrics: {unknown}")
        if use_compile and any(name in _EDGE_PROFILE_METRICS for name in selected):
            raise ValueError("use_compile requires pixel metrics")

        self.metrics = selected
        self.sigma_n = sigma_n
        self.r_max = r_max
        self.step = step
        self.tail_vector_q = tail_vector_q
        self.tail_spurious_q = tail_spurious_q

        def evaluate_tuple(
            g_x: torch.Tensor,
            g_y: torch.Tensor,
            g_x_t: torch.Tensor,
            g_y_t: torch.Tensor,
            signal_mask: torch.Tensor | None,
            flat_mask: torch.Tensor | None,
        ) -> tuple[torch.Tensor, ...]:
            out = evaluate_metrics(
                g_x,
                g_y,
                g_x_t,
                g_y_t,
                metrics=selected,
                signal_mask=signal_mask,
                flat_mask=flat_mask,
                sigma_n=sigma_n,
                r_max=r_max,
                step=step,
                tail_vector_q=tail_vector_q,
                tail_spurious_q=tail_spurious_q,
            )
            return tuple(out[name] for name in selected)

        self._evaluate_tuple: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
            ],
            tuple[torch.Tensor, ...],
        ]
        if use_compile:
            self._evaluate_tuple = torch.compile(evaluate_tuple, mode=compile_mode)
        else:
            self._evaluate_tuple = evaluate_tuple

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
        values = self._evaluate_tuple(g_x, g_y, g_x_t, g_y_t, signal_mask, flat_mask)
        return dict(zip(self.metrics, values, strict=True))


def _check_optional_mask(mask: torch.Tensor | None, shape: torch.Size, name: str) -> None:
    if mask is not None and mask.shape != shape:
        raise ValueError(f"{name} {mask.shape} must match (B, H, W) {shape}")


def _profile_mean(values: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(values)
    count = valid.sum()
    total = torch.where(valid, values, torch.zeros_like(values)).sum()
    return torch.where(
        count > 0,
        total / count.clamp_min(1),
        torch.tensor(float("nan"), dtype=values.dtype, device=values.device),
    )


def evaluate_metrics(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    *,
    metrics: Sequence[MetricName] = ALL_METRICS,
    signal_mask: torch.Tensor | None,
    flat_mask: torch.Tensor | None,
    sigma_n: float | None = None,
    r_max: float = 16.0,
    step: float = 0.5,
    tail_vector_q: float = 0.95,
    tail_spurious_q: float = 0.99,
) -> dict[str, torch.Tensor]:
    """Evaluate selected metrics for one `(B, H, W)` gradient batch.

    Pass `signal_mask=None` or `flat_mask=None` to evaluate the corresponding
    signal or flat-region metrics over every pixel. Profile metrics require an
    explicit `signal_mask` because all-pixel profile sampling is usually too
    large for full-field gradient images.
    """
    check_grad_pair(g_x, g_y, name="filter gradient")
    check_grad_pair(g_x_t, g_y_t, name="ground-truth gradient")
    if g_x_t.shape != g_x.shape:
        raise ValueError(f"ground-truth tensors {g_x_t.shape} must match filter {g_x.shape}")
    _check_optional_mask(signal_mask, g_x.shape, "signal_mask")
    _check_optional_mask(flat_mask, g_x.shape, "flat_mask")

    selected = tuple(metrics)
    unknown = sorted(set(selected) - set(ALL_METRICS))
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")
    if "noise_gain" in selected and sigma_n is None:
        raise ValueError("sigma_n is required for noise_gain")
    sigma_value = float(sigma_n) if sigma_n is not None else 0.0
    if signal_mask is None and any(name in _EDGE_PROFILE_METRICS for name in selected):
        raise ValueError("profile metrics require an explicit signal_mask")

    out: dict[str, torch.Tensor] = {}
    selected_set = set(selected)

    needs_mag_f = bool(
        selected_set
        & {
            "angular_mae",
            "localization_offset",
            "magnitude_bias",
            "edge_fwhm",
            "sidelobe_ratio",
            "noise_gain",
            "tail_spurious_grad",
        }
    )
    needs_mag_t = bool(
        selected_set & {"nrmse", "angular_mae", "tangential_normal_leak", "magnitude_bias"}
    )
    mag_f = magnitude(g_x, g_y) if needs_mag_f else None
    mag_t = magnitude(g_x_t, g_y_t) if needs_mag_t else None
    needs_error = bool(selected_set & {"nrmse", "tail_vector_error"})
    err_x = g_x - g_x_t if needs_error else None
    err_y = g_y - g_y_t if needs_error else None
    needs_dot = bool(selected_set & {"angular_mae", "tangential_normal_leak"})
    dot_ft = g_x * g_x_t + g_y * g_y_t if needs_dot else None

    if "nrmse" in selected_set:
        if mag_t is None:
            mag_t = magnitude(g_x_t, g_y_t)
        assert err_x is not None and err_y is not None
        err_sq = err_x * err_x + err_y * err_y
        count = masked_count_per_image(signal_mask, err_sq)
        num = torch.sqrt(masked_sum_per_image(err_sq, signal_mask) / count.clamp_min(1.0))
        den = masked_mean_per_image(mag_t, signal_mask).clamp_min(1e-30)
        value = num / den
        out["nrmse"] = torch.where(count > 0, value, torch.full_like(value, float("nan")))

    if "angular_mae" in selected_set:
        if mag_f is None:
            mag_f = magnitude(g_x, g_y)
        if mag_t is None:
            mag_t = magnitude(g_x_t, g_y_t)
        assert dot_ft is not None
        denom = (mag_f * mag_t).clamp_min(1e-12)
        cos_theta = (dot_ft / denom).clamp(-1.0, 1.0)
        theta_deg = torch.arccos(cos_theta) * (180.0 / torch.pi)
        valid = (mag_f > 1e-12) & (mag_t > 1e-12)
        if signal_mask is not None:
            valid = signal_mask & valid
        out["angular_mae"] = masked_mean_per_image(theta_deg, valid)

    if "tail_vector_error" in selected_set:
        assert err_x is not None and err_y is not None
        err_mag = torch.sqrt(err_x * err_x + err_y * err_y)
        out["tail_vector_error"] = masked_quantile_per_image(err_mag, signal_mask, tail_vector_q)

    if "tangential_normal_leak" in selected_set:
        if mag_t is None:
            mag_t = magnitude(g_x_t, g_y_t)
        assert dot_ft is not None
        safe = mag_t.clamp_min(1e-12)
        inv_mag_t = torch.where(mag_t > 1e-12, safe.reciprocal(), torch.zeros_like(safe))
        g_n_sq = (dot_ft * inv_mag_t) ** 2
        cross_ft = g_y * g_x_t - g_x * g_y_t
        g_t_sq = (cross_ft * inv_mag_t) ** 2
        count = masked_count_per_image(signal_mask, g_n_sq)
        e_n = masked_sum_per_image(g_n_sq, signal_mask) / count.clamp_min(1.0)
        e_t = masked_sum_per_image(g_t_sq, signal_mask) / count.clamp_min(1.0)
        finite = 10.0 * torch.log10(e_t / e_n)
        value = torch.where(e_n < 1e-30, torch.where(e_t < 1e-30, -torch.inf, torch.inf), finite)
        value = torch.where((e_n >= 1e-30) & (e_t < 1e-30), -torch.inf, value)
        out["tangential_normal_leak"] = torch.where(
            count > 0, value, torch.full_like(value, float("nan"))
        )

    if "magnitude_bias" in selected_set:
        if mag_f is None:
            mag_f = magnitude(g_x, g_y)
        if mag_t is None:
            mag_t = magnitude(g_x_t, g_y_t)
        count = masked_count_per_image(signal_mask, mag_f)
        num = masked_sum_per_image(mag_f, signal_mask)
        den = masked_sum_per_image(mag_t, signal_mask).clamp_min(1e-30)
        value = num / den - 1.0
        out["magnitude_bias"] = torch.where(count > 0, value, torch.full_like(value, float("nan")))

    if "noise_gain" in selected_set:
        if mag_f is None:
            mag_f = magnitude(g_x, g_y)
        out["noise_gain"] = masked_std_per_image(mag_f, flat_mask, min_count=2) / sigma_value

    if "tail_spurious_grad" in selected_set:
        if mag_f is None:
            mag_f = magnitude(g_x, g_y)
        out["tail_spurious_grad"] = masked_quantile_per_image(mag_f, flat_mask, tail_spurious_q)

    if selected_set & _EDGE_PROFILE_METRICS:
        if mag_f is None:
            mag_f = magnitude(g_x, g_y)
        assert signal_mask is not None
        filt_profiles, t, _ = cross_edge_profile(
            mag_f, g_x_t, g_y_t, signal_mask, r_max=r_max, step=step
        )

        truth_profiles = None
        if "localization_offset" in selected_set:
            if mag_t is None:
                mag_t = magnitude(g_x_t, g_y_t)
            truth_profiles, _, _ = cross_edge_profile(
                mag_t, g_x_t, g_y_t, signal_mask, r_max=r_max, step=step
            )

        B = g_x.shape[0]
        if "localization_offset" in selected_set:
            loc = torch.empty(B, dtype=torch.float32, device=g_x.device)
            assert truth_profiles is not None
            for i in range(B):
                if filt_profiles[i].shape[0] == 0:
                    loc[i] = float("nan")
                    continue
                t_filter = _peak_positions(filt_profiles[i], t, step)
                t_truth = _peak_positions(truth_profiles[i], t, step)
                loc[i] = (t_filter - t_truth).abs().mean()
            out["localization_offset"] = loc

        if "edge_fwhm" in selected_set:
            fwhm = torch.empty(B, dtype=torch.float32, device=g_x.device)
            for i in range(B):
                if filt_profiles[i].shape[0] == 0:
                    fwhm[i] = float("nan")
                    continue
                fwhm[i] = _profile_mean(_fwhm_profiles(filt_profiles[i], t))
            out["edge_fwhm"] = fwhm

        if "sidelobe_ratio" in selected_set:
            sidelobe = torch.empty(B, dtype=torch.float32, device=g_x.device)
            for i in range(B):
                if filt_profiles[i].shape[0] == 0:
                    sidelobe[i] = float("nan")
                    continue
                ratios = _sidelobe_ratios(filt_profiles[i])
                valid = ~torch.isnan(ratios)
                log_ratios = torch.where(valid, torch.log10(ratios), torch.zeros_like(ratios))
                count = valid.sum()
                sidelobe[i] = torch.where(
                    count > 0,
                    20.0 * log_ratios.sum() / count.clamp_min(1),
                    torch.tensor(float("nan"), dtype=torch.float32, device=g_x.device),
                )
            out["sidelobe_ratio"] = sidelobe

    return {name: out[name] for name in selected}


def evaluate_all_metrics(
    g_x: torch.Tensor,
    g_y: torch.Tensor,
    g_x_t: torch.Tensor,
    g_y_t: torch.Tensor,
    *,
    signal_mask: torch.Tensor | None,
    flat_mask: torch.Tensor | None,
    sigma_n: float,
    r_max: float = 16.0,
    step: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Evaluate all metric names in `ALL_METRICS` for one gradient batch."""
    return evaluate_metrics(
        g_x,
        g_y,
        g_x_t,
        g_y_t,
        metrics=ALL_METRICS,
        signal_mask=signal_mask,
        flat_mask=flat_mask,
        sigma_n=sigma_n,
        r_max=r_max,
        step=step,
    )
