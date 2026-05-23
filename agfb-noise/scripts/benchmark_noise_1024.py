"""Benchmark shipped noise models on a single 1024 x 1024 image."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from agfb_noise import (
    add_dark_current,
    add_dead_pixels,
    add_fixed_pattern,
    add_gamma_speckle,
    add_gaussian,
    add_local_variance,
    add_pepper,
    add_poisson,
    add_poisson_gaussian,
    add_quantization,
    add_random_impulse,
    add_rayleigh,
    add_rician,
    add_salt,
    add_salt_pepper,
    add_speckle,
    add_stripe,
    add_uniform,
)
from agfb_noise.helpers.notebook import synthetic_1024_image

ClampRange = tuple[float, float]
CaseFunc = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class BenchmarkCase:
    noise: str
    setting: str
    call: CaseFunc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def seeded(
    function: Callable[..., torch.Tensor],
    *,
    seed: int,
    **kwargs: Any,
) -> CaseFunc:
    def call(image: torch.Tensor) -> torch.Tensor:
        return function(image, seed=seed, **kwargs)

    return call


def unseeded(function: Callable[..., torch.Tensor], **kwargs: Any) -> CaseFunc:
    def call(image: torch.Tensor) -> torch.Tensor:
        return function(image, **kwargs)

    return call


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90_index = int(round((len(ordered) - 1) * 0.9))
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p90_ms": ordered[p90_index],
    }


def benchmark_case(
    case: BenchmarkCase,
    image: torch.Tensor,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        out = case.call(image)
        torch.cuda.synchronize()
        validate_output(case, image, out)

    wall_values: list[float] = []
    event_values: list[float] = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        out = case.call(image)
        end_event.record()
        torch.cuda.synchronize()
        validate_output(case, image, out)
        wall_values.append((time.perf_counter() - wall_start) * 1000.0)
        event_values.append(float(start_event.elapsed_time(end_event)))

    row: dict[str, Any] = {
        "noise": case.noise,
        "setting": case.setting,
        "repeats": repeats,
        "warmups": warmups,
        "shape": "x".join(str(part) for part in image.shape),
        "dtype": str(image.dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(image.device),
    }
    for key, value in summarize(wall_values).items():
        row[f"wall_{key}"] = value
    for key, value in summarize(event_values).items():
        row[f"cuda_event_{key}"] = value
    return row


def validate_output(case: BenchmarkCase, image: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != image.shape:
        raise RuntimeError(f"{case.noise}/{case.setting} returned {tuple(out.shape)}")


def build_cases(
    *,
    seed: int,
    clamp: ClampRange,
    variance_field: torch.Tensor,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    def add(noise: str, setting: str, call: CaseFunc) -> None:
        cases.append(BenchmarkCase(noise=noise, setting=setting, call=call))

    add("gaussian", "sigma=0.01", seeded(add_gaussian, seed=seed, sigma=0.01, clamp=clamp))
    add("gaussian", "sigma=0.035", seeded(add_gaussian, seed=seed, sigma=0.035, clamp=clamp))
    add("gaussian", "sigma=0.10", seeded(add_gaussian, seed=seed, sigma=0.10, clamp=clamp))

    add(
        "local_variance",
        "variance=1e-4",
        seeded(add_local_variance, seed=seed, variance=0.0001, clamp=clamp),
    )
    add(
        "local_variance",
        "variance=field",
        seeded(add_local_variance, seed=seed, variance=variance_field, clamp=clamp),
    )
    add(
        "local_variance",
        "variance=1e-2",
        seeded(add_local_variance, seed=seed, variance=0.01, clamp=clamp),
    )

    add("uniform", "half_width=0.02", seeded(add_uniform, seed=seed, half_width=0.02, clamp=clamp))
    add("uniform", "half_width=0.06", seeded(add_uniform, seed=seed, half_width=0.06, clamp=clamp))
    add("uniform", "half_width=0.15", seeded(add_uniform, seed=seed, half_width=0.15, clamp=clamp))

    add("poisson", "peak=128", seeded(add_poisson, seed=seed, peak=128.0, clamp=clamp))
    add("poisson", "peak=512", seeded(add_poisson, seed=seed, peak=512.0, clamp=clamp))
    add("poisson", "peak=4096", seeded(add_poisson, seed=seed, peak=4096.0, clamp=clamp))

    add(
        "poisson_gaussian",
        "peak=128 read_sigma=0.005",
        seeded(add_poisson_gaussian, seed=seed, peak=128.0, read_sigma=0.005, clamp=clamp),
    )
    add(
        "poisson_gaussian",
        "peak=512 read_sigma=0.015",
        seeded(add_poisson_gaussian, seed=seed, peak=512.0, read_sigma=0.015, clamp=clamp),
    )
    add(
        "poisson_gaussian",
        "peak=4096 read_sigma=0.03",
        seeded(add_poisson_gaussian, seed=seed, peak=4096.0, read_sigma=0.03, clamp=clamp),
    )

    add(
        "dark_current",
        "rate=1 exposure=0.1",
        seeded(
            add_dark_current,
            seed=seed,
            dark_rate=1.0,
            exposure_time=0.1,
            peak=512.0,
            read_sigma=0.002,
            clamp=clamp,
        ),
    )
    add(
        "dark_current",
        "rate=6 exposure=0.5",
        seeded(
            add_dark_current,
            seed=seed,
            dark_rate=6.0,
            exposure_time=0.5,
            peak=512.0,
            read_sigma=0.004,
            clamp=clamp,
        ),
    )
    add(
        "dark_current",
        "rate=20 exposure=1",
        seeded(
            add_dark_current,
            seed=seed,
            dark_rate=20.0,
            exposure_time=1.0,
            peak=512.0,
            read_sigma=0.008,
            clamp=clamp,
        ),
    )

    add("salt", "amount=0.005", seeded(add_salt, seed=seed, amount=0.005, salt_value=1.0))
    add("salt", "amount=0.02", seeded(add_salt, seed=seed, amount=0.02, salt_value=1.0))
    add("salt", "amount=0.10", seeded(add_salt, seed=seed, amount=0.10, salt_value=1.0))

    add("pepper", "amount=0.005", seeded(add_pepper, seed=seed, amount=0.005, pepper_value=0.0))
    add("pepper", "amount=0.02", seeded(add_pepper, seed=seed, amount=0.02, pepper_value=0.0))
    add("pepper", "amount=0.10", seeded(add_pepper, seed=seed, amount=0.10, pepper_value=0.0))

    add(
        "salt_pepper",
        "amount=0.005",
        seeded(
            add_salt_pepper,
            seed=seed,
            amount=0.005,
            salt_vs_pepper=0.5,
            salt_value=1.0,
            pepper_value=0.0,
        ),
    )
    add(
        "salt_pepper",
        "amount=0.03",
        seeded(
            add_salt_pepper,
            seed=seed,
            amount=0.03,
            salt_vs_pepper=0.5,
            salt_value=1.0,
            pepper_value=0.0,
        ),
    )
    add(
        "salt_pepper",
        "amount=0.10",
        seeded(
            add_salt_pepper,
            seed=seed,
            amount=0.10,
            salt_vs_pepper=0.5,
            salt_value=1.0,
            pepper_value=0.0,
        ),
    )

    add(
        "random_impulse",
        "amount=0.005",
        seeded(add_random_impulse, seed=seed, amount=0.005, low=0.0, high=1.0),
    )
    add(
        "random_impulse",
        "amount=0.03",
        seeded(add_random_impulse, seed=seed, amount=0.03, low=0.0, high=1.0),
    )
    add(
        "random_impulse",
        "amount=0.10",
        seeded(add_random_impulse, seed=seed, amount=0.10, low=0.0, high=1.0),
    )

    add(
        "dead_pixel",
        "amount=0.005 hot=0.05",
        seeded(
            add_dead_pixels,
            seed=seed,
            amount=0.005,
            hot_fraction=0.05,
            dead_value=0.0,
            hot_value=1.0,
        ),
    )
    add(
        "dead_pixel",
        "amount=0.015 hot=0.15",
        seeded(
            add_dead_pixels,
            seed=seed,
            amount=0.015,
            hot_fraction=0.15,
            dead_value=0.0,
            hot_value=1.0,
        ),
    )
    add(
        "dead_pixel",
        "amount=0.10 hot=0.50",
        seeded(
            add_dead_pixels,
            seed=seed,
            amount=0.10,
            hot_fraction=0.50,
            dead_value=0.0,
            hot_value=1.0,
        ),
    )

    add("speckle", "sigma=0.05", seeded(add_speckle, seed=seed, sigma=0.05, clamp=clamp))
    add("speckle", "sigma=0.18", seeded(add_speckle, seed=seed, sigma=0.18, clamp=clamp))
    add("speckle", "sigma=0.35", seeded(add_speckle, seed=seed, sigma=0.35, clamp=clamp))

    add("gamma_speckle", "looks=1", seeded(add_gamma_speckle, seed=seed, looks=1, clamp=clamp))
    add("gamma_speckle", "looks=4", seeded(add_gamma_speckle, seed=seed, looks=4, clamp=clamp))
    add("gamma_speckle", "looks=8", seeded(add_gamma_speckle, seed=seed, looks=8, clamp=clamp))

    add("rician", "sigma=0.02", seeded(add_rician, seed=seed, sigma=0.02, clamp=clamp))
    add("rician", "sigma=0.05", seeded(add_rician, seed=seed, sigma=0.05, clamp=clamp))
    add("rician", "sigma=0.10", seeded(add_rician, seed=seed, sigma=0.10, clamp=clamp))

    add("rayleigh", "sigma=0.02", seeded(add_rayleigh, seed=seed, sigma=0.02, clamp=clamp))
    add("rayleigh", "sigma=0.04", seeded(add_rayleigh, seed=seed, sigma=0.04, clamp=clamp))
    add("rayleigh", "sigma=0.10", seeded(add_rayleigh, seed=seed, sigma=0.10, clamp=clamp))

    add("quantization", "levels=16", unseeded(add_quantization, levels=16))
    add("quantization", "levels=256", unseeded(add_quantization, levels=256))
    add("quantization", "levels=4096", unseeded(add_quantization, levels=4096))

    add(
        "fixed_pattern",
        "offset=0.005 gain=0.01",
        seeded(add_fixed_pattern, seed=seed, offset_sigma=0.005, gain_sigma=0.01, clamp=clamp),
    )
    add(
        "fixed_pattern",
        "offset=0.025 gain=0.04",
        seeded(add_fixed_pattern, seed=seed, offset_sigma=0.025, gain_sigma=0.04, clamp=clamp),
    )
    add(
        "fixed_pattern",
        "offset=0.05 gain=0.08",
        seeded(add_fixed_pattern, seed=seed, offset_sigma=0.05, gain_sigma=0.08, clamp=clamp),
    )

    add(
        "stripe",
        "row=0.005 column=0.005",
        seeded(add_stripe, seed=seed, row_sigma=0.005, column_sigma=0.005, clamp=clamp),
    )
    add(
        "stripe",
        "row=0.025 column=0.018",
        seeded(add_stripe, seed=seed, row_sigma=0.025, column_sigma=0.018, clamp=clamp),
    )
    add(
        "stripe",
        "row=0.05 column=0.04",
        seeded(add_stripe, seed=seed, row_sigma=0.05, column_sigma=0.04, clamp=clamp),
    )
    return cases


def output_path(path: Path | None) -> Path:
    if path is not None:
        return path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"noise_1024_cuda0_{timestamp}.csv"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This benchmark expects an available CUDA device.")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    image = synthetic_1024_image(
        height=args.height,
        width=args.width,
        device=device,
        dtype=torch.float32,
    )
    variance_field = (0.00005 + 0.0015 * image).contiguous()
    torch.cuda.synchronize()

    cases = build_cases(seed=args.seed, clamp=(0.0, 1.0), variance_field=variance_field)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device),
                "cases": len(cases),
                "shape": list(image.shape),
                "warmups": args.warmups,
                "repeats": args.repeats,
            }
        )
    )

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        row = benchmark_case(case, image, warmups=args.warmups, repeats=args.repeats)
        rows.append(row)
        print(
            f"{index:02d}/{len(cases)} {case.noise:17s} {case.setting:28s} "
            f"wall_median={row['wall_median_ms']:.3f} ms "
            f"event_median={row['cuda_event_median_ms']:.3f} ms",
            flush=True,
        )

    path = output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV={path}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
