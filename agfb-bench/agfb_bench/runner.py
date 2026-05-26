"""Study orchestration — Chapter 5 of BENCHMARK_DESIGN.md.

Each study renders clean fields, injects noise, runs the filter grid, and scores
the result, writing one Parquet row per (cell x noise x filter x metric) plus a
JSON manifest. The seed axis is the shard axis (spec 5.4): a shard is one
``(study, seed)`` pair, written as ``<study>_seed<NN>.parquet``.

The metric-set schedule (spec 4.5) is applied here: the pixel set runs on every
condition; the profile set is added only on the clean pass. The clean-accuracy
study is clean, so it collects all ten metrics; the awgn-robustness and
noise-breadth studies are noisy, so they collect seven.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from agfb_bench.catalog import Cell, build_canonical_subset, build_catalog
from agfb_bench.config import PIXEL_METRICS, PRODUCTION_SEEDS, PROFILE_METRICS
from agfb_bench.fields import render_cell
from agfb_bench.filters import FilterConfig, build_backend_sweep_grid, build_filter_configs
from agfb_bench.metrics import build_masks, evaluate
from agfb_bench.noise import (
    NoiseCondition,
    awgn_conditions,
    native_conditions,
    noisy_awgn_conditions,
)
from agfb_bench.progress import ProgressWriter, progress_path

ROW_COLUMNS = (
    "study",
    "seed",
    "image_size",
    "cell_id",
    "generator",
    "structure_class",
    "angle_deg",
    "noise_condition_id",
    "noise_model",
    "noise_kind",
    "snr_db",
    "sigma_n",
    "filter_family",
    "filter_config_id",
    "filter_path",
    "metric",
    "value",
    "is_nan",
)


@dataclass
class StudySpec:
    name: str
    cells: list[Cell]
    conditions: list[NoiseCondition]
    filters: list[FilterConfig]
    seeds: tuple[int, ...]


# Descriptive study names are canonical; the original single-letter codes are
# kept as aliases so historical commands and notes keep resolving. Shard files,
# manifests, and the ``study`` result column all carry the canonical name.
STUDY_ALIASES = {
    "a": "clean_accuracy",
    "b": "awgn_robustness",
    "c": "noise_breadth",
    "cg": "cpgf_grid",
    "d": "walltime_scaling",
    "e": "backend_timing",
    "r": "edges",
    "r_ss": "supersampled",
}


def canonical_study(study: str) -> str:
    """Resolve a study name or legacy letter code to its canonical name."""
    key = study.strip().lower()
    return STUDY_ALIASES.get(key, key)


def build_study(
    study: str,
    *,
    filter_profile: str | None = None,
    seeds: tuple[int, ...] | None = None,
    limit_cells: int | None = None,
    limit_filters: int | None = None,
) -> StudySpec:
    """Assemble the (cells, noise, filters, seeds) for a study (spec 5.2)."""
    name = canonical_study(study)
    if name == "clean_accuracy":
        cells = build_catalog()
        conditions = [c for c in awgn_conditions() if c.kind == "clean"]
        filters = build_filter_configs(filter_profile or "full")
        default_seeds = (0,)
    elif name == "awgn_robustness":
        cells = build_catalog()
        conditions = noisy_awgn_conditions()
        filters = build_filter_configs(filter_profile or "full")
        default_seeds = PRODUCTION_SEEDS
    elif name in ("noise_breadth", "cpgf_grid"):
        # cpgf_grid is the CPGF degree-sweep variant of noise_breadth: identical
        # native-noise recipe (canonical subset x native conditions), but the
        # CPGF radius x degree grid instead of the core baseline set.
        cells = build_canonical_subset()
        conditions = native_conditions()
        default_profile = "cpgf_grid" if name == "cpgf_grid" else "core"
        filters = build_filter_configs(filter_profile or default_profile)
        default_seeds = PRODUCTION_SEEDS
    else:
        raise ValueError(
            f"study {study!r} has no row-generating spec "
            "(walltime_scaling/backend_timing are timing, edges is real-image)"
        )

    if limit_cells is not None:
        cells = cells[:limit_cells]
    if limit_filters is not None:
        filters = filters[:limit_filters]
    return StudySpec(name, cells, conditions, filters, seeds or default_seeds)


def _metric_names_for(condition: NoiseCondition) -> tuple[str, ...]:
    """Pixel set always; add the profile set only on the clean pass (spec 4.5)."""
    if condition.kind == "clean":
        return PIXEL_METRICS + PROFILE_METRICS
    return PIXEL_METRICS


def _free_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def iter_rows(spec: StudySpec, seed: int, image_size: int, device: torch.device, on_cell=None):
    """Yield result rows for one shard ``(study, seed)``.

    ``on_cell(cells_done, rows_emitted)`` is called after each cell completes,
    so callers can emit a progress heartbeat without buffering the whole shard.
    """
    import agfb_filters

    definitions = {
        config.config_id: agfb_filters.get_filter_definition(config.family, **config.params)
        for config in spec.filters
    }
    paths = {config.config_id: agfb_filters.ExecutionPath[config.path] for config in spec.filters}

    rows_emitted = 0
    for cell_index, cell in enumerate(spec.cells):
        frame = render_cell(cell, image_size, device)
        gx_t, gy_t = frame.gx, frame.gy
        mask_dict = build_masks(gx_t, gy_t, image_size)
        signal_mask, flat_mask = mask_dict["signal"], mask_dict["flat"]

        for condition in spec.conditions:
            if condition.deterministic and seed != spec.seeds[0]:
                continue  # quantization is deterministic: scored on one seed only
            from agfb_bench.noise import apply_noise

            noisy, sigma_n = apply_noise(
                condition,
                frame.I,
                contrast=cell.contrast,
                cell_seed=seed,
                flat_mask=flat_mask,
            )
            names = _metric_names_for(condition)

            for config in spec.filters:
                definition = definitions[config.config_id]
                try:
                    gx, gy = agfb_filters.run_filter(
                        definition,
                        noisy,
                        path=paths[config.config_id],
                        boundary=definition.default_boundary,
                    )
                    scores = evaluate(
                        gx,
                        gy,
                        gx_t,
                        gy_t,
                        names=names,
                        signal_mask=signal_mask,
                        flat_mask=flat_mask,
                        sigma_n=sigma_n,
                    )
                except torch.cuda.OutOfMemoryError:
                    _free_cuda(device)
                    scores = {name: float("nan") for name in names}

                for metric, value in scores.items():
                    rows_emitted += 1
                    yield {
                        "study": spec.name,
                        "seed": seed,
                        "image_size": image_size,
                        "cell_id": cell.cell_id,
                        "generator": cell.generator,
                        "structure_class": cell.structure_class,
                        "angle_deg": cell.angle_deg,
                        "noise_condition_id": condition.condition_id,
                        "noise_model": condition.model,
                        "noise_kind": condition.kind,
                        "snr_db": condition.snr_db,
                        "sigma_n": sigma_n,
                        "filter_family": config.family,
                        "filter_config_id": config.config_id,
                        "filter_path": config.path,
                        "metric": metric,
                        "value": value,
                        "is_nan": value != value,
                    }
        del frame, gx_t, gy_t, mask_dict, signal_mask, flat_mask
        _free_cuda(device)
        if on_cell is not None:
            on_cell(cell_index + 1, rows_emitted)


def run_study(
    spec: StudySpec,
    *,
    image_size: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Run every seed shard of a study, writing Parquet shards + a JSON manifest."""
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    shard_files: list[str] = []
    total_rows = 0
    started = time.perf_counter()
    for seed in spec.seeds:
        reporter = ProgressWriter(
            progress_path(out_dir, spec.name, seed),
            study=spec.name,
            seed=seed,
            device=str(device),
            n_cells=len(spec.cells),
            n_conditions=len(spec.conditions),
            n_filters=len(spec.filters),
        )
        try:
            rows = list(iter_rows(spec, seed, image_size, device, on_cell=reporter.update))
        except Exception as error:  # noqa: BLE001 — record then re-raise so the shard fails loudly
            # An OOM crash dumps a multi-paragraph CUDA report; collapse it to a
            # one-line status so the dashboard stays readable. Other failures keep
            # their concrete type and message.
            if isinstance(error, torch.cuda.OutOfMemoryError):
                message = "Stopped. OOM error"
            else:
                message = f"{type(error).__name__}: {error}"
            reporter.finish(status="error", error=message)
            raise
        frame = pl.DataFrame(rows, schema={c: None for c in ROW_COLUMNS} if not rows else None)
        shard_path = out_dir / f"{spec.name}_seed{seed:02d}.parquet"
        frame.write_parquet(shard_path)
        shard_files.append(shard_path.name)
        total_rows += len(rows)
        reporter.finish(status="done", rows=len(rows))
    elapsed = time.perf_counter() - started

    manifest = {
        "study": spec.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "device": str(device),
        "image_size": image_size,
        "seeds": list(spec.seeds),
        "n_cells": len(spec.cells),
        "n_noise_conditions": len(spec.conditions),
        "n_filter_configs": len(spec.filters),
        "filter_config_ids": [c.config_id for c in spec.filters],
        "pixel_metrics": list(PIXEL_METRICS),
        "profile_metrics": list(PROFILE_METRICS),
        "total_rows": total_rows,
        "wall_seconds": round(elapsed, 3),
        "shards": shard_files,
    }
    manifest_path = out_dir / f"{spec.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def run_timing(
    *,
    image_size: int,
    device: torch.device,
    out_dir: Path,
    repeats: int = 50,
    filter_profile: str = "headline",
) -> dict:
    """Walltime-scaling study: wall-clock per filter config on one cell (no metrics)."""
    import agfb_filters
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    cell = build_canonical_subset()[0]
    frame = render_cell(cell, image_size, device)
    configs = build_filter_configs(filter_profile)

    # Heartbeat so this study shows on the dashboard like the accuracy ones. It
    # has no seeds or cells, so each timed filter config is one progress unit
    # (seed pinned to 0).
    reporter = ProgressWriter(
        progress_path(out_dir, "walltime_scaling", 0),
        study="walltime_scaling",
        seed=0,
        device=str(device),
        n_cells=len(configs),
        n_conditions=1,
        n_filters=len(configs),
    )

    def time_config(definition, path) -> float:
        agfb_filters.run_filter(
            definition, frame.I, path=path, boundary=definition.default_boundary
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            agfb_filters.run_filter(
                definition, frame.I, path=path, boundary=definition.default_boundary
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / repeats * 1e3

    rows = []
    try:
        for config in configs:
            definition = agfb_filters.get_filter_definition(config.family, **config.params)
            path = agfb_filters.ExecutionPath[config.path]
            per_call_ms = time_config(definition, path)
            rows.append(
                {
                    "filter_family": config.family,
                    "filter_config_id": config.config_id,
                    "filter_path": config.path,
                    "image_size": image_size,
                    "repeats": repeats,
                    "ms_per_call": round(per_call_ms, 4),
                }
            )
            reporter.update(cells_done=len(rows), rows=len(rows))
    except Exception as error:  # noqa: BLE001 — record then re-raise so the shard fails loudly
        if isinstance(error, torch.cuda.OutOfMemoryError):
            message = "Stopped. OOM error"
        else:
            message = f"{type(error).__name__}: {error}"
        reporter.finish(status="error", error=message)
        raise

    pl.DataFrame(rows).write_parquet(out_dir / f"walltime_scaling_{image_size}.parquet")
    reporter.finish(status="done", rows=len(rows))
    return {
        "study": "walltime_scaling",
        "image_size": image_size,
        "n_configs": len(rows),
        "rows": rows,
    }


# All concrete runner paths except the orientation bank, which has a separate
# entry point (run_orientation_bank) and is out of scope by spec 3.3.
_SWEEP_PATHS = (
    "SEPARABLE",
    "SPATIAL_DENSE",
    "FFT",
    "SPARSE_OFFSETS",
    "ANTIPODAL_PAIRS",
    "STENCIL",
    "BOX_INTEGRAL",
    "RECURSIVE",
    "NONLINEAR_WINDOW",
    "ITERATIVE",
)


def run_backend_sweep(
    *,
    image_sizes: tuple[int, ...] = (1024, 2048, 4096),
    device: torch.device,
    out_dir: Path,
    repeats: int = 30,
) -> dict:
    """Backend-timing study: force every candidate filter onto every compatible execution path.

    For each ``(filter, path, image_size)`` the runner attempts ``run_filter``
    with that path forced. A path the filter cannot take raises ``ValueError``
    (recorded as ``status='unsupported'``); a path that runs but exhausts memory
    is recorded as ``status='oom'``; a successful run is timed and recorded as
    ``status='ok'``. One Parquet captures the whole compatibility-and-cost matrix
    so the paper can compare backend strategies head to head.
    """
    import agfb_filters
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    candidates = build_backend_sweep_grid()
    cell = build_canonical_subset()[0]

    # Progress is one unit per (candidate, image_size); the per-path inner loop
    # is fast because most paths reject immediately.
    total_units = len(candidates) * len(image_sizes)
    reporter = ProgressWriter(
        progress_path(out_dir, "backend_timing", 0),
        study="backend_timing",
        seed=0,
        device=str(device),
        n_cells=total_units,
        n_conditions=len(_SWEEP_PATHS),
        n_filters=len(candidates),
    )

    def time_path(definition, path, image) -> float:
        agfb_filters.run_filter(definition, image, path=path, boundary=definition.default_boundary)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            agfb_filters.run_filter(
                definition, image, path=path, boundary=definition.default_boundary
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / repeats * 1e3

    rows = []
    units_done = 0
    try:
        for image_size in image_sizes:
            frame = render_cell(cell, image_size, device)
            image = frame.I
            for config in candidates:
                definition = agfb_filters.get_filter_definition(config.family, **config.params)
                for path_name in _SWEEP_PATHS:
                    path = agfb_filters.ExecutionPath[path_name]
                    status = "ok"
                    ms = float("nan")
                    try:
                        ms = round(time_path(definition, path, image), 4)
                    except torch.cuda.OutOfMemoryError:
                        status = "oom"
                        _free_cuda(device)
                    except (ValueError, NotImplementedError):
                        status = "unsupported"
                    except Exception as error:  # noqa: BLE001 — log the path, keep sweeping
                        status = f"error:{type(error).__name__}"
                        _free_cuda(device)
                    rows.append(
                        {
                            "filter_family": config.family,
                            "filter_config_id": config.config_id,
                            "native_path": config.path,
                            "forced_path": path_name,
                            "image_size": image_size,
                            "repeats": repeats,
                            "status": status,
                            "ms_per_call": ms,
                        }
                    )
                units_done += 1
                reporter.update(cells_done=units_done, rows=len(rows))
            del frame, image
            _free_cuda(device)
    except Exception as error:  # noqa: BLE001 — record then re-raise so the shard fails loudly
        if isinstance(error, torch.cuda.OutOfMemoryError):
            message = "Stopped. OOM error"
        else:
            message = f"{type(error).__name__}: {error}"
        reporter.finish(status="error", error=message)
        raise

    pl.DataFrame(rows).write_parquet(out_dir / "backend_timing_sweep.parquet")
    reporter.finish(status="done", rows=len(rows))
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    return {
        "study": "backend_timing",
        "image_sizes": list(image_sizes),
        "n_candidates": len(candidates),
        "n_rows": len(rows),
        "n_timed": n_ok,
    }
