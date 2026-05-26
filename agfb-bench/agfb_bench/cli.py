"""Command-line entry point for the AGFB benchmark runner.

Examples
--------
Smoke test (small image, a few cells, one filter, CPU)::

    agfb-bench run --study clean_accuracy --image-size 96 --limit-cells 4 \
        --filter-profile headline --limit-filters 2 --out-dir runs/smoke

Production AWGN shard for one seed on a GPU::

    agfb-bench run --study awgn_robustness --seeds 0 --device cuda \
        --out-dir runs/synthetic/awgn_robustness

Aggregate finished shards::

    agfb-bench aggregate --shard-dir runs/synthetic/awgn_robustness \
        --out runs/synthetic/awgn_robustness/aggregate.parquet

Study names accept either the descriptive name or its legacy code letter
(A/B/C/CG/D/E/R/R_ss), so old commands keep working.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_seeds(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    return tuple(int(part) for part in text.split(",") if part.strip())


def _cmd_run(args: argparse.Namespace) -> None:
    from agfb_bench.runner import (
        build_study,
        canonical_study,
        run_backend_sweep,
        run_study,
        run_timing,
    )

    device = _resolve_device(args.device)
    out_dir = Path(args.out_dir)
    study = canonical_study(args.study)

    if study == "walltime_scaling":
        manifest = run_timing(
            image_size=args.image_size,
            device=device,
            out_dir=out_dir,
            repeats=args.repeats,
            filter_profile=args.filter_profile or "headline",
        )
        print(
            f"[walltime_scaling] timed {manifest['n_configs']} configs "
            f"at {args.image_size}px -> {out_dir}"
        )
        return

    if study == "backend_timing":
        sizes = tuple(int(s) for s in args.image_sizes.split(",") if s.strip())
        manifest = run_backend_sweep(
            image_sizes=sizes,
            device=device,
            out_dir=out_dir,
            repeats=args.repeats,
        )
        print(
            f"[backend_timing] {manifest['n_timed']}/{manifest['n_rows']} (filter,path) timed "
            f"across {manifest['n_candidates']} candidates at {sizes} -> {out_dir}"
        )
        return

    if study in ("edges", "supersampled"):
        from agfb_bench.realimg import DATASETS, run_realimg

        datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
        modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
        manifest = run_realimg(
            device=device,
            out_dir=out_dir,
            data_root=Path(args.data_root),
            datasets=datasets or DATASETS,
            modes=modes,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            supersample=args.supersample,
        )
        print(
            f"[{manifest['study']}] {manifest['n_rows']} rows over {manifest['datasets']} "
            f"(shard {manifest['shard']}, {manifest['n_filters']} filters) -> {out_dir}"
        )
        return

    spec = build_study(
        args.study,
        filter_profile=args.filter_profile,
        seeds=_parse_seeds(args.seeds),
        limit_cells=args.limit_cells,
        limit_filters=args.limit_filters,
    )
    print(
        f"[{spec.name}] cells={len(spec.cells)} noise={len(spec.conditions)} "
        f"filters={len(spec.filters)} seeds={spec.seeds} device={device} size={args.image_size}"
    )
    manifest = run_study(spec, image_size=args.image_size, device=device, out_dir=out_dir)
    print(
        f"[{spec.name}] {manifest['total_rows']} rows in {manifest['wall_seconds']}s "
        f"-> {out_dir} ({len(manifest['shards'])} shard(s))"
    )


def _cmd_aggregate(args: argparse.Namespace) -> None:
    from agfb_bench.aggregate import aggregate

    out_path = Path(args.out) if args.out else None
    grouped = aggregate(Path(args.shard_dir), out_path)
    print(grouped.head(20))
    if out_path:
        print(f"wrote {grouped.height} aggregated rows -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agfb-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="run an accuracy study, timing/backend sweep, or real-image study",
    )
    run.add_argument(
        "--study",
        required=True,
        help=(
            "clean_accuracy | awgn_robustness | noise_breadth | cpgf_grid | "
            "walltime_scaling | backend_timing | edges | supersampled "
            "(legacy A/B/C/CG/D/E/R/R_ss also accepted)"
        ),
    )
    run.add_argument("--seeds", default=None, help="comma-separated seeds (default: study default)")
    run.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    run.add_argument("--image-size", type=int, default=4096)
    run.add_argument(
        "--image-sizes",
        default="1024,2048,4096",
        help="comma-separated sizes for the backend_timing sweep",
    )
    run.add_argument("--filter-profile", default=None, help="headline | core | full")
    run.add_argument("--limit-cells", type=int, default=None, help="smoke-test cell cap")
    run.add_argument("--limit-filters", type=int, default=None, help="smoke-test filter cap")
    run.add_argument(
        "--repeats", type=int, default=50, help="timing repeats (walltime_scaling/backend_timing)"
    )
    run.add_argument(
        "--data-root", default=None, help="real-image dataset root (edges/supersampled)"
    )
    run.add_argument(
        "--datasets",
        default="bsds500,drive,bbbc039",
        help="comma-separated real-image datasets (edges/supersampled)",
    )
    run.add_argument(
        "--modes", default="raw,nms", help="edge-extraction modes for edges/supersampled: raw,nms"
    )
    run.add_argument(
        "--shard-index", type=int, default=0, help="this shard's index (real-image filter split)"
    )
    run.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="total shards splitting the catalog (edges/supersampled)",
    )
    run.add_argument(
        "--supersample",
        type=int,
        default=1,
        help="upscale factor for anti-aliased filtering (1 = off, writes the edges study; "
        ">1 writes the supersampled study)",
    )
    run.add_argument("--out-dir", required=True)
    run.set_defaults(func=_cmd_run)

    agg = sub.add_parser("aggregate", help="reduce shards to per-family statistics")
    agg.add_argument("--shard-dir", required=True)
    agg.add_argument("--out", default=None, help="output parquet path")
    agg.set_defaults(func=_cmd_aggregate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
