"""Reduction step — Chapter 4.4 of BENCHMARK_DESIGN.md.

Concatenate the per-seed Parquet shards and aggregate across the cells within a
generator family and across seeds, giving per-(family, filter, metric, noise)
mean, standard error, and the 5th/50th/95th percentile interval. NaN-producing
cells are reported as a ``valid_fraction`` rather than silently dropped.
"""

from __future__ import annotations

from pathlib import Path

GROUP_KEYS = (
    "study",
    "structure_class",
    "noise_condition_id",
    "snr_db",
    "filter_config_id",
    "filter_family",
    "filter_path",
    "metric",
)


def load_shards(shard_dir: Path):
    """Lazily concatenate every result shard under ``shard_dir``."""
    import polars as pl

    files = sorted(shard_dir.glob("*_seed*.parquet"))
    if not files:
        raise FileNotFoundError(f"no result shards (*_seed*.parquet) under {shard_dir}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def aggregate(shard_dir: Path, out_path: Path | None = None):
    """Aggregate shards into per-(family, filter, metric, noise) statistics."""
    import polars as pl

    frame = load_shards(shard_dir)
    valid = pl.col("value").filter(~pl.col("is_nan"))
    grouped = (
        frame.group_by(GROUP_KEYS)
        .agg(
            pl.len().alias("n_total"),
            (~pl.col("is_nan")).sum().alias("n_valid"),
            valid.mean().alias("mean"),
            valid.std().alias("std"),
            valid.quantile(0.05).alias("p05"),
            valid.quantile(0.50).alias("p50"),
            valid.quantile(0.95).alias("p95"),
        )
        .with_columns(
            (pl.col("n_valid") / pl.col("n_total")).alias("valid_fraction"),
            (pl.col("std") / pl.col("n_valid").sqrt()).alias("std_error"),
        )
        .sort(["study", "structure_class", "metric", "filter_config_id", "snr_db"])
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.write_parquet(out_path)
    return grouped
