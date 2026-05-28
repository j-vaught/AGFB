"""Shared synthetic-result reductions."""

from __future__ import annotations

import polars as pl


def deduplicate_synthetic_results(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse repeated synthetic measurements by cell before aggregation.

    Older synthetic shards can contain duplicate rows for ten smoothed-step
    cells. Current packaged shards are already de-duplicated; this reduction
    remains as a defensive guard before computing catalog means.
    """
    if "cell_id" not in df.columns or "value" not in df.columns:
        return df

    keys = [c for c in df.columns if c not in {"value", "is_nan"}]
    return df.unique(subset=keys, maintain_order=True)
