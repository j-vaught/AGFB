"""Live progress heartbeats for the dashboard.

Each shard (one ``(study, seed)`` process) writes a small JSON file next to its
Parquet output, refreshed as cells complete. The dashboard globs every
``*.progress.json`` across both servers and merges them into one view, so the
file is self-describing: it names its host, device, study, and seed.

The write is atomic (temp file + ``os.replace``) so a reader never sees a
half-written file, and throttled so a fast run does not thrash the disk.
"""

from __future__ import annotations

import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path


class ProgressWriter:
    """Accumulates per-cell progress for one shard and writes a JSON heartbeat."""

    def __init__(
        self,
        path: Path,
        *,
        study: str,
        seed: int,
        device: str,
        n_cells: int,
        n_conditions: int,
        n_filters: int,
        min_interval_s: float = 2.0,
    ) -> None:
        self._path = path
        self._min_interval = min_interval_s
        self._started = time.perf_counter()
        self._last_write = 0.0
        self._state = {
            "study": study,
            "host": platform.node(),
            "device": device,
            "pid": os.getpid(),
            "seed": seed,
            "n_cells": n_cells,
            "n_conditions": n_conditions,
            "n_filters": n_filters,
            "cells_done": 0,
            "rows": 0,
            "fraction": 0.0,
            "status": "running",
            "started_utc": datetime.now(UTC).isoformat(),
            "updated_utc": None,
            "elapsed_s": 0.0,
            "eta_s": None,
            "error": None,
        }
        self._write(force=True)

    def update(self, cells_done: int, rows: int) -> None:
        """Record progress after a cell finishes; write if the throttle allows."""
        elapsed = time.perf_counter() - self._started
        n_cells = self._state["n_cells"] or 1
        fraction = cells_done / n_cells
        eta = (elapsed / cells_done) * (n_cells - cells_done) if cells_done else None
        self._state.update(
            cells_done=cells_done,
            rows=rows,
            fraction=round(fraction, 4),
            elapsed_s=round(elapsed, 1),
            eta_s=round(eta, 1) if eta is not None else None,
        )
        self._write(force=False)

    def finish(
        self, status: str = "done", rows: int | None = None, error: str | None = None
    ) -> None:
        """Write a terminal heartbeat (``done`` or ``error``)."""
        if rows is not None:
            self._state["rows"] = rows
        if status == "done":
            self._state["fraction"] = 1.0
            self._state["cells_done"] = self._state["n_cells"]
            self._state["eta_s"] = 0.0
        self._state["status"] = status
        self._state["error"] = error
        self._state["elapsed_s"] = round(time.perf_counter() - self._started, 1)
        self._write(force=True)

    def _write(self, *, force: bool) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_write) < self._min_interval:
            return
        self._last_write = now
        self._state["updated_utc"] = datetime.now(UTC).isoformat()
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, self._path)


def progress_path(out_dir: Path, study: str, seed: int) -> Path:
    """Heartbeat path for a shard, sibling to its ``<study>_seed<NN>.parquet``."""
    return out_dir / f"{study}_seed{seed:02d}.progress.json"
