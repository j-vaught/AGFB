"""Locate the AGFB workspace and expose its four component packages.

The component packages (`agfb-generators`, `agfb-noise`, `agfb-filters`,
`agfb-metrics`) are not pip-installed; they are consumed directly from the
workspace checkout by inserting their directories onto `sys.path`, mirroring
the workspace's own `all_components_smoke_test.py`. Set the `AGFB_WORKSPACE`
environment variable to point at the directory holding the four folders if it
cannot be found by searching upward from the current location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

COMPONENT_DIRS = (
    "agfb-generators",
    "agfb-noise",
    "agfb-filters",
    "agfb-metrics",
)


def _has_components(candidate: Path) -> bool:
    return all((candidate / name).is_dir() for name in COMPONENT_DIRS)


def _candidates(start: Path):
    for base in (start, *start.parents):
        yield base
        yield base / "AGFB"
        yield base / "Documents" / "New project" / "AGFB"


def find_workspace(start: Path | None = None) -> Path:
    """Return the directory that contains all four AGFB component folders."""
    env = os.environ.get("AGFB_WORKSPACE")
    if env:
        candidate = Path(env).expanduser().resolve()
        if _has_components(candidate):
            return candidate
        raise RuntimeError(f"AGFB_WORKSPACE={env!r} does not contain the four component folders.")

    search_start = (Path(__file__).resolve().parent if start is None else start).resolve()
    for candidate in _candidates(search_start):
        if _has_components(candidate):
            return candidate
    raise RuntimeError(
        "Could not locate the AGFB workspace. Set AGFB_WORKSPACE to the folder "
        "holding agfb-generators / agfb-noise / agfb-filters / agfb-metrics."
    )


def ensure_components_importable(workspace: Path | None = None) -> Path:
    """Insert the four component directories onto `sys.path` (idempotent)."""
    ws = workspace or find_workspace()
    for component_dir in reversed(COMPONENT_DIRS):
        path = str(ws / component_dir)
        if path not in sys.path:
            sys.path.insert(0, path)
    return ws
