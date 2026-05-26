"""AGFB synthetic benchmark runner.

Implements the locked specification in ``benchmark/BENCHMARK_DESIGN.md``: the
generator catalog (Chapter 1), the noise system (Chapter 2), the filter grid
(Chapter 3), the metric sets (Chapter 4), and the Study A-D protocol with
seed sharding (Chapter 5).

Importing this package makes the four AGFB component packages importable.
"""

from __future__ import annotations

from agfb_bench._paths import ensure_components_importable

ensure_components_importable()

__all__ = ["ensure_components_importable"]
