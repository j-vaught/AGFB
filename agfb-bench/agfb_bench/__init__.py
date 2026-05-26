"""AGFB synthetic benchmark runner.

Implements the synthetic benchmark: the generator catalog, the noise system,
the filter grid, the metric sets, and the Study A-D protocol with seed
sharding.

Importing this package makes the four AGFB component packages importable.
"""

from __future__ import annotations

from agfb_bench._paths import ensure_components_importable

ensure_components_importable()

__all__ = ["ensure_components_importable"]
