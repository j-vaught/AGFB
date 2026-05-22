**AGFB Filters**

`agfb-filters` provides batched gradient filter implementations. Inputs are floating-point `torch.Tensor` values with shape `(batch, height, width)`. Gradient filters return `(gradient_x, gradient_y)` with the same shape and dtype as the input. Orientation-bank filters use `run_orientation_bank` and return raw directional responses with shape `(batch, angles, height, width)`.

**Custom Filters**

The easiest way to add a filter outside the package is to create a `GradientFilterDefinition` with one of the helper constructors and pass it to `run_filter`. Dense filters provide full horizontal and vertical kernels. Separable filters provide one smoothing kernel and one derivative kernel. Sparse-offset, box, recursive, nonlinear local-window, iterative, and orientation-bank helpers are available for filters that should not be represented as ordinary dense kernels.

```python
import torch

from agfb_filters import ExecutionPath, define_dense_filter, run_filter

definition = define_dense_filter(
    name="my_difference",
    kernel_x=[
        [0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0],
    ],
    kernel_y=[
        [0.0, -0.5, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ],
)

image = torch.randn(4, 128, 128)
gradient_x, gradient_y = run_filter(
    definition,
    image,
    path=ExecutionPath.SPATIAL_DENSE,
    boundary=definition.default_boundary,
)
```

Reusable filters can be registered by name. Registration stores a factory rather than one tensor instance, so parameterized filters can accept arguments when the definition is requested.

```python
from agfb_filters import define_separable_filter, get_filter_definition, register_filter


def make_my_filter(scale: float = 1.0):
    return define_separable_filter(
        name="my_separable_difference",
        smooth_kernel_1d=[1.0],
        derivative_kernel_1d=[-0.5 * scale, 0.0, 0.5 * scale],
    )


register_filter("my_separable_difference", make_my_filter)
definition = get_filter_definition("my_separable_difference", scale=2.0)
```

**Orientation Banks**

Orientation-bank definitions are intentionally separate from gradient-pair filters. The raw bank response is the primary output. `collapse_orientation_bank` can project those responses with either `max_projection` or `least_squares_projection` when a gradient-like result is needed.

```python
import torch

from agfb_filters import (
    ExecutionPath,
    collapse_orientation_bank,
    get_filter_definition,
    run_orientation_bank,
)

definition = get_filter_definition(
    "anisotropic_gaussian_orientation_bank",
    angle_count=8,
    sigma_parallel=1.0,
    sigma_perpendicular=2.0,
)
image = torch.randn(2, 64, 64)
bank = run_orientation_bank(
    definition,
    image,
    path=ExecutionPath.ORIENTATION_BANK,
    boundary=definition.default_boundary,
)
collapsed = collapse_orientation_bank(bank, mode="least_squares_projection")
```

**Built-In Registry**

Built-in filters are available through the same registry. `get_filter_definition("sobel_3")` returns the Sobel 3-tap definition. Parameterized entries such as `get_filter_definition("cpgf", radius=2, degree=2)`, `get_filter_definition("haar_box_gradient", radius=2)`, and `get_filter_definition("perona_malik_gradient", iterations=5, step_size=0.15, kappa=0.2)` construct generated filters.

**Adding Shipped Filters**

Each shipped filter module owns a `FILTER_SPECS` tuple. To add a default filter to the package, create a module in `agfb_filters/filters/` with a definition factory such as `my_filter_definition()`, public wrapper functions or classes, and one spec dictionary with the registry name, definition factory name, public exports, and smoke-test settings. The catalog collects those module-local specs for package root exports, `agfb_filters.filters` exports, built-in registry wiring, and smoke tests.

The filter module should keep the math local and delegate execution to `run_filter` or `run_orientation_bank`. A simple fixed separable filter can define one `_DEFINITION`, return it from `my_filter_definition()`, and expose a small function that calls `run_filter(_DEFINITION, image, path=path, boundary=boundary)`. Parameterized filters should expose a definition factory that accepts the parameters and returns a fresh `GradientFilterDefinition`.

```python
from __future__ import annotations

import torch

from agfb_filters.filters.definitions import define_separable_filter
from agfb_filters.runtime.execution import BoundaryCondition, BoundaryMode, ExecutionPath
from agfb_filters.runtime.runner import run_filter

_DEFINITION = define_separable_filter(
    name="my_filter",
    default_boundary=BoundaryCondition(BoundaryMode.REPLICATE),
    smooth_kernel_1d=torch.tensor([1.0, 2.0, 1.0]) / 4.0,
    derivative_kernel_1d=torch.tensor([-1.0, 0.0, 1.0]) / 2.0,
    metadata={"kernel_size": 3},
)
FILTER_SPECS = (
    {
        "name": "my_filter",
        "definition_factory": "my_filter_definition",
        "description": "my separable filter",
        "exports": ("my_filter", "my_filter_definition"),
        "smoke_path": "separable",
    },
)


def my_filter_definition():
    return _DEFINITION


def my_filter(
    image: torch.Tensor,
    *,
    path: ExecutionPath | str,
    boundary: BoundaryCondition | None = None,
):
    return run_filter(_DEFINITION, image, path=path, boundary=boundary)
```

**Validation**

Custom definitions are validated when they are created. Dense kernels must be finite floating-point tensors with matching two-dimensional shapes. Even-sized dense kernels require explicit `spatial_padding` because the runner must know how to preserve the input shape. Separable kernels must be one-dimensional, finite, floating-point, and odd length. Orientation-bank angles are radians in `[0, pi)`, strictly increasing, and use `theta=0` for the positive column direction and `theta=pi/2` for the positive row direction.
