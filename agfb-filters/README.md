# agfb-filters

**Overview**

`agfb-filters` provides batched gradient filter definitions and explicit execution paths for PyTorch tensors. Inputs use `(batch, height, width)` floating-point tensors, and gradient filters return `(gradient_x, gradient_y)` tensors with the same shape.

The package includes finite impulse response filters, sparse offset filters, box-gradient filters, recursive Gaussian derivative filters, robust local-plane filters, Perona-Malik diffusion gradients, Riesz transforms, and orientation-bank filters.

**Installation**

Use `uv` to create the project environment and install development dependencies.

```sh
uv sync
```

**Example**

```python
import torch

from agfb_filters import ExecutionPath, get_filter_definition, run_filter

image = torch.randn(2, 64, 64)
definition = get_filter_definition("sobel_3")

gradient_x, gradient_y = run_filter(
    definition,
    image,
    path=ExecutionPath.SPATIAL_DENSE,
    boundary=definition.default_boundary,
)
```

**Checks**

The standard local validation workflow formats the code, applies lint fixes, runs type checks, and executes the test suite.

```sh
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```
