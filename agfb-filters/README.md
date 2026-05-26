# agfb-filters

**Overview**

`agfb-filters` is a small PyTorch package for evaluating batched image-gradient filters with explicit execution paths. It is designed for experiments where the same filter definition needs to be run through dense spatial kernels, separable kernels, fast Fourier transform execution, sparse-offset execution, recursive filters, nonlinear window filters, or orientation-bank filters without changing the input tensor convention.

The package uses `(batch, height, width)` floating-point tensors throughout. Standard gradient filters return `(gradient_x, gradient_y)` tensors with the same shape as the input. Orientation-bank filters return a response stack with shape `(batch, angles, height, width)` and the corresponding orientation angles.

**Installation**

The repository uses `uv` for environment and dependency management. From a local checkout, create the environment with the following command.

```sh
uv sync
```

The package can also be installed in editable mode from the repository root.

```sh
uv pip install -e .
```

**Basic Use**

The main entry point is `get_filter_definition`, which builds a named filter definition. The definition is then passed to `run_filter` with an explicit execution path and boundary condition.

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

The same definition can be run through another compatible path when the filter supports it.

```python
gradient_x, gradient_y = run_filter(
    definition,
    image,
    path=ExecutionPath.SEPARABLE,
    boundary=definition.default_boundary,
)
```

**Included Filters**

The package includes common finite impulse response filters such as central difference, Roberts, Prewitt, Sobel, Scharr, Ando consistent gradient operators, Farid-Simoncelli, derivative-of-Gaussian, Savitzky-Golay, and Freeman-Adelson filters. It also includes polynomial and compact polynomial gradient filters, sparse central differences, Haar-style box gradients, Deriche recursive Gaussian derivatives, robust local-plane gradients, Perona-Malik diffusion gradients, Riesz transforms, and several orientation-bank constructions.

The shipped filter catalog is available from Python.

```python
from agfb_filters import shipped_filter_specs

for spec in shipped_filter_specs():
    print(spec.name, spec.description)
```

Custom dense, separable, sparse-offset, recursive, nonlinear, iterative, Riesz, and orientation-bank definitions can be created with the `define_*` helper functions exposed from `agfb_filters`.

**Execution Paths**

Execution is deliberately explicit. `ExecutionPath.SPATIAL_DENSE` applies dense kernels with direct spatial correlation. `ExecutionPath.SEPARABLE` applies compatible separable filters as one-dimensional passes. `ExecutionPath.FFT` evaluates compatible dense filters or Riesz transforms in the frequency domain. `ExecutionPath.SPARSE_OFFSETS` evaluates sparse stencils directly. `ExecutionPath.ANTIPODAL_PAIRS` uses odd-symmetric kernel structure when available. `ExecutionPath.BOX_INTEGRAL`, `ExecutionPath.RECURSIVE`, `ExecutionPath.NONLINEAR_WINDOW`, `ExecutionPath.ITERATIVE`, and `ExecutionPath.ORIENTATION_BANK` select the specialized implementations for the corresponding filter families.

Boundary behavior is also explicit. Supported modes are reflect, replicate, constant, and circular boundaries through `BoundaryCondition` and `BoundaryMode`.

**Orientation Banks**

Orientation-bank filters can be run directly with `run_orientation_bank`.

```python
from agfb_filters import ExecutionPath, get_filter_definition, run_orientation_bank

definition = get_filter_definition(
    "anisotropic_gaussian_orientation_bank",
    angle_count=8,
    sigma_parallel=1.0,
    sigma_perpendicular=2.0,
)

result = run_orientation_bank(
    definition,
    image,
    path=ExecutionPath.ORIENTATION_BANK,
    boundary=definition.default_boundary,
)
```

Gradient pairs can also be projected onto angle grids with `steer_gradient` or `run_steered_filter_bank`. Raw orientation-bank responses can be collapsed with `collapse_orientation_bank` when a gradient-like summary is useful.

**Notebooks**

The notebooks are intentionally simple examples rather than a separate documentation system. Single-filter notebooks are in `notebooks/filters`. Orientation examples built from gradient pairs are in `notebooks/orientable_gxgy`. Rotated-kernel orientation-bank examples are in `notebooks/orientable_rotated`.

The notebooks are useful for inspecting filter responses, timing individual definitions, and comparing orientation behavior. They are not required to use the package as a library.

**Development**

The lightweight local checks are formatting, linting, type checking, and package building.

```sh
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv build
```

**License**

This project is distributed under the MIT License. See the `LICENSE` file at the repository root for the full text.
