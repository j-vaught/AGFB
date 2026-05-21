**AGFB Filters**

`agfb-filters` provides batched gradient filters for the Analytical Gradient Filter Benchmark. Inputs are `torch.Tensor` values with shape `(batch, height, width)`. Each filter returns `(gradient_x, gradient_y)` with the same shape.

**Custom Filters**

The easiest way to add a filter outside the package is to create a `GradientFilterDefinition` with one of the helper constructors and pass it to `run_filter`. Dense filters provide full horizontal and vertical kernels. Separable filters provide one smoothing kernel and one derivative kernel, and the package derives dense kernels for path comparisons.

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

**Built-In Registry**

Built-in filters are available through the same registry. For example, `get_filter_definition("sobel_3")` returns the Sobel 3-tap definition, while parameterized entries such as `get_filter_definition("cpgf", radius=2, degree=2)` construct generated filters.

**Validation**

Custom definitions are validated when they are created. Dense kernels must be finite floating-point tensors with matching two-dimensional shapes. Even-sized dense kernels require explicit `spatial_padding` because the runner must know how to preserve the input shape. Separable kernels must be one-dimensional, finite, floating-point, and odd length.
