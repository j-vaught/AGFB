# agfb-generators

Batched, GPU-accelerated analytic frame generators for the Analytic Gradient Filter Benchmark
(AGFB).

Every generator produces a closed-form intensity field `I` and an analytic gradient `(g_x, g_y)`
on the same grid. Parameters are passed as 1-D tensors of length `B` so that a single call yields
a batched output of shape `(B, H, W)` for intensity and `(B, 2, H, W)` for the gradient stack.

## Layout

| Folder | Purpose |
|--------|---------|
| `agfb_generator_visual_check.ipynb` | Interactive notebook for generator checks and visual inspection. |
| `agfb_generators/` | Generator package (one module per generator + shared base + helpers). |
| `tests/` | Numerical regression tests for analytic gradients and batched rendering. |

## Available Generators

| Family | Generators |
|--------|------------|
| Polynomial | `polynomial` |
| Edges and transitions | `smoothed_step`, `hard_step`, `finite_ramp`, `smoothed_ramp`, `roof_profile`, `mach_band` |
| Blobs and scale | `gaussian_blob`, `anisotropic_blob` |
| Ridges and bars | `gaussian_ridge`, `smoothed_bar`, `asymmetric_ridge`, `curved_ridge` |
| Circular boundaries | `curved_arc` |
| Frequency fields | `sinusoid`, `chirp`, `gabor_packet` |
| Junctions | `smoothed_l_junction`, `hard_l_junction`, `smoothed_t_junction`, `hard_t_junction`, `smoothed_y_junction`, `hard_y_junction`, `smoothed_x_junction`, `hard_x_junction` |
| Vessels | `vessel_crossing`, `vessel_bifurcation` |

Truth helpers are available for structures that need semantic masks or labels. Use
`junction_mask`, `vessel_crossing_truth`, and `vessel_bifurcation_truth` when metrics need
centerline, branch, radius, or junction-region maps in addition to the analytic gradients.

## Workflow

```
uv sync
uv run check-unicode README.md pyproject.toml uv.lock *.ipynb docs/**/*.md docs/**/*.typ docs/**/*.bib agfb_generators/*.py tests/*.py
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```

## Conventions

- Float32 on the device passed at call time (CPU or CUDA).
- Shapes: intensity `(B, H, W)`, gradient `(B, 2, H, W)` ordered as `(g_x, g_y)`.
- Coordinates: origin at image center, `+x` to the right, `+y` downward (matrix row order).
- All scalar parameters accept either a Python scalar (broadcast across the batch) or a
  1-D `torch.Tensor` of length `B`.

Author: J.C. Vaught.
