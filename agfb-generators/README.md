# cpgf-generators

Batched, GPU-accelerated synthetic frame generators for the CPGF benchmark suite (§1.1 of the
benchmark specification).

Every generator produces a closed-form intensity field `I` and an analytic gradient `(g_x, g_y)`
on the same grid. Parameters are passed as 1-D tensors of length `B` so that a single call yields
a batched output of shape `(B, H, W)` for intensity and `(B, 2, H, W)` for the gradient stack.

## Layout

| Folder | Purpose |
|--------|---------|
| `cpgf_generators/` | Generator package (one module per generator + shared base + helpers). |
| `tests/` | Numerical regression tests against the reference prototypes. |

## Workflow

```
uv sync
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
