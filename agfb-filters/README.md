# cpgf-filters

Batched, GPU-accelerated comparator gradient filters for the CPGF benchmark
suite (the 12 filters scored in §1.3–§1.7).

Every filter takes a `(B, H, W)` float32 intensity tensor and returns a tuple
`(g_x, g_y)` of `(B, H, W)` gradient tensors on the same device. Filters with
parameter sweeps (`DoG`, `SavitzkyGolay`, `FreemanAdelson`, `CPGF`) are
classes whose constructor builds the kernel once; small fixed filters
(`sobel_3`, `prewitt_3`, `scharr_3`, `roberts`, `farid_simoncelli_5`,
`sobel_5`, `sobel_7`) are plain functions.

## Filters

| Filter | API | Sweep |
|--------|-----|-------|
| Sobel-3 / 5 / 7 | `sobel_3(I)`, `sobel_5(I)`, `sobel_7(I)` | size |
| Prewitt-3 | `prewitt_3(I)` | — |
| Scharr-3 | `scharr_3(I)` | — |
| Roberts (2×2) | `roberts(I)` | — |
| Farid–Simoncelli-5 | `farid_simoncelli_5(I)` | — |
| Derivative of Gaussian | `DoG(sigma=...).apply(I)` | σ |
| Savitzky–Golay (2D) | `SavitzkyGolay(r=..., d=...).apply(I)` | (r, d) |
| CPGF (disc) | `CPGF(r=..., d=...).apply(I)` | (r, d) |
| Freeman–Adelson (G1) | `FreemanAdelsonG1(sigma=...).apply(I)` | σ |

## Workflow

```
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```

## Conventions

- Float32 on the device supplied by the input tensor.
- Input `(B, H, W)`; output `(g_x, g_y)` each `(B, H, W)`.
- Default padding: `replicate` for small spatial filters (matches the existing
  prototype), `reflect` for FFT-based filters.
- All small filters use `F.conv2d`; CPGF and Savitzky–Golay use the FFT path
  (`rfft2`/`irfft2` with `F · conj(K)`).

Author: J.C. Vaught.
