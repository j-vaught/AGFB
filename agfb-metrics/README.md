# agfb-metrics

Batched benchmark metrics for the Analytical Gradient Filter Benchmark (AGFB)
suite.

| Metric | Meaning |
|--------|---------|
| NRMSE | Headline error, normalised vector RMS on signal pixels |
| Angular MAE | Mean angular error, degrees, on signal pixels |
| Tail vector error | 95th-percentile per-pixel vector error on signal pixels |
| Localization offset | Mean shift of `|grad|` peak from the true-gradient crest, in pixels |
| Tangential-normal leak | `10 log10(E_t / E_n)`, tangential vs normal energy |
| Magnitude bias | `<|grad_filter|> / <|grad_true|> - 1` (signed) |
| Edge FWHM | Full-width-half-max of `|grad|` perpendicular to the true-gradient crest |
| Side-lobe ratio | Max-outside-main-lobe / peak, in dB |
| Noise gain | `std(|grad_filter|)_F / sigma_n` on flat regions |
| Tail spurious gradient | 99th-percentile `|grad_filter|` on flat regions |

## Layout

- `agfb_metrics/metrics/` contains metric definitions and shared metric
  helpers.

## Conventions

- Input `(B, H, W)` float32 gradient tensors `g_x`, `g_y` (filter output and
  ground truth), on the same device.
- Output is a length-`B` float32 tensor - one metric value per image.
- Masks (`signal`, `flat`) are computed from the true gradient field by
  `agfb_metrics.metrics.base.masks` and follow the Section 1.1 definition
  (inward-eroded background mask, dilate 8 px by default - matches the
  existing prototype).

## Workflow

```
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```

Author: J.C. Vaught.
