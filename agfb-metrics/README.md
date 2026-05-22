# agfb-metrics

`agfb-metrics` provides batched, GPU-accelerated benchmark metrics for the
Analytical Gradient Filter Benchmark (AGFB) suite. The package evaluates
predicted gradient fields against ground-truth gradient fields and returns one
score per image in the batch.

The public API is centered on metric functions, a shared PyTorch evaluator, and
a Triton evaluator for full-image pixel metrics. The package uses `(B, H, W)`
`float32` tensors throughout, so metric evaluation can run directly on CUDA
tensors produced by a benchmark pipeline.

## Install

Use `uv` to create the environment and install the pinned dependencies.

```bash
uv sync
```

On Linux `x86_64`, the project installs Triton for the fused CUDA pixel metric
backend. On macOS, the Triton CUDA tests skip automatically.

## Tensor Contract

Metric inputs are four gradient tensors with shape `(B, H, W)`.

| Tensor | Meaning |
|--------|---------|
| `g_x` | Predicted horizontal gradient |
| `g_y` | Predicted vertical gradient |
| `g_x_t` | Ground-truth horizontal gradient |
| `g_y_t` | Ground-truth vertical gradient |

All four tensors must be `torch.float32`, share the same shape, and live on the
same device. Metric outputs are `torch.float32` tensors with shape `(B,)`.

The standard masks are created from the true gradient field.

```python
from agfb_metrics.metrics import masks

mask_dict = masks(g_x_t, g_y_t)
signal_mask = mask_dict["signal"]
flat_mask = mask_dict["flat"]
```

`signal_mask` marks pixels where the true gradient magnitude is present.
`flat_mask` marks eroded background pixels. Passing `None` for a mask means the
corresponding pixel metric uses every pixel in each image.

## Metrics

The package exposes ten metric names in `ALL_METRICS`.

| Metric name | Function | Region | Definition |
|-------------|----------|--------|------------|
| `nrmse` | `nrmse(...)` | Signal pixels | Vector RMS error divided by mean true-gradient magnitude |
| `angular_mae` | `angular_mae(...)` | Signal pixels | Mean angular error in degrees |
| `tail_vector_error` | `tail_vector_error(...)` | Signal pixels | 95th-percentile per-pixel vector error |
| `tangential_normal_leak` | `tangential_normal_leak(...)` | Signal pixels | `10 log10(E_t / E_n)` |
| `magnitude_bias` | `magnitude_bias(...)` | Signal pixels | `mean(|grad_filter|) / mean(|grad_true|) - 1` |
| `noise_gain` | `noise_gain(...)` | Flat pixels | `std(|grad_filter|) / sigma_n` |
| `tail_spurious_grad` | `tail_spurious_grad(...)` | Flat pixels | 99th-percentile predicted gradient magnitude |
| `localization_offset` | `localization_offset(...)` | Cross-gradient profiles | Mean shift of the response peak from the true-gradient crest |
| `edge_fwhm` | `edge_fwhm(...)` | Cross-gradient profiles | Full width at half maximum of the gradient response profile |
| `sidelobe_ratio` | `sidelobe_ratio(...)` | Cross-gradient profiles | Largest side-lobe magnitude relative to the main lobe, in dB |

`PIXEL_METRICS` contains the seven metrics computed directly from per-pixel
field reductions. These are `nrmse`, `angular_mae`, `tail_vector_error`,
`tangential_normal_leak`, `magnitude_bias`, `noise_gain`, and
`tail_spurious_grad`.

The three profile metrics sample the predicted gradient magnitude along the
true-gradient normal direction. They use `signal_mask` to define the set of
profile anchor pixels.

## Basic Usage

Individual metric functions are useful for small checks and targeted tests.

```python
from agfb_metrics.metrics import masks, nrmse, tail_spurious_grad

mask_dict = masks(g_x_t, g_y_t)
score = nrmse(g_x, g_y, g_x_t, g_y_t, mask_dict["signal"])
tail = tail_spurious_grad(g_x, g_y, mask_dict["flat"])
```

`evaluate_metrics(...)` evaluates a selected metric set while sharing
intermediate tensors.

```python
from agfb_metrics.metrics import evaluate_metrics

out = evaluate_metrics(
    g_x,
    g_y,
    g_x_t,
    g_y_t,
    metrics=("nrmse", "angular_mae", "noise_gain"),
    signal_mask=signal_mask,
    flat_mask=flat_mask,
    sigma_n=0.01,
)

nrmse_value = out["nrmse"]
```

`evaluate_all_metrics(...)` evaluates every metric in `ALL_METRICS`.

```python
from agfb_metrics.metrics import evaluate_all_metrics

out = evaluate_all_metrics(
    g_x,
    g_y,
    g_x_t,
    g_y_t,
    signal_mask=signal_mask,
    flat_mask=flat_mask,
    sigma_n=0.01,
)
```

## Reusable Evaluators

`MetricEvaluator` fixes metric settings once and reuses the selected metric
plan across repeated calls. `use_compile=True` asks PyTorch to compile the
selected pixel metric graph for repeated same-shape CUDA batches.

```python
from agfb_metrics.metrics import PIXEL_METRICS, MetricEvaluator

evaluator = MetricEvaluator(
    metrics=PIXEL_METRICS,
    sigma_n=0.01,
    use_compile=True,
)

out = evaluator(
    g_x,
    g_y,
    g_x_t,
    g_y_t,
    signal_mask=None,
    flat_mask=None,
)
```

`TritonPixelEvaluator` evaluates full-image pixel metrics on CUDA tensors. It
specializes the Triton kernel to the selected metric names, so single-metric
calls run smaller kernels than full-set calls.

```python
from agfb_metrics.metrics import PIXEL_METRICS, TritonPixelEvaluator

evaluator = TritonPixelEvaluator(
    metrics=PIXEL_METRICS,
    sigma_n=0.01,
    tail_mode="histogram",
)

out = evaluator(
    g_x,
    g_y,
    g_x_t,
    g_y_t,
    signal_mask=None,
    flat_mask=None,
)
```

`tail_mode="exact"` uses exact PyTorch quantile semantics for percentile
metrics. `tail_mode="histogram"` uses a GPU histogram approximation for the
tail percentiles and is intended for high-throughput sweeps. `tail_bins`
controls the number of histogram bins and defaults to `4096`.

Use `is_triton_pixel_available()` before selecting the Triton backend in code
that also runs on CPU-only machines.

```python
from agfb_metrics.metrics import is_triton_pixel_available

if is_triton_pixel_available():
    evaluator = TritonPixelEvaluator(metrics=PIXEL_METRICS, sigma_n=0.01)
```

## Exact And Histogram Tail Metrics

The percentile metrics are the slowest pixel metrics because they require
global order statistics across every pixel. For one `4096x4096` image, each
tail metric works over `16,777,216` values.

Exact mode is appropriate for final reported tables. Histogram mode is useful
for large sweeps over filters, seeds, and parameter grids. A common workflow is
to run sweeps with histogram tails, select the cases of interest, and recompute
the selected results with exact tails.

## Performance

The following timings were measured on `comech-2422` using an NVIDIA RTX 6000
Ada GPU and PyTorch `2.8.0+cu128`. Inputs were full-image `4096x4096` CUDA
tensors with `signal_mask=None` and `flat_mask=None`. Times are median hot-path
milliseconds after warmup.

| Evaluator | Batch | Metric set | Time |
|-----------|-------|------------|------|
| `MetricEvaluator(use_compile=True)` | `B=1` | `PIXEL_METRICS`, exact tails | `6.08 ms` |
| `TritonPixelEvaluator(tail_mode="histogram")` | `B=1` | `PIXEL_METRICS`, histogram tails | `3.77 ms` |
| `MetricEvaluator(use_compile=True)` | `B=4` | `PIXEL_METRICS`, exact tails | `25.65 ms` |
| `TritonPixelEvaluator(tail_mode="histogram")` | `B=4` | `PIXEL_METRICS`, histogram tails | `6.44 ms` |

Single-metric timings for `B=1`, `4096x4096` full-image inputs are shown below.

| Metric | PyTorch compiled exact | Triton exact | Triton histogram |
|--------|------------------------|--------------|------------------|
| `nrmse` | `1.022 ms` | `0.354 ms` | `0.354 ms` |
| `angular_mae` | `0.988 ms` | `0.352 ms` | `0.352 ms` |
| `tail_vector_error` | `3.505 ms` | `3.488 ms` | `2.084 ms` |
| `tangential_normal_leak` | `0.987 ms` | `0.370 ms` | `0.369 ms` |
| `magnitude_bias` | `1.507 ms` | `0.348 ms` | `0.351 ms` |
| `noise_gain` | `0.676 ms` | `0.208 ms` | `0.208 ms` |
| `tail_spurious_grad` | `3.108 ms` | `3.095 ms` | `3.292 ms` |

The reduction-style metrics benefit most from Triton specialization. Exact
tail metrics are dominated by quantile computation. Histogram tails trade exact
quantile ordering for much higher sweep throughput.

## Repository Layout

`agfb_metrics/metrics/` contains metric functions, shared tensor helpers, the
PyTorch shared evaluator, and the Triton full-image pixel evaluator.
`tests/` contains unit tests, integration checks against the prototype behavior,
and skip-safe Triton tests.

## Development Workflow

Run the standard project checks before committing changes.

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run ty check .
uv run pytest
```

Author: J.C. Vaught.
