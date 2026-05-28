# AGFB - Analytic Gradient Filter Benchmark

AGFB is a benchmark for image-gradient filters. It renders synthetic intensity
fields whose horizontal and vertical gradients are known analytically, corrupts
them with a wide bank of noise models, runs each gradient filter, and scores the
result against the exact reference. Because the ground-truth gradient is known at
every pixel, accuracy is measured directly rather than against a finite-difference
approximation. The suite also includes a real-image edge-detection study scored
against human boundary annotations.

This repository is the workspace that ties the AGFB component packages together,
runs the benchmark, and stores the raw measurements behind every figure and table
in the accompanying paper on the Compact Polynomial Gradient Filter (CPGF). A
companion notebook recomputes each headline number from those raw measurements so
the results can be checked independently.

## What is in the repository

The project is a `uv` workspace of five installable packages plus the result data
and a reviewer notebook.

| Package | Role |
|---|---|
| `agfb-generators` | Batched PyTorch generators that render a synthetic field and its analytic `(gx, gy)` gradients. |
| `agfb-noise` | Batched noise models (Gaussian, Poisson, speckle, impulse, correlated, quantization, and more) for controlled corruption. |
| `agfb-filters` | Gradient filters with explicit execution paths (dense, separable, FFT, sparse-offset, recursive, nonlinear, orientation-bank). |
| `agfb-metrics` | GPU-accelerated gradient-field and edge-detection metrics, one score per image. |
| `agfb-bench` | The runner that composes the four packages into the studies and writes Parquet results. |

## The studies

Each row-study renders the generator catalog, applies a noise bank, runs the
filter grid, and writes one Parquet row per `(cell x noise x filter x metric)`.
Seeds are the shard axis: one shard is one `(study, seed)` pair.

| Study | Generators | Noise | Filters | Seeds |
|---|---|---|---|---|
| clean accuracy | full catalog (559) | clean only | full (110) | 1 |
| AWGN robustness | full catalog (559) | 12 dB levels | full (110) | 8 |
| noise breadth | canonical (24) | 79 native conditions | core (~26) | 8 |
| CPGF grid | canonical (24) | 79 native conditions | CPGF radius x degree grid | 8 |
| wall-clock / backend | 1 representative | clean + 10 dB | both execution paths | timing reps |
| real-image edges | BSDS500 images | - | gradient magnitude + threshold sweep | - |

## Quickstart

The workspace uses [`uv`](https://docs.astral.sh/uv/) for Python, environments,
and dependencies. From the repository root:

```bash
uv sync
```

This creates a single environment with all five packages installed editable, so
imports such as `import agfb_generators` and the `agfb-bench` CLI are available
immediately. Python 3.11-3.13 is supported. On Linux, `uv` resolves the CUDA 12.4
PyTorch wheels; on macOS it uses the standard CPU/MPS build.

Run a small smoke benchmark on CPU:

```bash
uv run agfb-bench run --study clean_accuracy --image-size 96 \
    --limit-cells 4 --filter-profile headline --limit-filters 2 \
    --out-dir runs/smoke
```

Run a production shard on a GPU and reduce finished shards:

```bash
uv run agfb-bench run --study awgn_robustness --seeds 0 --device cuda --out-dir runs/awgn
uv run agfb-bench aggregate --shard-dir runs/awgn --out runs/awgn/aggregate.parquet
```

## Reproducing the paper's results

`reproduce_paper_claims.ipynb` is a reviewer's cross-check. Every headline number
in the paper is a reduction of the raw per-image measurements stored under
`runs/`. The notebook reloads those Parquet files, recomputes each number from
scratch with the same scoring protocol, and places the recomputed value next to
the value printed in the paper. It reads only the result files, writes nothing,
and needs no GPU.

```bash
uv sync
uv run jupyter lab reproduce_paper_claims.ipynb   # or: Run All
```

Each section ends in a verdict table with `paper`, `recomputed`, `abs_diff`, and
`match` columns. Tolerances follow the precision the paper prints, so a `FAIL` row
is a genuine disagreement, not a rounding artifact.

## Repository layout

```
agfb-generators/   synthetic field + analytic gradient generators
agfb-noise/        noise models
agfb-filters/      gradient filters and execution paths
agfb-metrics/      gradient-field and edge-detection metrics
agfb-bench/        benchmark runner and CLI
runs/              raw measurements (Parquet shards) and analysis scripts
  synthetic/       clean_accuracy, awgn_robustness, noise_breadth, cpgf_grid
  realimg/         edges, supersampled
  timing/          backend_timing, walltime_scaling
  _analysis/       scripts that reduce shards into the paper's tables
reproduce_paper_claims.ipynb   reviewer cross-check notebook
```

## Results data

The raw per-image measurements that back every figure and table are the Parquet
shards under `runs/`. They are the input the reproduce notebook consumes; the
analysis scripts in `runs/_analysis/` show how each table in the paper is derived
from them.

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## Author

J.C. Vaught (`jvaught@sc.edu`).
