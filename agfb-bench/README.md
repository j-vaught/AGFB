# agfb-bench

The runner for the AGFB synthetic gradient benchmark. It implements
[`../BENCHMARK_DESIGN.md`](../BENCHMARK_DESIGN.md) — the locked specification —
exactly. The four AGFB component packages (`agfb-generators`, `agfb-noise`,
`agfb-filters`, `agfb-metrics`) are consumed directly from the workspace
checkout via `sys.path` injection, not pip-installed; set `AGFB_WORKSPACE` to the
folder holding the four component directories if it is not found automatically.

Author: J.C. Vaught (`jvaught@sc.edu`).

## What it does

A run renders a clean field (Chapter 1), injects a noise condition (Chapter 2),
applies each filter config (Chapter 3), and scores the result (Chapter 4),
writing one Parquet row per `(cell × noise × filter × metric)` plus a JSON
manifest. The seed axis is the shard axis (Chapter 5): a shard is one
`(study, seed)` pair.

The four studies (spec 5.2):

| Study | Generators | Noise | Filters | Seeds | Metrics |
|---|---|---|---|---|---|
| A — clean accuracy | full catalog (569) | clean only | `full` (~109) | 1 | all 10 |
| B — AWGN robustness | full catalog (569) | 12 dB levels | `full` (~109) | 8 | pixel 7 |
| C — noise breadth | canonical (24) | 79 native conditions | `core` (~26) | 8 | pixel 7 |
| D — wall-clock | 1 representative | clean + 10 dB | by profile, both paths | 50 reps | timing only |

The metric-set schedule (spec 4.5) is enforced in `runner.py`: the pixel set runs
on every condition, the profile set only on the clean pass. Study A is therefore
the only row-study that collects all ten metrics.

## Modules

- `config.py` — locked constants (sizes, seeds, dB grid, metric sets, `sigma_n`).
- `catalog.py` — the 569-cell generator catalog + the 24-cell canonical subset.
- `noise.py` — the 13-level AWGN grid + the 79 native-unit ladders, and
  `apply_noise` (which also returns the `sigma_n` that `noise_gain` needs).
- `filters.py` — the filter grid and the `headline` / `core` / `full` profiles;
  underdetermined polynomial configs are skipped at construction.
- `fields.py` — render a catalog cell into an `agfb-generators` Frame.
- `metrics.py` — masks and the pixel / profile / all evaluation wrappers.
- `runner.py` — Study A/B/C row generation and the Study D timing pass.
- `aggregate.py` — concatenate shards into per-family statistics.
- `cli.py` — `agfb-bench run` / `agfb-bench aggregate`.

## Usage

```bash
uv sync

# Smoke test: small image, a few cells, two filters, on CPU.
uv run agfb-bench run --study A --image-size 96 \
    --limit-cells 4 --filter-profile headline --limit-filters 2 \
    --out-dir runs/smoke

# Production AWGN shard for one seed on a GPU.
uv run agfb-bench run --study B --seeds 0 --device cuda --out-dir runs/B

# Reduce finished shards to per-family statistics.
uv run agfb-bench aggregate --shard-dir runs/B --out runs/B/aggregate.parquet
```
