# Analytic Gradient Filter Benchmark Rename Plan

This note records the preferred generic naming for the benchmark suite. The formal project name is Analytic Gradient Filter Benchmark, abbreviated as AGFB in package and repository names. The goal is to describe the suite as a reusable benchmark for analytic gradient-filter evaluation rather than as infrastructure tied to one filter family.

The generator component is named `agfb-generators`, with the Python package named `agfb_generators`. This package owns analytic stimulus generation. Each generator produces a closed-form intensity field and an analytic ground-truth gradient field for filter evaluation.

The filter component should be named `agfb-operators`, with the Python package named `agfb_operators`. This package owns built-in reference filters, candidate filter definitions, shared operator interfaces, and utilities for applying operators to benchmark frames.

The scoring component should be named `agfb-metrics`, with the Python package named `agfb_metrics`. This package owns masks, per-frame scoring functions, metric aggregation, and ranking logic.

The execution component should be named `agfb-pipeline`, with the Python package named `agfb_pipeline`. This package owns reproducible benchmark execution, manifests, sweeps, cached results, result tables, and report artifact generation. It should be the canonical headless workflow for published benchmark runs.

The interactive development component should be named `agfb-workbench`, with the Python package named `agfb_workbench`. This package owns visual inspection and debugging tools for new filters. It should support generator previews, operator views, metric error maps, side-by-side comparisons, and notebooks or dashboards for exploratory analysis. It should depend on the pipeline outputs where possible rather than replacing the reproducible pipeline.

The preferred command surface should make the pipeline and workbench feel like one suite. Pipeline commands should cover generation, scoring, ranking, and reporting. Workbench commands should cover inspecting generators, inspecting operators, inspecting metric outputs, and comparing a new filter against built-in baselines.

The implementation order should start with `agfb-generators` because this repository already contains that component. After that, update dependent packages in order. Rename imports in the operators, metrics, band-limit experiment, pipeline, tests, and documentation after the generator package provides the new import path.
