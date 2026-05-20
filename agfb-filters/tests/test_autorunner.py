from __future__ import annotations

from pathlib import Path

from agfb_filters import (
    AutoRunner,
    BenchmarkConfig,
    ExecutionPath,
    ExecutionPlan,
    InputSignature,
    cpgf_definition,
    sobel_definition,
)


def test_estimate_best_returns_concrete_path() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=1, height=16, width=17)
    plan = runner.estimate_best(sobel_definition(3), signature)

    assert isinstance(plan.path, ExecutionPath)
    assert plan.path == ExecutionPath.SEPARABLE
    assert plan.path.value != "auto"


def test_cached_best_stores_memory_cache_on_miss_and_hit() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=1, height=16, width=17)
    definition = cpgf_definition(radius=2, degree=2)

    cache_key = runner.cache_key(definition, signature)
    assert cache_key not in runner._memory_cache
    first_plan = runner.cached_best(definition, signature)
    assert cache_key in runner._memory_cache
    second_plan = runner.cached_best(definition, signature)

    assert first_plan == second_plan
    assert first_plan.path in {
        ExecutionPath.SPATIAL_DENSE,
        ExecutionPath.SPARSE_OFFSETS,
        ExecutionPath.ANTIPODAL_PAIRS,
        ExecutionPath.FFT,
    }


def test_json_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "autorunner-cache.json"
    signature = InputSignature.from_values(batch=1, height=16, width=17)
    definition = cpgf_definition(radius=2, degree=2)
    first_runner = AutoRunner(cache_path=cache_path)
    first_plan = first_runner.cached_best(definition, signature)

    second_runner = AutoRunner(cache_path=cache_path)
    second_plan = second_runner.cached_best(definition, signature)

    assert cache_path.exists()
    assert ExecutionPlan.from_json_dict(first_plan.to_json_dict()) == first_plan
    assert second_plan == first_plan


def test_benchmark_best_returns_empirical_result() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=1, height=8, width=9)
    definition = cpgf_definition(radius=2, degree=2)
    plan = runner.benchmark_best(
        definition,
        signature,
        BenchmarkConfig(
            candidate_paths=(ExecutionPath.SPATIAL_DENSE, ExecutionPath.SPARSE_OFFSETS),
            warmup_runs=0,
            min_run_time=0.001,
        ),
    )

    assert plan.path in {ExecutionPath.SPATIAL_DENSE, ExecutionPath.SPARSE_OFFSETS}
    assert plan.benchmark_result is not None
    assert plan.benchmark_result.median_seconds > 0
    assert all(result.path != "auto" for result in plan.benchmark_results)
