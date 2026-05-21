from __future__ import annotations

from pathlib import Path

from agfb_filters import (
    AutoRunner,
    BenchmarkConfig,
    BoundaryCondition,
    BoundaryMode,
    ExecutionPath,
    ExecutionPlan,
    InputSignature,
    cpgf_definition,
    sobel_definition,
)


def test_estimate_best_returns_concrete_path() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=1, height=16, width=17)
    definition = sobel_definition(3)
    plan = runner.estimate_best(
        definition,
        signature,
        boundary=definition.default_boundary,
    )

    assert isinstance(plan.path, ExecutionPath)
    assert plan.path == ExecutionPath.SPATIAL_DENSE
    assert plan.path.value != "auto"
    assert plan.boundary == definition.default_boundary


def test_estimate_best_keeps_wider_separable_filters_on_separable_path() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=8, height=256, width=256)
    definition = sobel_definition(5)

    plan = runner.estimate_best(
        definition,
        signature,
        boundary=definition.default_boundary,
    )

    assert plan.path == ExecutionPath.SEPARABLE


def test_estimate_best_uses_large_dense_kernel_paths() -> None:
    runner = AutoRunner()
    large_signature = InputSignature.from_values(batch=8, height=256, width=256)
    small_signature = InputSignature.from_values(batch=1, height=106, width=160)
    definition = cpgf_definition(radius=2, degree=2)

    large_cpgf_plan = runner.estimate_best(
        definition,
        large_signature,
        boundary=definition.default_boundary,
    )
    small_cpgf_plan = runner.estimate_best(
        definition,
        small_signature,
        boundary=definition.default_boundary,
    )

    assert large_cpgf_plan.path == ExecutionPath.ANTIPODAL_PAIRS
    assert small_cpgf_plan.path == ExecutionPath.SPATIAL_DENSE


def test_cached_best_stores_memory_cache_on_miss_and_hit() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=1, height=16, width=17)
    definition = cpgf_definition(radius=2, degree=2)

    cache_key = runner.cache_key(definition, signature, boundary=definition.default_boundary)
    assert cache_key not in runner._memory_cache
    first_plan = runner.cached_best(definition, signature, boundary=definition.default_boundary)
    assert cache_key in runner._memory_cache
    second_plan = runner.cached_best(definition, signature, boundary=definition.default_boundary)

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
    first_plan = first_runner.cached_best(
        definition,
        signature,
        boundary=definition.default_boundary,
    )

    second_runner = AutoRunner(cache_path=cache_path)
    second_plan = second_runner.cached_best(
        definition,
        signature,
        boundary=definition.default_boundary,
    )

    assert cache_path.exists()
    assert first_plan.to_json_dict()["boundary"] == definition.default_boundary.to_json_dict()
    assert ExecutionPlan.from_json_dict(first_plan.to_json_dict()) == first_plan
    assert second_plan == first_plan


def test_benchmark_best_returns_empirical_result() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=1, height=8, width=9)
    definition = cpgf_definition(radius=2, degree=2)
    plan = runner.benchmark_best(
        definition,
        signature,
        boundary=definition.default_boundary,
        benchmark_config=BenchmarkConfig(
            candidate_paths=(ExecutionPath.SPATIAL_DENSE, ExecutionPath.SPARSE_OFFSETS),
            warmup_runs=0,
            min_run_time=0.001,
        ),
    )

    assert plan.path in {ExecutionPath.SPATIAL_DENSE, ExecutionPath.SPARSE_OFFSETS}
    assert plan.benchmark_result is not None
    assert plan.benchmark_result.median_seconds > 0
    assert all(result.path != "auto" for result in plan.benchmark_results)
    assert plan.boundary == definition.default_boundary


def test_different_boundaries_produce_different_cache_keys() -> None:
    runner = AutoRunner()
    signature = InputSignature.from_values(batch=1, height=16, width=17)
    definition = sobel_definition(3)

    replicate_key = runner.cache_key(
        definition,
        signature,
        boundary=BoundaryCondition(BoundaryMode.REPLICATE),
    )
    constant_zero_key = runner.cache_key(
        definition,
        signature,
        boundary=BoundaryCondition(BoundaryMode.CONSTANT),
    )
    constant_nonzero_key = runner.cache_key(
        definition,
        signature,
        boundary=BoundaryCondition(BoundaryMode.CONSTANT, value=2.0),
    )

    assert replicate_key != constant_zero_key
    assert constant_zero_key != constant_nonzero_key
