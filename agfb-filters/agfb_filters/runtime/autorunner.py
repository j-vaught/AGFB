"""Path recommendation and benchmarking for explicit AGFB execution."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import torch
import torch.version as torch_version
from torch.utils.benchmark import Timer

from agfb_filters.filters.definitions import GradientFilterDefinition
from agfb_filters.runtime.execution import (
    PATH_VERSION,
    BenchmarkConfig,
    BenchmarkResult,
    BoundaryCondition,
    ExecutionPath,
    ExecutionPlan,
    InputSignature,
)
from agfb_filters.runtime.runner import run_filter


class AutoRunner:
    """Recommend or benchmark a concrete execution path without implicit running."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache_path = None if cache_path is None else Path(cache_path)
        self._memory_cache: dict[str, ExecutionPlan] = {}
        if self.cache_path is not None:
            self._load_json_cache()

    def estimate_best(
        self,
        definition: GradientFilterDefinition,
        input_signature: InputSignature,
        *,
        boundary: BoundaryCondition,
    ) -> ExecutionPlan:
        """Return a heuristic concrete path recommendation."""
        boundary = _require_boundary_condition(boundary)
        path, estimated_cost, reason = self._estimate_path(definition, input_signature)
        return ExecutionPlan(
            path=path,
            input_signature=input_signature,
            boundary=boundary,
            filter_fingerprint=definition.fingerprint(),
            reason=reason,
            estimated_cost=estimated_cost,
        )

    def cached_best(
        self,
        definition: GradientFilterDefinition,
        input_signature: InputSignature,
        *,
        boundary: BoundaryCondition,
    ) -> ExecutionPlan:
        """Return a cached plan, estimating and storing one on a miss."""
        boundary = _require_boundary_condition(boundary)
        cache_key = self.cache_key(definition, input_signature, boundary=boundary)
        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return cached

        plan = self.estimate_best(definition, input_signature, boundary=boundary)
        self._memory_cache[cache_key] = plan
        self._write_json_cache()
        return plan

    def benchmark_best(
        self,
        definition: GradientFilterDefinition,
        input_signature: InputSignature,
        *,
        boundary: BoundaryCondition,
        benchmark_config: BenchmarkConfig | None = None,
    ) -> ExecutionPlan:
        """Benchmark valid candidate paths on synthetic input and cache the winner."""
        boundary = _require_boundary_condition(boundary)
        config = BenchmarkConfig() if benchmark_config is None else benchmark_config
        image = torch.randn(
            input_signature.batch,
            input_signature.height,
            input_signature.width,
            dtype=input_signature.torch_dtype,
            device=input_signature.torch_device,
            requires_grad=input_signature.requires_grad,
        )
        candidates = config.candidate_paths or tuple(self.valid_paths(definition))
        results: list[BenchmarkResult] = []
        for path in candidates:
            try:
                for _ in range(config.warmup_runs):
                    run_filter(definition, image, path=path, boundary=boundary)
                    _synchronize_if_needed(image)
                timer = Timer(
                    stmt="run_filter(definition, image, path=path, boundary=boundary)",
                    globals={
                        "definition": definition,
                        "image": image,
                        "path": path,
                        "boundary": boundary,
                        "run_filter": run_filter,
                    },
                )
                measurement = timer.blocked_autorange(min_run_time=config.min_run_time)
                results.append(
                    BenchmarkResult(
                        path=path,
                        median_seconds=float(measurement.median),
                        iqr_seconds=float(measurement.iqr),
                        number_per_run=int(measurement.number_per_run),
                        rounds=len(measurement.raw_times),
                    )
                )
            except (RuntimeError, ValueError):
                continue

        if not results:
            raise ValueError(f"no benchmarkable execution paths for {definition.name}")

        best_result = min(results, key=lambda result: result.median_seconds)
        plan = ExecutionPlan(
            path=best_result.path,
            input_signature=input_signature,
            boundary=boundary,
            filter_fingerprint=definition.fingerprint(),
            reason="empirical benchmark",
            estimated_cost=best_result.median_seconds,
            benchmark_result=best_result,
            benchmark_results=tuple(results),
        )
        self._memory_cache[self.cache_key(definition, input_signature, boundary=boundary)] = plan
        self._write_json_cache()
        return plan

    def valid_paths(self, definition: GradientFilterDefinition) -> list[ExecutionPath]:
        """Return structurally valid paths for a definition."""
        paths: list[ExecutionPath] = []
        if definition.has_separable_kernels:
            paths.append(ExecutionPath.SEPARABLE)
        if not definition.has_dense_kernels:
            return paths

        kernel_x, kernel_y = definition.dense_kernels()
        if kernel_x.shape != kernel_y.shape or kernel_x.ndim != 2:
            return paths

        paths.extend(
            [
                ExecutionPath.SPATIAL_DENSE,
                ExecutionPath.FFT,
                ExecutionPath.SPARSE_OFFSETS,
            ]
        )
        kernel_height, kernel_width = kernel_x.shape
        if kernel_height <= 3 and kernel_width <= 3:
            paths.append(ExecutionPath.STENCIL)
        if _is_antipodal_candidate(kernel_x, definition):
            paths.append(ExecutionPath.ANTIPODAL_PAIRS)
        return paths

    def cache_key(
        self,
        definition: GradientFilterDefinition,
        input_signature: InputSignature,
        *,
        boundary: BoundaryCondition,
    ) -> str:
        """Return the runtime-sensitive cache key for a filter/input pair."""
        boundary = _require_boundary_condition(boundary)
        payload = {
            "filter_fingerprint": definition.fingerprint(),
            "boundary": boundary.to_json_dict(),
            "path_version": PATH_VERSION,
            "input_signature": input_signature.to_json_dict(),
            "torch_version": torch.__version__,
            "cuda_version": torch_version.cuda or "",
            "cpu_thread_count": torch.get_num_threads(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _estimate_path(
        self,
        definition: GradientFilterDefinition,
        input_signature: InputSignature,
    ) -> tuple[ExecutionPath, float, str]:
        valid_paths = self.valid_paths(definition)
        if not valid_paths:
            raise ValueError(f"no valid execution paths for {definition.name}")

        if ExecutionPath.SEPARABLE in valid_paths:
            _, derivative_kernel = definition.separable_kernels()
            kernel_size = int(derivative_kernel.shape[0])
            cost = (
                input_signature.batch * input_signature.height * input_signature.width * kernel_size
            )
            return ExecutionPath.SEPARABLE, float(cost), "separable kernels are available"

        kernel_x, kernel_y = definition.dense_kernels()
        kernel_height, kernel_width = kernel_x.shape
        kernel_area = int(kernel_height * kernel_width)
        image_area = input_signature.height * input_signature.width
        nonzero_count = int(torch.count_nonzero((kernel_x != 0) | (kernel_y != 0)).item())
        zero_fraction = 1.0 - nonzero_count / kernel_area

        if ExecutionPath.STENCIL in valid_paths:
            cost = input_signature.batch * image_area * max(nonzero_count, 1)
            return ExecutionPath.STENCIL, float(cost), "tiny stencil kernel"
        if zero_fraction >= 0.15 and ExecutionPath.SPARSE_OFFSETS in valid_paths:
            cost = input_signature.batch * image_area * max(nonzero_count, 1)
            return ExecutionPath.SPARSE_OFFSETS, float(cost), "sparse support"
        if max(kernel_height, kernel_width) >= 11 and ExecutionPath.FFT in valid_paths:
            cost = input_signature.batch * image_area * float(max(kernel_height, kernel_width))
            return ExecutionPath.FFT, float(cost), "large dense kernel"
        if ExecutionPath.ANTIPODAL_PAIRS in valid_paths:
            cost = input_signature.batch * image_area * max(nonzero_count // 2, 1)
            return ExecutionPath.ANTIPODAL_PAIRS, float(cost), "odd-symmetric derivative kernel"
        cost = input_signature.batch * image_area * kernel_area
        return ExecutionPath.SPATIAL_DENSE, float(cost), "dense spatial reference"

    def _load_json_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        plans = data.get("plans", {})
        if not isinstance(plans, dict):
            return
        for key, plan_data in plans.items():
            if isinstance(key, str) and isinstance(plan_data, dict):
                try:
                    self._memory_cache[key] = ExecutionPlan.from_json_dict(plan_data)
                except (KeyError, TypeError, ValueError):
                    continue

    def _write_json_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "path_version": PATH_VERSION,
            "plans": {key: plan.to_json_dict() for key, plan in sorted(self._memory_cache.items())},
        }
        self.cache_path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _is_antipodal_candidate(
    kernel: torch.Tensor,
    definition: GradientFilterDefinition,
) -> bool:
    if definition.spatial_padding is not None:
        left, right, top, bottom = definition.spatial_padding
        if left != right or top != bottom:
            return False
    kernel_height, kernel_width = kernel.shape
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        return False
    scale = max(float(kernel.abs().max()), 1.0)
    return bool(
        torch.allclose(
            kernel,
            -torch.flip(kernel, dims=(0, 1)),
            atol=1e-5 * scale,
            rtol=0.0,
        )
    )


def _require_boundary_condition(boundary: BoundaryCondition) -> BoundaryCondition:
    if not isinstance(boundary, BoundaryCondition):
        raise ValueError("boundary must be a BoundaryCondition")
    return boundary


def _synchronize_if_needed(image: torch.Tensor) -> None:
    if image.device.type == "cuda":
        torch.cuda.synchronize(image.device)
