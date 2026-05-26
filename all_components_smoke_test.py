"""Run one smoke test across the four AGFB component packages."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch

COMPONENT_DIRS = (
    "agfb-generators",
    "agfb-noise",
    "agfb-filters",
    "agfb-metrics",
)


def has_components(candidate: Path) -> bool:
    return all((candidate / name).is_dir() for name in COMPONENT_DIRS)


def workspace_candidates(start: Path):
    for candidate in (start, *start.parents):
        yield candidate
        yield candidate / "AGFB"
        yield candidate / "Documents" / "New project" / "AGFB"


def find_workspace(start: Path | None = None) -> Path:
    search_start = Path.cwd().resolve() if start is None else start.resolve()
    for candidate in workspace_candidates(search_start):
        if has_components(candidate):
            return candidate
    fallback = Path("/Users/user/Documents/New project/AGFB")
    if has_components(fallback):
        return fallback
    raise RuntimeError("Could not find the AGFB workspace with all four component folders.")


def add_component_paths(workspace: Path) -> None:
    for component_dir in reversed(COMPONENT_DIRS):
        path = str(workspace / component_dir)
        if path not in sys.path:
            sys.path.insert(0, path)


def run_smoke_test() -> dict[str, Any]:
    workspace = find_workspace()
    add_component_paths(workspace)

    from agfb_filters import ExecutionPath, get_filter_definition, run_filter
    from agfb_generators import smoothed_step
    from agfb_metrics import evaluate_metrics, masks
    from agfb_noise import add_gaussian

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_sigma = 0.01

    angles = torch.tensor([0.0, math.pi / 8.0, math.pi / 4.0], dtype=torch.float32, device=device)
    frame = smoothed_step(128, 128, angle_rad=angles, edge_sigma=2.5, device=device)
    noisy = add_gaussian(frame.I, sigma=noise_sigma, seed=7, clamp=(0.0, 1.0))

    filter_definition = get_filter_definition("sobel_3")
    gradient_x, gradient_y = run_filter(
        filter_definition,
        noisy,
        path=ExecutionPath.SEPARABLE,
        boundary=filter_definition.default_boundary,
    )

    mask_dict = masks(frame.gx, frame.gy)
    scores = evaluate_metrics(
        gradient_x,
        gradient_y,
        frame.gx,
        frame.gy,
        metrics=(
            "nrmse",
            "angular_mae",
            "magnitude_bias",
            "noise_gain",
            "tail_spurious_grad",
        ),
        signal_mask=mask_dict["signal"],
        flat_mask=mask_dict["flat"],
        sigma_n=noise_sigma,
    )

    score_tensor = torch.stack([value for value in scores.values()])
    if not torch.isfinite(score_tensor).all():
        raise RuntimeError("Smoke-test metrics contain non-finite values.")

    score_table = []
    for sample_index, angle in enumerate(angles.detach().cpu().tolist()):
        row = {"sample": sample_index, "angle_rad": round(float(angle), 6)}
        for name, values in scores.items():
            row[name] = round(float(values.detach().cpu()[sample_index]), 6)
        score_table.append(row)

    return {
        "workspace": str(workspace),
        "device": str(device),
        "image_shape": tuple(frame.I.shape),
        "truth_gradient_shape": tuple(frame.g.shape),
        "filter": filter_definition.name,
        "execution_path": ExecutionPath.SEPARABLE.value,
        "score_table": score_table,
    }


def main() -> None:
    result = run_smoke_test()
    print(f"Workspace: {result['workspace']}")
    print(f"Device: {result['device']}")
    print(f"Image shape: {result['image_shape']}")
    print(f"Truth gradient shape: {result['truth_gradient_shape']}")
    print(f"Filter: {result['filter']} via {result['execution_path']}")
    print()
    print("sample angle_rad nrmse angular_mae magnitude_bias noise_gain tail_spurious_grad")
    for row in result["score_table"]:
        print(
            f"{row['sample']} "
            f"{row['angle_rad']:.6f} "
            f"{row['nrmse']:.6f} "
            f"{row['angular_mae']:.6f} "
            f"{row['magnitude_bias']:.6f} "
            f"{row['noise_gain']:.6f} "
            f"{row['tail_spurious_grad']:.6f}"
        )


if __name__ == "__main__":
    main()
