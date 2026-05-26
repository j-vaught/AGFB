from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for component in ("agfb-generators", "agfb-noise", "agfb-filters"):
    component_path = str(PROJECT_ROOT / component)
    if component_path not in sys.path:
        sys.path.insert(0, component_path)

from agfb_filters import ExecutionPath, get_filter_definition, run_filter  # noqa: E402
from agfb_generators import smoothed_step  # noqa: E402
from agfb_noise import NoiseSpec, apply_noise_sequence  # noqa: E402

from agfb_metrics.metrics import ALL_METRICS, evaluate_all_metrics, masks  # noqa: E402


def test_generator_noise_filter_metric_pipeline_reports_all_metrics() -> None:
    height = 64
    width = 64
    sigma_n = 0.02
    frame = smoothed_step(
        height,
        width,
        angle_rad=torch.tensor([0.0, math.pi / 6.0, math.pi / 3.0]),
        center_offset=torch.tensor([-4.0, 0.0, 4.0]),
        edge_sigma=torch.tensor([2.0, 2.5, 3.0]),
    )
    noisy = apply_noise_sequence(
        frame.I,
        (
            NoiseSpec("gaussian", {"sigma": sigma_n}),
            NoiseSpec("quantization", {"levels": 256}),
        ),
        seed=24,
        clamp=(0.0, 1.0),
    )

    definition = get_filter_definition("sobel_3")
    gradient_x, gradient_y = run_filter(
        definition,
        noisy,
        path=ExecutionPath.SEPARABLE,
        boundary=definition.default_boundary,
    )
    mask_dict = masks(frame.gx, frame.gy, dilate_px=4, rel_eps=1e-3)

    scores = evaluate_all_metrics(
        gradient_x,
        gradient_y,
        frame.gx,
        frame.gy,
        signal_mask=mask_dict["signal"],
        flat_mask=mask_dict["flat"],
        sigma_n=sigma_n,
        r_max=6.0,
        step=1.0,
    )

    assert set(scores) == set(ALL_METRICS)
    assert not torch.equal(noisy, frame.I)
    assert gradient_x.shape == frame.I.shape
    assert gradient_y.shape == frame.I.shape
    for metric_name, values in scores.items():
        assert values.shape == (frame.batch_size,)
        assert values.dtype == torch.float32
        assert torch.isfinite(values).all(), metric_name
    assert (scores["nrmse"] >= 0.0).all()
    assert (scores["angular_mae"] >= 0.0).all()
    assert (scores["noise_gain"] >= 0.0).all()
