"""Regression checks for the notebook visualization helper module."""

from __future__ import annotations

import torch

from agfb_generators.notebook import (
    check_frame,
    generator_names,
    make_context,
    render_case,
    render_composite,
)


def test_notebook_context_lists_current_generators() -> None:
    """Check that the notebook exposes every currently implemented generator."""
    ctx = make_context(height=64, width=64, device=torch.device("cpu"))

    assert generator_names(ctx) == [
        "smoothed_step",
        "hard_step",
        "curved_arc",
        "sinusoid",
        "gaussian_blob",
        "gaussian_ridge",
        "smoothed_bar",
        "polynomial",
    ]


def test_notebook_case_renders_and_checks() -> None:
    """Check one notebook render path and its finite-difference metric."""
    ctx = make_context(height=96, width=96, device=torch.device("cpu"))

    frame = render_case(ctx, "gaussian_blob")
    metrics = check_frame("gaussian_blob", frame, rel_tol=1e-3)

    assert metrics["status"] == "pass"
    assert metrics["shape"] == "96x96"


def test_notebook_composite_renders() -> None:
    """Check the notebook composite example and junction mask shape."""
    ctx = make_context(height=96, width=96, device=torch.device("cpu"))

    frame, junction = render_composite(ctx)

    assert frame.I.shape == (1, 96, 96)
    assert frame.g.shape == (1, 2, 96, 96)
    assert junction.shape == (96, 96)
    assert bool(junction.any())
