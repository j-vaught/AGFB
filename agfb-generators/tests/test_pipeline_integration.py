"""End-to-end integration with the PGF_paper/benchmark/prototypes pipeline.

Imports the *actual* prototype modules (SmoothedStep, masks, sobel3, cpgf_fft,
cpgf_kernels, nrmse_vector) and checks that:

  1. The new `smoothed_step` (B=1) renders bit-identical I/gx/gy to the
     prototype's SmoothedStep.render(), using the prototype module directly
     (not an inline re-implementation).
  2. Feeding the new frame's tensors into the prototype's masks(), sobel3(),
     cpgf_fft(), and nrmse_vector() reproduces the prototype-only NRMSE to
     numerical precision.

If either side moves, this test catches it before §1.1 production.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from cpgf_generators import smoothed_step

_PROTOTYPES_DIR = Path("/Users/user/Documents/New project/PGF_paper/benchmark/prototypes")


@pytest.fixture(scope="module")
def proto():
    """Load the external prototype modules used by integration tests.

    Pytest injects this fixture into the CPGF pipeline tests so they can compare
    package output with the original prototype render, masks, filters, kernels,
    and vector NRMSE metric when that sibling checkout is present.
    """
    if not _PROTOTYPES_DIR.exists():
        pytest.skip(f"benchmark prototypes not present at {_PROTOTYPES_DIR}")
    sys.path.insert(0, str(_PROTOTYPES_DIR))
    try:
        from cpgf_mini import cpgf_fft, cpgf_kernels  # type: ignore[import-not-found]
        from mini import SmoothedStep, masks, nrmse_vector, sobel3  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return {
        "SmoothedStep": SmoothedStep,
        "masks": masks,
        "nrmse_vector": nrmse_vector,
        "sobel3": sobel3,
        "cpgf_kernels": cpgf_kernels,
        "cpgf_fft": cpgf_fft,
    }


def test_smoothed_step_matches_actual_prototype_module(proto) -> None:
    """Verify package `smoothed_step` matches the imported prototype renderer."""
    device = torch.device("cpu")
    H = W = 256
    theta = math.radians(30.0)

    ref = proto["SmoothedStep"](H, W, theta_rad=theta, x0=0.0, contrast=1.0, sigma_e=2.0).render(
        device
    )
    out = smoothed_step(H, W, theta_rad=theta, x0=0.0, contrast=1.0, sigma_e=2.0, device=device)

    assert torch.equal(out.I[0], ref["I"])
    assert torch.equal(out.gx[0], ref["gx"])
    assert torch.equal(out.gy[0], ref["gy"])


def test_new_generator_feeds_prototype_pipeline_with_same_nrmse(proto) -> None:
    """Verify CPGF prototype metrics accept the package frame without drift."""
    device = torch.device("cpu")
    H = W = 256
    theta = math.radians(30.0)
    sigma_e = 2.0

    # Prototype-only path.
    ref = proto["SmoothedStep"](
        H, W, theta_rad=theta, sigma_e=sigma_e, x0=0.0, contrast=1.0
    ).render(device)
    m_ref = proto["masks"](ref["gx"], ref["gy"])
    sob_gx_ref, sob_gy_ref = proto["sobel3"](ref["I"])
    K_x, K_y = proto["cpgf_kernels"](r=5, d=3, device=device)
    cpgf_gx_ref, cpgf_gy_ref = proto["cpgf_fft"](ref["I"], K_x, K_y)
    n_sobel_ref = proto["nrmse_vector"](
        sob_gx_ref, sob_gy_ref, ref["gx"], ref["gy"], m_ref["signal"]
    )
    n_cpgf_ref = proto["nrmse_vector"](
        cpgf_gx_ref, cpgf_gy_ref, ref["gx"], ref["gy"], m_ref["signal"]
    )

    # New-generator path — same prototype filters/metric, but the frame comes
    # from cpgf_generators.
    new = smoothed_step(H, W, theta_rad=theta, sigma_e=sigma_e, device=device)
    I = new.I[0]
    gx_true = new.gx[0]
    gy_true = new.gy[0]
    m_new = proto["masks"](gx_true, gy_true)
    sob_gx, sob_gy = proto["sobel3"](I)
    cpgf_gx, cpgf_gy = proto["cpgf_fft"](I, K_x, K_y)
    n_sobel = proto["nrmse_vector"](sob_gx, sob_gy, gx_true, gy_true, m_new["signal"])
    n_cpgf = proto["nrmse_vector"](cpgf_gx, cpgf_gy, gx_true, gy_true, m_new["signal"])

    assert torch.equal(m_new["signal"], m_ref["signal"])
    assert torch.equal(m_new["background"], m_ref["background"])
    assert n_sobel == pytest.approx(n_sobel_ref, abs=0.0, rel=0.0)
    assert n_cpgf == pytest.approx(n_cpgf_ref, abs=0.0, rel=0.0)
