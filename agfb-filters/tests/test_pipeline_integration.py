"""End-to-end integration with the PGF_paper/benchmark/prototypes pipeline
and with the cpgf-generators package.

Checks:
  1. `sobel_3` on a smoothed_step is bit-identical to the prototype's `sobel3`.
  2. `CPGF(r=5, d=3)` on a smoothed_step reproduces the prototype's CPGF NRMSE
     bit-exactly.
  3. Every filter applied to a smoothed_step produces an A1 NRMSE that is
     finite and within a reasonable upper bound.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from cpgf_filters import (
    CPGF,
    DoG,
    FreemanAdelsonG1,
    SavitzkyGolay,
    central_difference,
    farid_simoncelli_5,
    prewitt_3,
    roberts,
    scharr_3,
    sobel_3,
    sobel_5,
    sobel_7,
)

_PROTOTYPES_DIR = Path("/Users/user/Documents/New project/PGF_paper/benchmark/prototypes")
_GENERATORS_DIR = Path("/Users/user/Documents/New project/cpgf-generators")


@pytest.fixture(scope="module")
def proto():
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


@pytest.fixture(scope="module")
def generators():
    if not _GENERATORS_DIR.exists():
        pytest.skip(f"cpgf-generators not present at {_GENERATORS_DIR}")
    sys.path.insert(0, str(_GENERATORS_DIR))
    try:
        from cpgf_generators import smoothed_step  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return {"smoothed_step": smoothed_step}


def test_sobel_3_bit_identical_to_prototype(proto) -> None:
    device = torch.device("cpu")
    H = W = 256
    theta = math.radians(30.0)
    ref_frame = proto["SmoothedStep"](
        H, W, theta_rad=theta, sigma_e=2.0, x0=0.0, contrast=1.0
    ).render(device)
    ref_gx, ref_gy = proto["sobel3"](ref_frame["I"])
    I_batched = ref_frame["I"].unsqueeze(0)
    gx, gy = sobel_3(I_batched)
    assert torch.equal(gx[0], ref_gx)
    assert torch.equal(gy[0], ref_gy)


def test_cpgf_matches_prototype_nrmse(proto) -> None:
    device = torch.device("cpu")
    H = W = 256
    theta = math.radians(30.0)
    ref_frame = proto["SmoothedStep"](
        H, W, theta_rad=theta, sigma_e=2.0, x0=0.0, contrast=1.0
    ).render(device)
    m = proto["masks"](ref_frame["gx"], ref_frame["gy"])

    K_x, K_y = proto["cpgf_kernels"](r=5, d=3, device=device)
    cpgf_gx_ref, cpgf_gy_ref = proto["cpgf_fft"](ref_frame["I"], K_x, K_y)
    nrmse_ref = proto["nrmse_vector"](
        cpgf_gx_ref, cpgf_gy_ref, ref_frame["gx"], ref_frame["gy"], m["signal"]
    )

    flt = CPGF(r=5, d=3, device=device)
    I_batched = ref_frame["I"].unsqueeze(0)
    gx, gy = flt.apply(I_batched)
    nrmse_new = proto["nrmse_vector"](gx[0], gy[0], ref_frame["gx"], ref_frame["gy"], m["signal"])
    assert nrmse_new == pytest.approx(nrmse_ref, abs=1e-6, rel=0.0)


def _nrmse(gx, gy, gx_t, gy_t, signal) -> float:
    dx = (gx - gx_t)[signal]
    dy = (gy - gy_t)[signal]
    num = torch.mean(dx * dx + dy * dy)
    den = torch.mean(gx_t[signal] ** 2 + gy_t[signal] ** 2)
    return float(torch.sqrt(num / den))


def test_all_filters_finite_nrmse_on_smoothed_step(generators) -> None:
    """Sanity-rank: every filter must produce a finite A1 NRMSE < 2.0 on the
    σ=2 smoothed step. Anything wildly larger flags a sign or kernel-axis bug.
    """
    device = torch.device("cpu")
    H = W = 256
    theta = math.radians(30.0)
    frame = generators["smoothed_step"](H, W, theta_rad=theta, sigma_e=2.0, device=device)
    I_b = frame.I
    gx_t = frame.gx[0]
    gy_t = frame.gy[0]
    mag = torch.sqrt(gx_t * gx_t + gy_t * gy_t)
    signal = mag > 1e-3 * float(mag.max())
    assert int(signal.sum()) > 200

    filters = [
        ("central_difference", lambda I: central_difference(I)),
        ("sobel_3", lambda I: sobel_3(I)),
        ("sobel_5", lambda I: sobel_5(I)),
        ("sobel_7", lambda I: sobel_7(I)),
        ("prewitt_3", lambda I: prewitt_3(I)),
        ("scharr_3", lambda I: scharr_3(I)),
        ("roberts", lambda I: roberts(I)),
        ("farid_simoncelli_5", lambda I: farid_simoncelli_5(I)),
        ("DoG(1.0)", DoG(sigma=1.0).apply),
        ("DoG(2.0)", DoG(sigma=2.0).apply),
        ("SavitzkyGolay(r=3,d=3)", SavitzkyGolay(r=3, d=3).apply),
        ("SavitzkyGolay(r=5,d=3)", SavitzkyGolay(r=5, d=3).apply),
        ("CPGF(r=3,d=3)", CPGF(r=3, d=3, device=device).apply),
        ("CPGF(r=5,d=3)", CPGF(r=5, d=3, device=device).apply),
        ("FreemanAdelsonG1(1.5)", FreemanAdelsonG1(sigma=1.5).apply),
    ]
    for name, fn in filters:
        gx, gy = fn(I_b)
        n = _nrmse(gx[0], gy[0], gx_t, gy_t, signal)
        assert math.isfinite(n), f"{name}: NRMSE is not finite"
        assert n < 2.0, f"{name}: NRMSE={n:.3f} is implausibly large"
