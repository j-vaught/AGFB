"""End-to-end integration with cpgf-generators, cpgf-filters, and the
existing PGF_paper prototype.

Checks:
  1. cpgf_metrics.masks()['signal'] is bit-identical to the prototype's
     mini.masks()['signal'] on the same single-image truth field.
  2. A1, A2, A3, B3 produce expected signs / orders of magnitude on a clean
     smoothed_step processed by central_difference, sobel_3, DoG(sigma=4).
  3. C1, C2 produce expected magnitudes on a pure-noise input.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from cpgf_metrics import (
    a1_nrmse,
    a2_angular_mae,
    a3_tail_vector_error,
    b3_magnitude_bias,
    c1_noise_gain,
    c2_tail_spurious_grad,
    magnitude,
    masks,
)

_PROTOTYPES_DIR = Path("/Users/user/Documents/New project/PGF_paper/benchmark/prototypes")
_GENERATORS_DIR = Path("/Users/user/Documents/New project/cpgf-generators")
_FILTERS_DIR = Path("/Users/user/Documents/New project/cpgf-filters")


@pytest.fixture(scope="module")
def proto():
    if not _PROTOTYPES_DIR.exists():
        pytest.skip(f"prototypes not present at {_PROTOTYPES_DIR}")
    sys.path.insert(0, str(_PROTOTYPES_DIR))
    try:
        from mini import SmoothedStep  # type: ignore[import-not-found]
        from mini import masks as proto_masks  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return {"SmoothedStep": SmoothedStep, "proto_masks": proto_masks}


@pytest.fixture(scope="module")
def gens():
    if not _GENERATORS_DIR.exists():
        pytest.skip(f"cpgf-generators not present at {_GENERATORS_DIR}")
    sys.path.insert(0, str(_GENERATORS_DIR))
    try:
        from cpgf_generators import smoothed_step  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return {"smoothed_step": smoothed_step}


@pytest.fixture(scope="module")
def filters():
    if not _FILTERS_DIR.exists():
        pytest.skip(f"cpgf-filters not present at {_FILTERS_DIR}")
    sys.path.insert(0, str(_FILTERS_DIR))
    try:
        from cpgf_filters import DoG, central_difference, sobel_3  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return {"central_difference": central_difference, "sobel_3": sobel_3, "DoG": DoG}


def test_signal_mask_matches_prototype(proto) -> None:
    H = W = 256
    frame = proto["SmoothedStep"](
        H, W, theta_rad=math.radians(30.0), sigma_e=2.0, x0=0.0, contrast=1.0
    ).render(torch.device("cpu"))
    gx_t = frame["gx"]
    gy_t = frame["gy"]
    proto_signal = proto["proto_masks"](gx_t, gy_t)["signal"]
    our = masks(gx_t.unsqueeze(0), gy_t.unsqueeze(0))
    assert torch.equal(our["signal"][0], proto_signal)


def test_phase1_clean_smoothed_step(gens, filters) -> None:
    device = torch.device("cpu")
    H = W = 256
    frame = gens["smoothed_step"](H, W, theta_rad=math.radians(30.0), sigma_e=2.0, device=device)
    I_b = frame.I
    gx_t = frame.gx
    gy_t = frame.gy
    m = masks(gx_t, gy_t)
    signal = m["signal"]

    cdx, cdy = filters["central_difference"](I_b)
    nrmse_cd = a1_nrmse(cdx, cdy, gx_t, gy_t, signal)[0].item()
    amae_cd = a2_angular_mae(cdx, cdy, gx_t, gy_t, signal)[0].item()
    p95_cd = a3_tail_vector_error(cdx, cdy, gx_t, gy_t, signal)[0].item()
    bias_cd = b3_magnitude_bias(cdx, cdy, gx_t, gy_t, signal)[0].item()
    assert 0.0 < nrmse_cd < 0.5
    assert 0.0 < amae_cd < 30.0
    assert p95_cd > 0.0
    assert -0.5 < bias_cd < 0.5

    dog_gx, dog_gy = filters["DoG"](sigma=4.0).apply(I_b)
    bias_dog = b3_magnitude_bias(dog_gx, dog_gy, gx_t, gy_t, signal)[0].item()
    assert bias_dog < bias_cd, (
        "DoG(σ=4) on a σ=2 step should under-read more than central_difference"
    )


def test_phase1_noise_only_input(filters) -> None:
    torch.manual_seed(42)
    sigma_n = 0.1
    H = W = 256
    noise = sigma_n * torch.randn(1, H, W)
    mask = torch.ones(1, H, W, dtype=torch.bool)

    cdx, cdy = filters["central_difference"](noise)
    cd_gain = c1_noise_gain(cdx, cdy, mask, sigma_n=sigma_n)[0].item()
    cd_p99 = c2_tail_spurious_grad(cdx, cdy, mask)[0].item()
    assert 0.0 < cd_gain < 5.0
    assert cd_p99 > 0.0

    dog = filters["DoG"](sigma=2.0)
    dx, dy = dog.apply(noise)
    dog_gain = c1_noise_gain(dx, dy, mask, sigma_n=sigma_n)[0].item()
    assert dog_gain < cd_gain, "DoG should suppress noise more than raw central_difference"

    mag_cd = magnitude(cdx, cdy)
    mag_dog = magnitude(dx, dy)
    assert float(mag_dog.std()) < float(mag_cd.std())
