from __future__ import annotations

import torch

from agfb_noise import (
    NoiseSpec,
    add_gaussian,
    add_noise,
    add_poisson,
    add_salt_pepper,
    add_uniform,
)


def test_gaussian_noise_is_reproducible_with_seed() -> None:
    image = torch.zeros(2, 16, 16)

    first = add_gaussian(image, sigma=0.1, seed=7)
    second = add_gaussian(image, sigma=0.1, seed=7)

    assert torch.equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype


def test_batched_sigma_scales_each_image() -> None:
    image = torch.zeros(2, 2048)
    sigma = torch.tensor([0.01, 0.2])

    out = add_gaussian(image, sigma=sigma, seed=3)
    std = out.std(dim=1, unbiased=False)

    assert std[1] > std[0] * 10.0


def test_uniform_noise_stays_within_half_width() -> None:
    image = torch.zeros(4, 16, 16)

    out = add_uniform(image, half_width=0.25, seed=2)

    assert torch.all(out >= -0.25)
    assert torch.all(out <= 0.25)


def test_salt_pepper_replaces_requested_values() -> None:
    image = torch.full((1, 8, 8), 0.5)

    out = add_salt_pepper(
        image,
        amount=1.0,
        salt_vs_pepper=0.25,
        salt_value=2.0,
        pepper_value=-1.0,
        seed=4,
    )

    assert set(out.unique().tolist()) <= {-1.0, 2.0}


def test_poisson_noise_preserves_shape_and_nonnegative_values() -> None:
    image = torch.full((2, 16, 16), 0.5)

    out = add_poisson(image, peak=100.0, seed=5)

    assert out.shape == image.shape
    assert torch.all(out >= 0.0)


def test_dispatch_accepts_noise_spec() -> None:
    image = torch.zeros(1, 8, 8)
    spec = NoiseSpec("gaussian", {"sigma": 0.1})

    out = add_noise(image, spec, seed=1)

    assert out.shape == image.shape
