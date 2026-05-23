from __future__ import annotations

import torch

from agfb_noise import (
    NoiseSpec,
    add_dark_current,
    add_dead_pixels,
    add_fixed_pattern,
    add_gamma_speckle,
    add_gaussian,
    add_local_variance,
    add_noise,
    add_pepper,
    add_poisson,
    add_poisson_gaussian,
    add_quantization,
    add_random_impulse,
    add_rayleigh,
    add_rician,
    add_salt,
    add_salt_pepper,
    add_speckle,
    add_stripe,
    add_uniform,
    registered_noises,
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


def test_local_variance_accepts_per_pixel_variance() -> None:
    image = torch.zeros(1, 16, 16)
    variance = torch.full_like(image, 0.01)

    out = add_local_variance(image, variance=variance, seed=9)

    assert out.shape == image.shape


def test_uniform_noise_stays_within_half_width() -> None:
    image = torch.zeros(4, 16, 16)

    out = add_uniform(image, half_width=0.25, seed=2)

    assert torch.all(out >= -0.25)
    assert torch.all(out <= 0.25)


def test_poisson_family_preserves_shape_and_nonnegative_values() -> None:
    image = torch.full((2, 16, 16), 0.5)

    poisson = add_poisson(image, peak=100.0, seed=5)
    poisson_gaussian = add_poisson_gaussian(image, peak=100.0, read_sigma=0.01, seed=5)
    dark = add_dark_current(image, dark_rate=3.0, exposure_time=0.1, peak=100.0, seed=5)

    assert poisson.shape == image.shape
    assert poisson_gaussian.shape == image.shape
    assert dark.shape == image.shape
    assert torch.all(poisson >= 0.0)


def test_impulse_models_replace_requested_values() -> None:
    image = torch.full((1, 8, 8), 0.5)

    salt_pepper = add_salt_pepper(
        image,
        amount=1.0,
        salt_vs_pepper=0.25,
        salt_value=2.0,
        pepper_value=-1.0,
        seed=4,
    )
    random_impulse = add_random_impulse(image, amount=1.0, low=-2.0, high=2.0, seed=4)
    dead = add_dead_pixels(
        image,
        amount=1.0,
        hot_fraction=0.5,
        dead_value=-1.0,
        hot_value=2.0,
        seed=4,
    )

    assert set(salt_pepper.unique().tolist()) <= {-1.0, 2.0}
    assert torch.all(random_impulse >= -2.0)
    assert torch.all(random_impulse <= 2.0)
    assert set(dead.unique().tolist()) <= {-1.0, 2.0}


def test_salt_pepper_can_disable_one_side() -> None:
    image = torch.full((1, 8, 8), 0.5)

    salt_only = add_salt_pepper(
        image,
        amount=1.0,
        pepper=False,
        salt_value=2.0,
        pepper_value=-1.0,
        seed=4,
    )
    pepper_only = add_salt_pepper(
        image,
        amount=1.0,
        salt=False,
        salt_value=2.0,
        pepper_value=-1.0,
        seed=4,
    )
    disabled = add_salt_pepper(
        image,
        amount=1.0,
        salt=False,
        pepper=False,
        salt_value=2.0,
        pepper_value=-1.0,
        seed=4,
    )

    assert torch.equal(salt_only, torch.full_like(image, 2.0))
    assert torch.equal(pepper_only, torch.full_like(image, -1.0))
    assert torch.equal(disabled, image)


def test_salt_and_pepper_wrappers_match_salt_pepper_toggles() -> None:
    image = torch.full((1, 16, 16), 0.5)

    salt = add_salt(image, amount=0.25, salt_value=2.0, seed=7)
    salt_toggle = add_salt_pepper(image, amount=0.25, pepper=False, salt_value=2.0, seed=7)
    pepper = add_pepper(image, amount=0.25, pepper_value=-1.0, seed=7)
    pepper_toggle = add_salt_pepper(image, amount=0.25, salt=False, pepper_value=-1.0, seed=7)

    assert torch.equal(salt, salt_toggle)
    assert torch.equal(pepper, pepper_toggle)


def test_multiplicative_and_magnitude_models_preserve_shape() -> None:
    image = torch.ones(2, 16, 16)

    speckle = add_speckle(image, sigma=0.1, seed=6)
    gamma = add_gamma_speckle(image, looks=2, seed=6)
    rician = add_rician(image, sigma=0.1, seed=6)
    rayleigh = add_rayleigh(image, sigma=0.1, seed=6)

    assert speckle.shape == image.shape
    assert gamma.shape == image.shape
    assert rician.shape == image.shape
    assert rayleigh.shape == image.shape
    assert torch.all(rician >= 0.0)


def test_sensor_artifact_models_preserve_shape() -> None:
    image = torch.linspace(0.0, 1.0, steps=64).view(1, 8, 8)

    quantized = add_quantization(image, levels=4)
    fixed = add_fixed_pattern(image, offset_sigma=0.01, gain_sigma=0.01, seed=8)
    stripe = add_stripe(image, row_sigma=0.01, column_sigma=0.01, seed=8)

    assert quantized.shape == image.shape
    assert fixed.shape == image.shape
    assert stripe.shape == image.shape
    assert len(quantized.unique()) <= 4


def test_dispatch_accepts_noise_spec_and_aliases() -> None:
    image = torch.zeros(1, 8, 8)
    spec = NoiseSpec("gaussian", {"sigma": 0.1})

    out = add_noise(image, spec, seed=1)
    alias = add_noise(image, "awgn", sigma=0.1, seed=1)

    assert torch.equal(out, alias)
    assert "poisson_gaussian" in registered_noises()
