"""Noise model definitions."""

from agfb_noise.definitions.correlated_gaussian import (
    add_anisotropic_gaussian,
    add_correlated_gaussian,
    add_correlated_speckle,
    add_powerlaw_gaussian,
)
from agfb_noise.definitions.dark_current import add_dark_current
from agfb_noise.definitions.dead_pixel import add_dead_pixels
from agfb_noise.definitions.fixed_pattern import add_fixed_pattern
from agfb_noise.definitions.gamma_speckle import add_gamma_speckle
from agfb_noise.definitions.gaussian import add_gaussian
from agfb_noise.definitions.local_variance import add_local_variance
from agfb_noise.definitions.pepper import add_pepper
from agfb_noise.definitions.poisson import add_poisson
from agfb_noise.definitions.poisson_gaussian import add_poisson_gaussian
from agfb_noise.definitions.quantization import add_quantization
from agfb_noise.definitions.random_impulse import add_random_impulse
from agfb_noise.definitions.rayleigh import add_rayleigh
from agfb_noise.definitions.rician import add_rician
from agfb_noise.definitions.salt import add_salt
from agfb_noise.definitions.salt_pepper import add_salt_pepper
from agfb_noise.definitions.speckle import add_speckle
from agfb_noise.definitions.stripe import add_stripe
from agfb_noise.definitions.uniform import add_uniform

__all__ = [
    "add_dark_current",
    "add_dead_pixels",
    "add_fixed_pattern",
    "add_gamma_speckle",
    "add_gaussian",
    "add_correlated_gaussian",
    "add_powerlaw_gaussian",
    "add_anisotropic_gaussian",
    "add_correlated_speckle",
    "add_local_variance",
    "add_pepper",
    "add_poisson",
    "add_poisson_gaussian",
    "add_quantization",
    "add_random_impulse",
    "add_rayleigh",
    "add_rician",
    "add_salt",
    "add_salt_pepper",
    "add_speckle",
    "add_stripe",
    "add_uniform",
]
