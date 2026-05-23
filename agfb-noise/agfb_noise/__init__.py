"""Fast batched noise models for AGFB images."""

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
from agfb_noise.helpers.base import ClampRange, Numeric
from agfb_noise.helpers.catalog import BuiltInNoiseSpec, shipped_noise_specs
from agfb_noise.helpers.registry import (
    NoiseRegistration,
    get_noise_registration,
    register_noise,
    registered_noises,
)
from agfb_noise.helpers.runner import NoiseSpec, add_noise, apply_noise_sequence

__all__ = [
    "BuiltInNoiseSpec",
    "ClampRange",
    "NoiseRegistration",
    "NoiseSpec",
    "Numeric",
    "add_dark_current",
    "add_dead_pixels",
    "add_fixed_pattern",
    "add_gamma_speckle",
    "add_gaussian",
    "add_local_variance",
    "add_noise",
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
    "apply_noise_sequence",
    "get_noise_registration",
    "registered_noises",
    "register_noise",
    "shipped_noise_specs",
]
