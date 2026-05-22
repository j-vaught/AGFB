"""Fast batched noise models for AGFB images."""

from agfb_noise.base import ClampRange, Numeric
from agfb_noise.catalog import BuiltInNoiseSpec, shipped_noise_specs
from agfb_noise.dark_current import add_dark_current
from agfb_noise.dead_pixel import add_dead_pixels
from agfb_noise.fixed_pattern import add_fixed_pattern
from agfb_noise.gamma_speckle import add_gamma_speckle
from agfb_noise.gaussian import add_gaussian
from agfb_noise.local_variance import add_local_variance
from agfb_noise.pepper import add_pepper
from agfb_noise.poisson import add_poisson
from agfb_noise.poisson_gaussian import add_poisson_gaussian
from agfb_noise.quantization import add_quantization
from agfb_noise.random_impulse import add_random_impulse
from agfb_noise.rayleigh import add_rayleigh
from agfb_noise.registry import (
    NoiseRegistration,
    get_noise_registration,
    register_noise,
    registered_noises,
)
from agfb_noise.rician import add_rician
from agfb_noise.runtime.runner import NoiseSpec, add_noise, apply_noise_sequence
from agfb_noise.salt import add_salt
from agfb_noise.salt_pepper import add_salt_pepper
from agfb_noise.speckle import add_speckle
from agfb_noise.stripe import add_stripe
from agfb_noise.uniform import add_uniform

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
