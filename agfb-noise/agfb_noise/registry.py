"""Compatibility imports for noise-model registration."""

from agfb_noise.helpers.registry import (
    NoiseFunction,
    NoiseRegistration,
    get_noise_registration,
    register_noise,
    registered_noises,
)

__all__ = [
    "NoiseFunction",
    "NoiseRegistration",
    "get_noise_registration",
    "register_noise",
    "registered_noises",
]
