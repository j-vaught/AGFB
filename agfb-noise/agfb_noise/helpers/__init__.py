"""Shared helpers for noise models and dispatch."""

from agfb_noise.helpers.base import (
    ClampRange,
    Numeric,
    apply_clamp,
    batch_param,
    check_image,
    image_param,
    rand_like,
    rand_shape,
    randn_like,
    randn_shape,
    resolve_generator,
    validate_nonnegative,
    validate_positive,
    validate_probability,
)
from agfb_noise.helpers.catalog import BuiltInNoiseSpec, shipped_noise_specs
from agfb_noise.helpers.registry import (
    NoiseFunction,
    NoiseRegistration,
    get_noise_registration,
    register_noise,
    registered_noises,
)
from agfb_noise.helpers.runner import NoiseSpec, add_noise, apply_noise_sequence

__all__ = [
    "BuiltInNoiseSpec",
    "ClampRange",
    "NoiseFunction",
    "NoiseRegistration",
    "NoiseSpec",
    "Numeric",
    "add_noise",
    "apply_clamp",
    "apply_noise_sequence",
    "batch_param",
    "check_image",
    "get_noise_registration",
    "image_param",
    "rand_like",
    "rand_shape",
    "randn_like",
    "randn_shape",
    "register_noise",
    "registered_noises",
    "resolve_generator",
    "shipped_noise_specs",
    "validate_nonnegative",
    "validate_positive",
    "validate_probability",
]
