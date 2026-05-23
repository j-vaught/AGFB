"""Compatibility imports for shared tensor helpers."""

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

__all__ = [
    "ClampRange",
    "Numeric",
    "apply_clamp",
    "batch_param",
    "check_image",
    "image_param",
    "rand_like",
    "rand_shape",
    "randn_like",
    "randn_shape",
    "resolve_generator",
    "validate_nonnegative",
    "validate_positive",
    "validate_probability",
]
