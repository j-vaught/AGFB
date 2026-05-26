"""Locked constants from BENCHMARK_DESIGN.md (the single source of truth)."""

from __future__ import annotations

import math

# -- Conventions locked for every run (spec, "Conventions" table) -------------
PRODUCTION_IMAGE_SIZE = 4096
DTYPE = "float32"
INTENSITY_CLAMP = (0.0, 1.0)

PRODUCTION_SEEDS = tuple(range(8))  # 0..7
EXTENDED_SEEDS = tuple(range(16))  # 0..15 (headline + tail metrics)
VALIDATION_SEEDS = tuple(range(100, 108))  # 100..107 (parameter selection)

# -- AWGN robustness axis (spec 2.1) ------------------------------------------
# inf is the clean field (sigma_n = 0, no injection).
SNR_DB_GRID = (
    math.inf,
    30.0,
    25.0,
    20.0,
    15.0,
    12.0,
    10.0,
    7.5,
    5.0,
    2.5,
    1.0,
    0.5,
    0.0,
)
NOISY_SNR_DB = tuple(db for db in SNR_DB_GRID if math.isfinite(db))  # 12 levels
TUNING_SNR_DB = 10.0  # parameter selection operating point (spec 5.1)

# -- Metric sets (spec Chapter 4) ---------------------------------------------
PIXEL_METRICS = (
    "nrmse",
    "angular_mae",
    "tail_vector_error",
    "tangential_normal_leak",
    "magnitude_bias",
    "noise_gain",
    "tail_spurious_grad",
)
PROFILE_METRICS = (
    "localization_offset",
    "edge_fwhm",
    "sidelobe_ratio",
)
ALL_METRICS = (
    "nrmse",
    "angular_mae",
    "tail_vector_error",
    "localization_offset",
    "tangential_normal_leak",
    "magnitude_bias",
    "edge_fwhm",
    "sidelobe_ratio",
    "noise_gain",
    "tail_spurious_grad",
)

# Profile-metric sampling parameters (spec Chapter 4 table).
PROFILE_R_MAX = 16.0
PROFILE_STEP = 0.5
# Mask parameters (spec 4.1) at production size; dilate_px scales with image.
MASK_REL_EPS = 1e-6
MASK_DILATE_PX_AT_4096 = 8


def mask_dilate_px(image_size: int) -> int:
    """Dilation radius for masks, scaled proportionally from the 4096 setting."""
    scaled = round(MASK_DILATE_PX_AT_4096 * image_size / PRODUCTION_IMAGE_SIZE)
    return max(1, scaled)


def awgn_sigma(contrast: float, snr_db: float) -> float:
    """sigma_n = contrast / 10**(snr_db/20); inf dB -> 0 (clean)."""
    if math.isinf(snr_db):
        return 0.0
    return float(contrast) / (10.0 ** (snr_db / 20.0))


def noise_seed(cell_seed: int, level_index: int) -> int:
    """Seed convention shared by every noise model (spec 2.x)."""
    return cell_seed * 10_000 + level_index
