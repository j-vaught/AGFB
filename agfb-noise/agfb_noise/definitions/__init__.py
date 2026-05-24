"""Noise model definitions."""

from agfb_noise.definitions.coherent_medical import (
    add_k_speckle,
    add_log_speckle,
    add_lognormal_scintillation,
    add_nakagami_speckle,
    add_noncentral_chi,
    add_oct_speckle,
    add_sar_multilook,
)
from agfb_noise.definitions.compression_digitization import (
    add_aliasing,
    add_block_dropout,
    add_dither,
    add_gradient_banding,
    add_jpeg,
    add_mosquito_noise,
    add_overshoot,
    add_posterization,
    add_wavelet_ringing,
)
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
from agfb_noise.definitions.sensor_physics import (
    add_adc_nonlinearity,
    add_amp_glow,
    add_blooming_smear,
    add_dsnu,
    add_hot_pixel_clusters,
    add_ktc_reset,
    add_photon_transfer_chain,
    add_prnu,
    add_rolling_shutter,
    add_row_correlated_read_noise,
    add_rts_noise,
    add_saturation_clip,
    add_vignetting,
)
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
    "add_nakagami_speckle",
    "add_k_speckle",
    "add_oct_speckle",
    "add_sar_multilook",
    "add_noncentral_chi",
    "add_log_speckle",
    "add_lognormal_scintillation",
    "add_prnu",
    "add_row_correlated_read_noise",
    "add_rts_noise",
    "add_saturation_clip",
    "add_photon_transfer_chain",
    "add_dsnu",
    "add_ktc_reset",
    "add_amp_glow",
    "add_blooming_smear",
    "add_rolling_shutter",
    "add_adc_nonlinearity",
    "add_hot_pixel_clusters",
    "add_vignetting",
    "add_jpeg",
    "add_gradient_banding",
    "add_overshoot",
    "add_block_dropout",
    "add_aliasing",
    "add_mosquito_noise",
    "add_wavelet_ringing",
    "add_posterization",
    "add_dither",
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
