"""Noise system.

The robustness axis is additive white Gaussian over a dB grid; the
remaining 21 ``agfb-noise`` models are characterized on the canonical subset
with native-unit mild->severe ladders. ``add_gaussian`` realizes a target
SNR via ``sigma_n = contrast / 10**(snr_db/20)``, so a given dB is the same
effective SNR regardless of field contrast and the injected ``sigma_n`` (which
the ``noise_gain`` metric needs) is known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from agfb_bench.config import SNR_DB_GRID, awgn_sigma, noise_seed


@dataclass
class NoiseCondition:
    """One noise condition (a column of the corruption grid)."""

    condition_id: str
    model: str  # "add_gaussian", "add_uniform", ... ("clean" for the inf-dB pass)
    kind: str  # "clean" | "awgn" | "native"
    level_index: int
    params: dict[str, Any] = field(default_factory=dict)
    snr_db: float | None = None
    deterministic: bool = False
    # How noise_gain's sigma_n is obtained: "exact" (AWGN), "measured" (native),
    # "quantization" (theoretical step error), or "none" (clean pass).
    sigma_mode: str = "exact"


# -- 2.1 Additive white Gaussian - the robustness axis ------------------------
def awgn_conditions() -> list[NoiseCondition]:
    """13-level dB grid; level 0 is the clean field (no injection)."""
    conditions: list[NoiseCondition] = []
    for index, snr_db in enumerate(SNR_DB_GRID):
        if snr_db == float("inf"):
            conditions.append(
                NoiseCondition("clean", "clean", "clean", index, snr_db=snr_db, sigma_mode="none")
            )
        else:
            conditions.append(
                NoiseCondition(
                    f"awgn_{snr_db:g}dB",
                    "add_gaussian",
                    "awgn",
                    index,
                    snr_db=snr_db,
                    sigma_mode="exact",
                )
            )
    return conditions


def noisy_awgn_conditions() -> list[NoiseCondition]:
    """The 12 finite-dB AWGN conditions (excludes the clean pass)."""
    return [c for c in awgn_conditions() if c.kind == "awgn"]


# -- 2.2 Noise-type breadth - native-unit ladders -----------------------------
# (model, fixed kwargs, ladder-parameter name(s), ladder values, deterministic,
#  sigma_mode). Tuples of names map element-wise onto tuples of values.
_NATIVE_LADDERS: tuple[tuple, ...] = (
    ("add_uniform", {}, ("half_width",), (0.02, 0.05, 0.1, 0.2, 0.4), False, "measured"),
    ("add_local_variance", {}, ("variance",), (5e-4, 2e-3, 8e-3, 2e-2, 5e-2), False, "measured"),
    ("add_speckle", {}, ("sigma",), (0.05, 0.1, 0.2, 0.4, 0.8), False, "measured"),
    ("add_rayleigh", {}, ("sigma",), (0.02, 0.05, 0.1, 0.2), False, "measured"),
    ("add_poisson", {}, ("peak",), (1000, 200, 50, 20, 5), False, "measured"),
    (
        "add_poisson_gaussian",
        {},
        ("peak", "read_sigma"),
        ((400, 0.005), (100, 0.02), (25, 0.05)),
        False,
        "measured",
    ),
    (
        "add_dark_current",
        {"exposure_time": 1.0, "peak": 20.0, "read_sigma": 0.02},
        ("dark_rate",),
        (0.5, 1.0, 2.0, 4.0, 8.0),
        False,
        "measured",
    ),
    ("add_salt", {}, ("amount",), (0.005, 0.01, 0.03, 0.05, 0.1), False, "measured"),
    ("add_pepper", {}, ("amount",), (0.005, 0.01, 0.03, 0.05, 0.1), False, "measured"),
    (
        "add_salt_pepper",
        {"salt_vs_pepper": 0.5},
        ("amount",),
        (0.01, 0.03, 0.05, 0.1, 0.2),
        False,
        "measured",
    ),
    ("add_random_impulse", {}, ("amount",), (0.01, 0.03, 0.05, 0.1, 0.2), False, "measured"),
    (
        "add_dead_pixels",
        {"hot_fraction": 0.5},
        ("amount",),
        (0.005, 0.01, 0.02, 0.05, 0.1),
        False,
        "measured",
    ),
    ("add_gamma_speckle", {}, ("looks",), (64, 16, 4, 2, 1), False, "measured"),
    ("add_rician", {}, ("sigma",), (0.02, 0.05, 0.1, 0.2), False, "measured"),
    ("add_quantization", {}, ("levels",), (256, 64, 16, 8, 4), True, "quantization"),
    (
        "add_fixed_pattern",
        {},
        ("offset_sigma", "gain_sigma"),
        ((0.005, 0.005), (0.01, 0.01), (0.02, 0.02), (0.05, 0.05)),
        False,
        "measured",
    ),
    (
        "add_stripe",
        {},
        ("row_sigma", "column_sigma"),
        ((0.02, 0), (0, 0.02), (0.02, 0.02), (0.05, 0.05)),
        False,
        "measured",
    ),
    # Tier-1 spatially correlated fields. sigma is the true marginal std
    # (the library decouples variance from correlation length), so these
    # amplitude ladders pair against AWGN and add_speckle on equal footing.
    (
        "add_correlated_gaussian",
        {"correlation_length": 4.0},
        ("sigma",),
        (0.02, 0.05, 0.1, 0.2, 0.4),
        False,
        "measured",
    ),
    (
        "add_powerlaw_gaussian",
        {"sigma": 0.1},
        ("beta",),
        (0.5, 1.0, 1.5, 2.0, 3.0),
        False,
        "measured",
    ),
    (
        "add_anisotropic_gaussian",
        {"sigma": 0.1, "angle": 0.0},
        ("correlation_length_x", "correlation_length_y"),
        ((1.0, 2.0), (1.0, 4.0), (1.0, 8.0), (1.0, 16.0)),
        False,
        "measured",
    ),
    (
        "add_correlated_speckle",
        {"correlation_length": 4.0},
        ("sigma",),
        (0.05, 0.1, 0.2, 0.4, 0.8),
        False,
        "measured",
    ),
)


def native_conditions() -> list[NoiseCondition]:
    """98 non-AWGN conditions across the 21 native-unit models."""
    conditions: list[NoiseCondition] = []
    for model, fixed, names, values, deterministic, sigma_mode in _NATIVE_LADDERS:
        for level_index, value in enumerate(values):
            tuple_value = value if isinstance(value, tuple) else (value,)
            params = dict(fixed)
            params.update(dict(zip(names, tuple_value, strict=True)))
            label = "_".join(f"{v:g}" for v in tuple_value)
            conditions.append(
                NoiseCondition(
                    f"{model}__{label}",
                    model,
                    "native",
                    level_index,
                    params=params,
                    deterministic=deterministic,
                    sigma_mode=sigma_mode,
                )
            )
    return conditions


def apply_noise(
    condition: NoiseCondition,
    clean: torch.Tensor,
    *,
    contrast: float,
    cell_seed: int,
    flat_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Inject ``condition`` into a clean field. Returns ``(noisy, sigma_n)``.

    ``sigma_n`` is the value passed to ``noise_gain``: exact for AWGN, the
    measured flat-region std of ``(noisy - clean)`` for native models, and the
    theoretical step error ``q/sqrt(12)`` for quantization. The clean pass
    returns ``(clean, 0.0)``.
    """
    import agfb_noise as noise

    if condition.kind == "clean":
        return clean, 0.0

    seed = None if condition.deterministic else noise_seed(cell_seed, condition.level_index)

    if condition.kind == "awgn":
        sigma_n = awgn_sigma(contrast, condition.snr_db)
        noisy = noise.add_gaussian(clean, sigma=sigma_n, seed=seed, clamp=(0.0, 1.0))
        return noisy, sigma_n

    add_fn = getattr(noise, condition.model)
    kwargs = dict(condition.params)
    if not condition.deterministic:
        kwargs["seed"] = seed
    noisy = add_fn(clean, clamp=(0.0, 1.0), **kwargs)

    if condition.sigma_mode == "quantization":
        levels = int(condition.params["levels"])
        step = 1.0 / max(levels - 1, 1)
        sigma_n = step / (12.0**0.5)
    else:
        sigma_n = _measured_sigma(noisy - clean, flat_mask)
    return noisy, sigma_n


def _measured_sigma(residual: torch.Tensor, flat_mask: torch.Tensor | None) -> float:
    """Std of the noise residual over the flat region (fallback: whole frame)."""
    if flat_mask is not None and bool(flat_mask.any()):
        # residual and flat_mask are both (1, H, W); flatten before masking so
        # the comparison is shape-agnostic.
        values = residual.reshape(-1)[flat_mask.reshape(-1).bool()]
    else:
        values = residual
    sigma = float(values.float().std().item())
    return sigma if sigma > 0 else 1e-12
