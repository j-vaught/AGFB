"""Filter grid — Chapter 3 of BENCHMARK_DESIGN.md.

A :class:`FilterConfig` is one ``agfb-filters`` family with one parameter
dictionary on one :class:`ExecutionPath`. CPGF and square Savitzky-Golay are
swept densely (the support-size methods the paper is about); the rest are
baselines. Underdetermined polynomial cells (a 2-D degree ``d`` needs
``(d+1)(d+2)/2`` samples <= support) are skipped at construction: the library
raises ``ValueError`` for them, which :func:`build_filter_configs` catches.

Orientation-bank filters (the ``ORIENTATION_BANK`` path) are excluded (spec 3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilterConfig:
    """One filter family + parameters + execution path."""

    family: str
    params: dict[str, Any]
    path: str
    config_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.config_id:
            parts = [self.family]
            for key, value in self.params.items():
                parts.append(
                    f"{key}{value:g}" if isinstance(value, (int, float)) else f"{key}{value}"
                )
            self.config_id = "_".join(parts)


# -- 3.1 Fixed baselines (no free parameters) ---------------------------------
# roberts has no separable form; it runs on STENCIL. The rest are separable.
# ando_4 is omitted: its even support has no separable path and lands on the
# dual grid, so it is not a like-for-like centered baseline.
_FIXED_BASELINES = (
    ("central_difference", "SEPARABLE"),
    ("roberts", "STENCIL"),
    ("prewitt_3", "SEPARABLE"),
    ("scharr_3", "SEPARABLE"),
    ("sobel_3", "SEPARABLE"),
    ("sobel_5", "SEPARABLE"),
    ("sobel_7", "SEPARABLE"),
    ("ando_3", "SEPARABLE"),
    ("ando_5", "SEPARABLE"),
    ("farid_simoncelli_5", "SEPARABLE"),
    ("farid_simoncelli_7", "SEPARABLE"),
)


def _cpgf_path(radius: int) -> str:
    """CPGF path is selected by support size: sparse offsets <=8 px, FFT >=9 px."""
    return "SPARSE_OFFSETS" if radius <= 8 else "FFT"


def _try_build(family: str, path: str, **params) -> FilterConfig | None:
    """Return a config, or ``None`` if the library rejects it as underdetermined."""
    import agfb_filters as filters

    try:
        filters.get_filter_definition(family, **params)
    except ValueError as error:
        # A 2-D degree d needs (d+1)(d+2)/2 samples <= support; the library
        # rejects such cells as "underdetermined" or "rank deficient" (spec 3).
        message = str(error).lower()
        if "underdetermined" in message or "rank deficient" in message:
            return None
        raise
    return FilterConfig(family, params, path)


def build_filter_configs(profile: str = "full") -> list[FilterConfig]:
    """Build the filter configs for a profile.

    Profiles: ``headline`` | ``core`` | ``full`` | ``cpgf_grid``. The
    ``cpgf_grid`` profile is the CPGF radius x degree matrix only (no
    baselines), for isolating the operator's own tuning surface across noise.
    """
    if profile not in ("headline", "core", "full", "cpgf_grid"):
        raise ValueError(f"unknown filter profile {profile!r}")

    configs: list[FilterConfig] = []

    def keep(config: FilterConfig | None) -> None:
        if config is not None:
            configs.append(config)

    if profile == "cpgf_grid":
        # CPGF only, swept across degrees 1/3/5/7 so degree x radius x noise is
        # fully crossed. The radius ladder extends well past the ``core`` study's
        # top of 45 px (up to 255 px, a 511 px window on the 4096 px field) to
        # show where added support stops helping or boundary effects take over.
        # Underdetermined (degree too high for support) combos drop.
        for radius in (3, 5, 7, 11, 15, 21, 31, 45, 63, 91, 127, 181, 255):
            for degree in (1, 3, 5, 7):
                keep(_try_build("cpgf", _cpgf_path(radius), radius=radius, degree=degree))
        return configs

    if profile == "full":
        for family, path in _FIXED_BASELINES:
            keep(FilterConfig(family, {}, path))
        for radius in (3, 5, 7, 11, 15, 21, 31, 45):
            for degree in (1, 3, 5, 7):
                keep(_try_build("cpgf", _cpgf_path(radius), radius=radius, degree=degree))
        for window in (3, 5, 7, 11, 15, 21, 31):
            radius = (window - 1) // 2
            for degree in (1, 3, 5):
                keep(_try_build("savitzky_golay", "SPATIAL_DENSE", radius=radius, degree=degree))
        for sigma in (0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 9, 12):
            keep(FilterConfig("derivative_of_gaussian", {"sigma": float(sigma)}, "SEPARABLE"))
        for sigma in (1, 1.5, 2, 3, 4, 6, 8):
            keep(FilterConfig("freeman_adelson_g1", {"sigma": float(sigma)}, "SEPARABLE"))
        # 3.3 other baselines
        for sigma in (1, 2, 4, 8):
            keep(
                FilterConfig(
                    "deriche_recursive_gaussian_derivative", {"sigma": float(sigma)}, "RECURSIVE"
                )
            )
        for radius in (1, 2, 4, 8, 16):
            keep(FilterConfig("haar_box_gradient", {"radius": radius}, "BOX_INTEGRAL"))
        for radius in (1, 2, 3):
            keep(FilterConfig("sparse_central_difference", {"radius": radius}, "SPARSE_OFFSETS"))
        for radius in (1, 2, 3):
            for weighting in ("none", "huber", "bilateral", "tukey"):
                keep(
                    FilterConfig(
                        "robust_local_plane_gradient",
                        {"radius": radius, "weighting": weighting},
                        "NONLINEAR_WINDOW",
                    )
                )
        for iterations in (5, 10, 20):
            for kappa in (0.05, 0.2):
                keep(
                    FilterConfig(
                        "perona_malik_gradient",
                        {"iterations": iterations, "kappa": kappa},
                        "ITERATIVE",
                    )
                )
        keep(FilterConfig("riesz_transform", {"epsilon": 1e-12}, "FFT"))
        return configs

    # -- headline (~15): 11 fixed baselines + 4 representative tuned configs ---
    for family, path in _FIXED_BASELINES:
        keep(FilterConfig(family, {}, path))
    keep(_try_build("cpgf", _cpgf_path(7), radius=7, degree=3))
    keep(FilterConfig("derivative_of_gaussian", {"sigma": 2.0}, "SEPARABLE"))
    keep(_try_build("savitzky_golay", "SPATIAL_DENSE", radius=3, degree=3))  # square SG h7,d3
    keep(FilterConfig("freeman_adelson_g1", {"sigma": 2.0}, "SEPARABLE"))
    if profile == "headline":
        return configs

    # -- core (~26): headline + tuned sub-ladders -----------------------------
    for radius in (3, 5, 7, 11, 15, 21, 31, 45):
        if radius == 7:
            continue  # cpgf_r7_d3 already in headline
        keep(_try_build("cpgf", _cpgf_path(radius), radius=radius, degree=3))
    for sigma in (1, 2, 3, 4, 6):
        if sigma == 2:
            continue  # dog_sigma2 already in headline
        keep(FilterConfig("derivative_of_gaussian", {"sigma": float(sigma)}, "SEPARABLE"))
    for window in (5, 7, 11, 15):
        radius = (window - 1) // 2
        if window == 7:
            continue  # square_sg_h7_d3 already in headline
        keep(_try_build("savitzky_golay", "SPATIAL_DENSE", radius=radius, degree=3))
    return configs


def build_backend_sweep_grid() -> list[FilterConfig]:
    """Candidate filters for the backend-forcing timing sweep (Study E).

    The ``path`` on each config is only its *native* path, kept for reference;
    the sweep runner ignores it and forces every compatible
    :class:`ExecutionPath` in turn (discovered by try/except on ``run_filter``).
    The grid is deliberately denser than the ``full`` profile so that each
    family is sampled across the support sizes / parameters where a backend's
    cost crosses over (e.g. SPARSE_OFFSETS vs FFT for CPGF as radius grows).
    """
    configs: list[FilterConfig] = []

    def keep(config: FilterConfig | None) -> None:
        if config is not None:
            configs.append(config)

    # Fixed FIR baselines: each can be forced onto several dense/sparse paths.
    for family, path in _FIXED_BASELINES:
        keep(FilterConfig(family, {}, path))

    # CPGF: the paper's operator. Sweep radius across the sparse<->FFT crossover
    # (<=8 px sparse, >=9 px FFT natively) and a fuller degree ladder.
    for radius in (3, 5, 7, 9, 11, 15, 21, 31, 45, 63):
        for degree in (1, 2, 3, 4, 5, 6, 7):
            keep(_try_build("cpgf", _cpgf_path(radius), radius=radius, degree=degree))

    # Square Savitzky-Golay: dense polynomial windows.
    for window in (3, 5, 7, 9, 11, 15, 21, 31, 41):
        radius = (window - 1) // 2
        for degree in (1, 2, 3, 4, 5):
            keep(_try_build("savitzky_golay", "SPATIAL_DENSE", radius=radius, degree=degree))

    # Derivative-of-Gaussian and Freeman-Adelson G1: separable FIR families.
    for sigma in (0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 9, 12, 16):
        keep(FilterConfig("derivative_of_gaussian", {"sigma": float(sigma)}, "SEPARABLE"))
    for sigma in (1, 1.5, 2, 3, 4, 6, 8, 12):
        keep(FilterConfig("freeman_adelson_g1", {"sigma": float(sigma)}, "SEPARABLE"))

    # Specialized single-path families: included so the sweep records which
    # paths they reject and times them on the one path they do support.
    for sigma in (1, 2, 4, 8, 16):
        keep(
            FilterConfig(
                "deriche_recursive_gaussian_derivative", {"sigma": float(sigma)}, "RECURSIVE"
            )
        )
    for radius in (1, 2, 4, 8, 16, 32):
        keep(FilterConfig("haar_box_gradient", {"radius": radius}, "BOX_INTEGRAL"))
    for radius in (1, 2, 3, 4):
        keep(FilterConfig("sparse_central_difference", {"radius": radius}, "SPARSE_OFFSETS"))
    for radius in (1, 2, 3):
        for weighting in ("none", "huber", "bilateral", "tukey"):
            keep(
                FilterConfig(
                    "robust_local_plane_gradient",
                    {"radius": radius, "weighting": weighting},
                    "NONLINEAR_WINDOW",
                )
            )
    for iterations in (5, 10, 20, 40):
        for kappa in (0.05, 0.1, 0.2):
            keep(
                FilterConfig(
                    "perona_malik_gradient",
                    {"iterations": iterations, "kappa": kappa},
                    "ITERATIVE",
                )
            )
    keep(FilterConfig("riesz_transform", {"epsilon": 1e-12}, "FFT"))
    return configs
