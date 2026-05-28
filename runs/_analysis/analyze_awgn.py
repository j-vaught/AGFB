"""AWGN SNR-ladder scaling of the optimal CPGF support.

The awgn_robustness study sweeps a single noise model, additive white Gaussian,
across twelve SNR levels in dB, each repeated over eight seeds, on the full 559
unique-cell catalog at 4096^2. Holding the polynomial degree at one, this isolates how
the NRMSE-optimal support radius scales with noise: the optimum walks from the
smallest competitive radius at high SNR to the widest at low SNR, the empirical
counterpart of the optimal-radius law shown analytically in Section 5.

Reductions follow the AGFB protocol: per-image NRMSE is averaged over the 559
unique cells within each seed, and the eight seed means feed a Student-t 95%
confidence interval per (radius, SNR). Emits the CeTZ CSV (radius, snr_db, mean,
lo, hi, is_optimal) for the NRMSE-vs-SNR crossover family.
"""

import glob
import re
from typing import cast

import polars as pl
from synthetic_dedup import deduplicate_synthetic_results

OUT_CSV = "../PGF_paper/figures/cetz_src/main/fig_sec06_awgn_snr.csv"
OUT_FAM_CSV = "../PGF_paper/figures/cetz_src/main/fig_sec06_awgn_family.csv"
T_975_7 = 2.364624251  # Student-t, 0.975 quantile, 7 dof (8 seeds)

# Coarse family fold for the cross-filter comparison: classical stencils are
# grouped by operator rather than by stencil size.
_FOLD = {
    "sobel_3": "Sobel",
    "sobel_5": "Sobel",
    "sobel_7": "Sobel",
    "scharr_3": "Scharr",
    "central_difference": "Central diff.",
    "sparse_central_difference": "Central diff.",
    "derivative_of_gaussian": "DoG",
    "deriche_recursive_gaussian_derivative": "Deriche",
    "freeman_adelson_g1": "Freeman-Adelson",
    "savitzky_golay": "Savitzky-Golay",
    "cpgf": "CPGF",
}


def rd(s):
    m = re.search(r"radius(\d+)_degree(\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


fs = sorted(glob.glob("runs/synthetic/awgn_robustness/*.parquet"))
df = deduplicate_synthetic_results(pl.concat([pl.read_parquet(f) for f in fs], how="diagonal"))
print(f"shards={len(fs)} seeds={sorted(df['seed'].unique().to_list())}")

nr = df.filter(
    (pl.col("metric") == "nrmse") & (~pl.col("is_nan")) & (pl.col("filter_family") == "cpgf")
)

# per-seed cell-mean, then mean and 95% CI across the eight seeds
per_seed = nr.group_by("filter_config_id", "snr_db", "seed").agg(
    pl.col("value").mean().alias("seed_mean")
)
stats = per_seed.group_by("filter_config_id", "snr_db").agg(
    pl.col("seed_mean").mean().alias("nrmse"),
    pl.col("seed_mean").std(ddof=1).alias("sd"),
    pl.len().alias("n_seeds"),
)
stats = stats.with_columns(
    (T_975_7 * pl.col("sd") / pl.col("n_seeds").sqrt()).alias("half")
).with_columns(
    (pl.col("nrmse") - pl.col("half")).alias("lo"),
    (pl.col("nrmse") + pl.col("half")).alias("hi"),
    pl.col("filter_config_id").map_elements(lambda s: rd(s)[0], return_dtype=pl.Int64).alias("r"),
    pl.col("filter_config_id").map_elements(lambda s: rd(s)[1], return_dtype=pl.Int64).alias("d"),
)
g = stats.filter(pl.col("d") == 1)

# mark the per-SNR optimum (lowest mean NRMSE) so the figure can trace the envelope
opt = g.sort("nrmse").group_by("snr_db").agg(pl.col("r").first().alias("r_opt"))
g = g.join(opt, on="snr_db").with_columns(
    (pl.col("r") == pl.col("r_opt")).cast(pl.Int8).alias("is_optimal")
)

out = g.select("r", "snr_db", "nrmse", "lo", "hi", "is_optimal").sort("r", "snr_db")
out.write_csv(OUT_CSV)
print(f"wrote {OUT_CSV} ({out.height} rows)")

# ===== cross-filter comparison: best NRMSE per family vs SNR =================
# For every (folded family, SNR) the lowest-NRMSE configuration is selected, and
# its mean and 95% across-seed CI are reported, giving each family's achievable
# lower envelope across the SNR ladder.
allf = df.filter((pl.col("metric") == "nrmse") & (~pl.col("is_nan")))
per_seed_all = allf.group_by("filter_config_id", "filter_family", "snr_db", "seed").agg(
    pl.col("value").mean().alias("seed_mean")
)
fam_stats = (
    per_seed_all.group_by("filter_config_id", "filter_family", "snr_db")
    .agg(
        pl.col("seed_mean").mean().alias("nrmse"),
        pl.col("seed_mean").std(ddof=1).alias("sd"),
        pl.len().alias("n_seeds"),
    )
    .with_columns(
        pl.col("filter_family").replace(_FOLD).alias("fam"),
        (T_975_7 * pl.col("sd") / pl.col("n_seeds").sqrt()).alias("half"),
    )
)
# pick the best config per (fam, snr)
fam_best = (
    fam_stats.sort("nrmse")
    .group_by("fam", "snr_db")
    .agg(
        pl.col("nrmse").first(),
        pl.col("half").first(),
        pl.col("filter_config_id").first().alias("best_config"),
    )
    .with_columns(
        (pl.col("nrmse") - pl.col("half")).alias("lo"),
        (pl.col("nrmse") + pl.col("half")).alias("hi"),
    )
)
FAMILIES = ["CPGF", "DoG", "Freeman-Adelson", "Savitzky-Golay", "Deriche", "Sobel", "Scharr"]
fam_out = (
    fam_best.filter(pl.col("fam").is_in(FAMILIES))
    .select("fam", "snr_db", "nrmse", "lo", "hi", "best_config")
    .sort("fam", "snr_db")
)
fam_out.write_csv(OUT_FAM_CSV)
print(f"wrote {OUT_FAM_CSV} ({fam_out.height} rows)")

radii = sorted(g["r"].unique().to_list())
snrs = sorted(g["snr_db"].unique().to_list())
print("\nNRMSE-optimal radius vs SNR (degree 1):")
for s in snrs:
    row = g.filter(pl.col("snr_db") == s).sort("nrmse").row(0, named=True)
    print(f"  SNR {s:5.1f} dB -> r={row['r']:3d}  NRMSE={row['nrmse']:.3f}  +/-{row['half']:.3f}")

print("\ntypical 95% CI half-width (fraction of mean), competitive radii r>=11:")
comp = g.filter(pl.col("r") >= 11).with_columns((pl.col("half") / pl.col("nrmse")).alias("frac"))
median_frac = comp["frac"].median()
max_frac = comp["frac"].max()
median_pct = (
    float(cast(int | float, median_frac)) * 100 if median_frac is not None else float("nan")
)
max_pct = float(cast(int | float, max_frac)) * 100 if max_frac is not None else float("nan")
print(f"  median {median_pct:.2f}%  max {max_pct:.2f}%")
