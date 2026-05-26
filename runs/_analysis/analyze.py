"""Compact stats for the results section. Reductions match the AGFB protocol:
per-image metric values averaged over cells/seeds; noise_gain uses the median
(outlier-robust, as in the dashboard)."""

import glob

import polars as pl

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_width_chars(200)


def load(subdir):
    fs = sorted(glob.glob(f"runs/{subdir}/*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in fs], how="diagonal")


def agg_filter(df, metric, fn="mean"):
    d = df.filter((pl.col("metric") == metric) & (~pl.col("is_nan")))
    expr = pl.col("value").median() if fn == "median" else pl.col("value").mean()
    return d.group_by("filter_config_id", "filter_family").agg(
        expr.alias(metric), pl.len().alias("n")
    )


def show(title, df, sort_col, descending=False, k=12):
    print(f"\n### {title}")
    out = df.sort(sort_col, descending=descending).head(k)
    for r in out.iter_rows(named=True):
        cols = " ".join(
            f"{c}={r[c]:.4g}" if isinstance(r[c], float) else f"{c}={r[c]}"
            for c in out.columns
            if c not in ("filter_family", "n")
        )
        print(f"  {r['filter_config_id']:42s} [{r['filter_family']}] {cols}")


# ===== A: clean headline =====
A = load("synthetic/clean_accuracy")
print("=" * 70, "\nSTUDY A - CLEAN (SNR=inf), 110 filters, 4096^2, all classes")
print("cells per filter:", A.filter(pl.col("metric") == "nrmse").height // 110)
nr = agg_filter(A, "nrmse")
am = agg_filter(A, "angular_mae")
mb = agg_filter(A, "magnitude_bias")
clean = nr.join(am, on=["filter_config_id", "filter_family"]).join(
    mb.select("filter_config_id", "magnitude_bias"), on="filter_config_id"
)
show("best NRMSE (clean)", clean, "nrmse")
show("best angular_mae (clean)", clean, "angular_mae")
print("\nCPGF on clean (by nrmse):")
show("  cpgf only", clean.filter(pl.col("filter_family") == "cpgf"), "nrmse", k=8)
# rank of best cpgf
cl_sorted = clean.sort("nrmse").with_row_index("rank")
cp = cl_sorted.filter(pl.col("filter_family") == "cpgf").head(1)
print("best CPGF nrmse rank:", cp.select("rank", "filter_config_id", "nrmse").to_dicts())
# named baselines
for fid in [
    "sobel_3",
    "sobel_5",
    "sobel_7",
    "scharr_3",
    "central_difference",
    "derivative_of_gaussian_sigma2",
    "deriche_recursive_gaussian_derivative_sigma4",
]:
    row = cl_sorted.filter(pl.col("filter_config_id") == fid)
    if row.height:
        r = row.row(0, named=True)
        print(f"  {fid:46s} rank={r['rank']:3d} nrmse={r['nrmse']:.4g} angmae={r['angular_mae']:.4g}")

# ===== C: mixed-catalog noise robustness =====
C = load("synthetic/noise_breadth")
print("\n" + "=" * 70, "\nSTUDY C - categorical noise, 29 filters, 8 seeds")
print("noise models:", C["noise_model"].unique().to_list())
print("conditions:", C["noise_condition_id"].n_unique())
cn = agg_filter(C, "nrmse").join(agg_filter(C, "angular_mae"), on=["filter_config_id", "filter_family"])
cn = cn.join(agg_filter(C, "noise_gain", "median").select("filter_config_id", "noise_gain"), on="filter_config_id")
show("best NRMSE under noise (C, all models pooled)", cn, "nrmse")
cn_sorted = cn.sort("nrmse").with_row_index("rank")
cp2 = cn_sorted.filter(pl.col("filter_family") == "cpgf").head(1)
print("best CPGF rank under noise:", cp2.select("rank", "filter_config_id", "nrmse", "noise_gain").to_dicts())

# ===== CG: CPGF radius/degree grid under noise =====
CG = load("synthetic/cpgf_grid")
print("\n" + "=" * 70, "\nSTUDY CG - CPGF grid under noise, 51 filters, 8 seeds")
print("families:", CG["filter_family"].unique().to_list())
print("noise models:", CG["noise_model"].unique().to_list())
cg = agg_filter(CG, "nrmse").join(
    agg_filter(CG, "noise_gain", "median").select("filter_config_id", "noise_gain"), on="filter_config_id"
)
cg = cg.join(agg_filter(CG, "angular_mae").select("filter_config_id", "angular_mae"), on="filter_config_id")
show("best NRMSE (CG, CPGF grid under noise)", cg, "nrmse", k=10)
# radius trend: parse r and d from config id like cpgf_radius15_degree1
import re

def rd(s):
    m = re.search(r"radius(\d+)_degree(\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

cg2 = cg.with_columns(
    pl.col("filter_config_id").map_elements(lambda s: rd(s)[0], return_dtype=pl.Int64).alias("r"),
    pl.col("filter_config_id").map_elements(lambda s: rd(s)[1], return_dtype=pl.Int64).alias("d"),
)
print("\nnoise_gain (median) vs radius, degree=1:")
print(cg2.filter(pl.col("d") == 1).sort("r").select("r", "noise_gain", "nrmse").to_dicts())

# ===== D: wall-clock =====
D = load("timing/walltime_scaling")
print("\n" + "=" * 70, "\nSTUDY D - timing 4096^2 (ms/call)")
D = D.with_columns((pl.col("ms_per_call")).alias("ms"))
for fid in [
    "central_difference",
    "sobel_3",
    "sobel_7",
    "derivative_of_gaussian_sigma2",
    "deriche_recursive_gaussian_derivative_sigma4",
]:
    row = D.filter(pl.col("filter_config_id") == fid)
    if row.height:
        r = row.row(0, named=True)
        print(f"  {fid:46s} {r['filter_path']:18s} {r['ms']:.3f} ms")
print("\nCPGF timing by path/radius:")
cpd = D.filter(pl.col("filter_family") == "cpgf").with_columns(
    pl.col("filter_config_id").map_elements(lambda s: rd(s)[0], return_dtype=pl.Int64).alias("r")
).sort("r")
for r in cpd.iter_rows(named=True):
    print(f"  {r['filter_config_id']:30s} r={r['r']} path={r['filter_path']:12s} {r['ms_per_call']:.3f} ms")
