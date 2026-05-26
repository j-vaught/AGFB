"""Compare native vs 8x-supersampled BSDS500 edge detection.

The supersampling pipeline upscales each image 8x with bilinear interpolation,
applies a 9-tap Gaussian (sigma 1.5) anti-alias prefilter, runs each gradient
filter on the high-resolution field, area-averages the gradient components back
to native resolution, and recombines the magnitude. This isolates how much of
each filter's edge score is limited by aliasing at the native sampling grid.
Both runs are scored identically against the BSDS500 region-boundary ground
truth in raw mode. Emits the CeTZ scatter CSV (native ODS vs supersampled ODS).
"""

import glob

import polars as pl

OUT_CSV = "../PGF_paper/figures/cetz_src/main/fig_sec07_ss8_ods.csv"


def best_raw_ods(study_glob):
    fs = sorted(glob.glob(study_glob))
    df = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal")
    df = df.filter(
        (pl.col("dataset") == "bsds500")
        & (pl.col("mode") == "raw")
        & (pl.col("metric") == "ods")
    )
    return df.group_by("filter_config_id", "filter_family").agg(
        pl.col("value").mean().alias("ods")
    )


native = best_raw_ods("runs/realimg/edges/*.parquet").rename({"ods": "ods_native"})
ss8 = best_raw_ods("runs/realimg/supersampled/*.parquet").rename({"ods": "ods_ss8"})

j = native.join(ss8.select("filter_config_id", "ods_ss8"), on="filter_config_id")
j = j.with_columns(
    (pl.col("filter_family") == "cpgf").cast(pl.Int8).alias("is_cpgf"),
    (pl.col("ods_ss8") - pl.col("ods_native")).alias("delta"),
).sort("ods_native", descending=True)

print(f"filters compared: {j.height}")
print(f"improved by supersampling: {j.filter(pl.col('delta') > 0).height}")
print(f"mean delta: {j['delta'].mean():.4f}  median: {j['delta'].median():.4f}")
print(f"max gain: {j['delta'].max():.4f}  max loss: {j['delta'].min():.4f}")
print("\nCPGF rows:")
for r in j.filter(pl.col("is_cpgf") == 1).sort("ods_ss8", descending=True).head(8).iter_rows(named=True):
    print(f"  {r['filter_config_id']:30s} native={r['ods_native']:.4f} ss8={r['ods_ss8']:.4f} d={r['delta']:+.4f}")
print("\nbiggest gains:")
for r in j.sort("delta", descending=True).head(8).iter_rows(named=True):
    print(f"  {r['filter_config_id']:30s} [{r['filter_family']}] native={r['ods_native']:.4f} ss8={r['ods_ss8']:.4f} d={r['delta']:+.4f}")
print("\nbest ss8 overall:")
for r in j.sort("ods_ss8", descending=True).head(6).iter_rows(named=True):
    print(f"  {r['filter_config_id']:30s} [{r['filter_family']}] native={r['ods_native']:.4f} ss8={r['ods_ss8']:.4f}")

j.select("filter_config_id", "filter_family", "is_cpgf", "ods_native", "ods_ss8", "delta").write_csv(OUT_CSV)
print(f"\nwrote {OUT_CSV}")
