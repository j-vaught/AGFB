"""Backend-crossover data for the CPGF execution-path timing figure.

The backend_timing study forces every (filter, execution path, image size) combination and times
it. For the CPGF the interesting axis is support radius. Spatial backends (dense
convolution, sparse offsets) cost O(r^2) per pixel, while the FFT backend is
independent of radius, so beyond a small crossover radius the FFT is the only
viable path. Timing is degree-independent (verified: the FFT path is 8.00-8.15ms
across all seven degrees at r15), so degree 1 is taken as representative. The
ANTIPODAL_PAIRS path is omitted: it degenerates at the largest radii (r63 returns
in 0.26ms for low degree) and is not a default backend. Emits the CeTZ CSV of
ms_per_call against radius at 4096^2 for the three well-behaved spatial/FFT
backends, plus an image-size scaling summary for the prose.
"""

import re

import polars as pl

OUT_CSV = "../PGF_paper/figures/cetz_src/main/fig_sec06_backend.csv"
PATHS = ["FFT", "SPATIAL_DENSE", "SPARSE_OFFSETS"]

d = pl.read_parquet("runs/timing/backend_timing/backend_timing_sweep.parquet")
cp = d.filter(
    (pl.col("filter_family") == "cpgf")
    & (pl.col("status") == "ok")
    & (pl.col("filter_config_id").str.contains("degree1"))
    & (pl.col("forced_path").is_in(PATHS))
).with_columns(
    pl.col("filter_config_id")
    .map_elements(lambda s: int(re.search(r"radius(\d+)", s).group(1)), return_dtype=pl.Int64)
    .alias("radius")
)

at4096 = (
    cp.filter(pl.col("image_size") == 4096)
    .select("radius", "forced_path", "ms_per_call")
    .sort("radius", "forced_path")
)
at4096.write_csv(OUT_CSV)
print(f"wrote {OUT_CSV} ({at4096.height} rows)")

print("\nms_per_call at 4096^2 (degree 1):")
wide = at4096.pivot(values="ms_per_call", index="radius", on="forced_path").sort("radius")
print(wide)

# crossover radius: smallest radius where FFT beats every spatial backend
print("\ncrossover (FFT vs fastest spatial):")
for r in sorted(at4096["radius"].unique().to_list()):
    row = at4096.filter(pl.col("radius") == r)
    fft = row.filter(pl.col("forced_path") == "FFT")["ms_per_call"][0]
    spat = row.filter(pl.col("forced_path") != "FFT")["ms_per_call"].min()
    flag = "FFT wins" if fft < spat else "spatial wins"
    print(f"  r={r:3d}  FFT={fft:7.2f}  best-spatial={spat:8.2f}  -> {flag}")

print("\nimage-size scaling at r15 (degree 1):")
sz = cp.filter(pl.col("radius") == 15).select("image_size", "forced_path", "ms_per_call")
for p in PATHS:
    vals = sz.filter(pl.col("forced_path") == p).sort("image_size")
    print(
        f"  {p:14s}",
        [(r["image_size"], round(r["ms_per_call"], 2)) for r in vals.iter_rows(named=True)],
    )
