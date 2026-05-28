"""AWGN SNR-ladder appendix tables.

Emits display-ready CSVs (numbers pre-formatted as strings) for the additive
white Gaussian noise appendix, the detailed companion to the SNR-scaling
subsection of Section 6. Reductions follow the AGFB protocol used in
analyze_appendix.py: per-image NRMSE and angular MAE are averaged over cells and
seeds, while noise gain uses the outlier-robust median. The confidence table
instead computes the per-seed cell-mean and a Student-t 95% interval across the
eight seeds, to quantify how tight the radius ordering is. Output lands in
../PGF_paper/figures/tables relative to the AGFB working directory.
"""

import glob
import re
from collections import Counter
from pathlib import Path
from typing import cast

import polars as pl
from synthetic_dedup import deduplicate_synthetic_results

OUT = Path("../PGF_paper/figures/tables")
OUT.mkdir(parents=True, exist_ok=True)
T_975_7 = 2.364624251  # Student-t, 0.975 quantile, 7 dof (8 seeds)

_RD = re.compile(r"radius(\d+)_degree(\d+)")
_SUB = re.compile(r"sigma(\d+(?:\.\d+)?)")

# Coarse family fold for the cross-filter matrices, matching the companion
# figure: classical stencils grouped by operator rather than by stencil size.
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
# Families shown in the cross-filter matrices, in display order.
FAMILIES = ["CPGF", "DoG", "Freeman-Adelson", "Savitzky-Golay", "Deriche", "Sobel", "Scharr"]
NumericCell = int | float | None
NumericRows = list[tuple[str, list[NumericCell]]]

# Full filter-type grouping for the per-family catalog (folds stencil sizes).
_FAMILY_RULES = (
    ("cpgf", "CPGF"),
    ("deriche_recursive_gaussian_derivative", "Deriche"),
    ("derivative_of_gaussian", "DoG"),
    ("freeman_adelson", "Freeman-Adelson"),
    ("robust_local_plane", "Robust local-plane"),
    ("perona_malik", "Perona-Malik"),
    ("savitzky_golay", "Savitzky-Golay"),
    ("sparse_central_difference", "Sparse central diff."),
    ("haar_box", "Haar box"),
    ("farid_simoncelli", "Farid-Simoncelli"),
    ("riesz_transform", "Riesz transform"),
    ("central_difference", "Central difference"),
    ("scharr", "Scharr"),
    ("sobel", "Sobel"),
    ("prewitt", "Prewitt"),
    ("ando", "Ando"),
    ("roberts", "Roberts"),
)


def family(cid: str) -> str:
    for k, v in _FAMILY_RULES:
        if cid.startswith(k):
            return v
    return cid


def pretty(cid: str) -> str:
    m = _RD.search(cid)
    if cid.startswith("cpgf") and m:
        return f"CPGF $r {m.group(1)}, d {m.group(2)}$"
    rules = (
        ("deriche_recursive_gaussian_derivative_sigma", "Deriche sigma"),
        ("derivative_of_gaussian_sigma", "DoG sigma"),
        ("freeman_adelson_g1", "Freeman-Adelson G1"),
        ("robust_local_plane_gradient", "Robust local-plane"),
        ("perona_malik_gradient", "Perona-Malik"),
        ("savitzky_golay", "Savitzky-Golay"),
        ("sparse_central_difference", "Sparse central diff."),
        ("haar_box_gradient", "Haar box"),
        ("farid_simoncelli_", "Farid-Simoncelli "),
        ("riesz_transform", "Riesz transform"),
        ("central_difference", "Central difference"),
        ("scharr_", "Scharr "),
        ("sobel_", "Sobel "),
        ("prewitt_", "Prewitt "),
        ("ando_", "Ando "),
        ("roberts", "Roberts"),
    )
    out = cid
    for k, v in rules:
        if cid.startswith(k):
            out = cid.replace(k, v, 1)
            break
    out = _SUB.sub(lambda mm: f"$sigma {mm.group(1)}$", out)
    out = (
        out.replace("radius", "r")
        .replace("degree", "d")
        .replace("iterations", "it")
        .replace("kappa", "$kappa$")
    )
    out = out.replace("_", " ").strip()
    out = re.sub(
        r"\b(Sobel|Scharr|Prewitt|Ando|Farid-Simoncelli) (\d+)\b",
        lambda m: f"{m.group(1)} ${m.group(2)} times {m.group(2)}$",
        out,
    )
    return out


def natkey(s: str) -> list[str]:
    return [t.zfill(8) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def fam_order(fam: str) -> tuple:
    if fam == "CPGF":
        return (0, "")
    if fam == "Other":
        return (2, "")
    return (1, fam)


def snr_label(s: float) -> str:
    return f"{int(s)}" if float(s).is_integer() else f"{s:g}"


def f(x: NumericCell, p: int = 3) -> str:
    return "--" if x is None else f"{x:.{p}f}"


def pct(x: object) -> NumericCell:
    return float(cast(int | float, x)) * 100 if x is not None else None


def write(name: str, header: list[str], rows: list[list[str]]) -> None:
    frame = pl.DataFrame({header[i]: [r[i] for r in rows] for i in range(len(header))})
    frame.write_csv(OUT / name)
    print(f"wrote {name} ({len(rows)} rows)")


def rd(s: str) -> tuple[int | None, int | None]:
    m = _RD.search(s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


# ----- load -----------------------------------------------------------------
fs = sorted(glob.glob("runs/synthetic/awgn_robustness/*.parquet"))
df = deduplicate_synthetic_results(pl.concat([pl.read_parquet(f) for f in fs], how="diagonal"))
SNRS = sorted(df["snr_db"].unique().to_list())
print(
    f"shards={len(fs)} snr={SNRS} seeds={df['seed'].n_unique()} "
    f"configs={df['filter_config_id'].n_unique()}"
)


def agg(sub: pl.DataFrame, metric: str, fn: str = "mean") -> pl.DataFrame:
    d = sub.filter((pl.col("metric") == metric) & (~pl.col("is_nan")))
    e = pl.col("value").median() if fn == "median" else pl.col("value").mean()
    return d.group_by("filter_config_id", "filter_family", "snr_db").agg(e.alias(metric))


def bold_per_row(rows: NumericRows, p: int, dirn: str = "min") -> list[list[str]]:
    """Format a label+numeric matrix, bolding the best value in each row."""
    out: list[list[str]] = []
    for label, vals in rows:
        nums = [(i, float(v)) for i, v in enumerate(vals) if v is not None]
        target = None
        if nums:
            key = (lambda v: abs(v)) if dirn == "zero" else (lambda v: v)
            pick = min if dirn in ("min", "zero") else max
            target = round(key(pick(nums, key=lambda t: key(t[1]))[1]), p)
        cells = [label]
        for v in vals:
            s = f(v, p)
            if (
                v is not None
                and target is not None
                and round((abs(v) if dirn == "zero" else v), p) == target
            ):
                s = "*" + s + "*"
            cells.append(s)
        out.append(cells)
    return out


# ===== Table 1+2: CPGF radius x SNR matrices (degree 1) =====================
# Rows are SNR levels, columns are support radii; reading down a row exposes the
# optimal-radius walk, with the best radius at each SNR set bold.
cp = df.filter(pl.col("filter_family") == "cpgf")
RADII_D1 = (3, 5, 7, 11, 15, 21, 31, 45)


def cpgf_matrix(metric: str, fn: str, p: int, dirn: str):
    g = (
        agg(cp, metric, fn)
        .with_columns(
            pl.col("filter_config_id")
            .map_elements(lambda s: rd(s)[0], return_dtype=pl.Int64)
            .alias("r"),
            pl.col("filter_config_id")
            .map_elements(lambda s: rd(s)[1], return_dtype=pl.Int64)
            .alias("d"),
        )
        .filter((pl.col("d") == 1) & pl.col("r").is_in(list(RADII_D1)))
    )
    rows: NumericRows = []
    for s in SNRS:
        vals: list[NumericCell] = []
        for r in RADII_D1:
            cell = g.filter((pl.col("r") == r) & (pl.col("snr_db") == s))
            vals.append(float(cell[metric][0]) if cell.height else None)
        rows.append((snr_label(s), vals))
    header = ["SNR (dB)"] + [f"$r {r}$" for r in RADII_D1]
    return header, bold_per_row(rows, p, dirn)


h, r = cpgf_matrix("nrmse", "mean", 3, "min")
write("appendix_awgn_cpgf_nrmse.csv", h, r)
h, r = cpgf_matrix("noise_gain", "median", 3, "min")
write("appendix_awgn_cpgf_noisegain.csv", h, r)

# ===== Table 3+4: best-per-family x SNR matrices ============================
# For every (folded family, SNR) the best configuration of that family is taken,
# giving each family's achievable envelope across the ladder. Two metrics:
# NRMSE and angular MAE.
allf_fam = df.with_columns(pl.col("filter_family").replace(_FOLD).alias("fam"))


def family_matrix(metric: str, fn: str, p: int, dirn: str):
    g = allf_fam.filter((pl.col("metric") == metric) & (~pl.col("is_nan")))
    e = pl.col("value").median() if fn == "median" else pl.col("value").mean()
    g = g.group_by("filter_config_id", "fam", "snr_db").agg(e.alias(metric))
    best = (
        g.sort(metric, descending=(dirn == "max"))
        .group_by("fam", "snr_db")
        .agg(pl.col(metric).first())
    )
    rows: NumericRows = []
    for s in SNRS:
        vals: list[NumericCell] = []
        for fam in FAMILIES:
            cell = best.filter((pl.col("fam") == fam) & (pl.col("snr_db") == s))
            vals.append(float(cell[metric][0]) if cell.height else None)
        rows.append((snr_label(s), vals))
    header = ["SNR (dB)"] + FAMILIES
    return header, bold_per_row(rows, p, dirn)


h, r = family_matrix("nrmse", "mean", 2, "min")
write("appendix_awgn_family_nrmse.csv", h, r)
h, r = family_matrix("angular_mae", "mean", 2, "min")
write("appendix_awgn_family_angmae.csv", h, r)

# ===== Table 5: per-SNR leaderboard =========================================
# Best filter overall and best CPGF, with the CPGF's rank in the full catalog,
# at each SNR level.
rows = []
for s in SNRS:
    sub = df.filter(pl.col("snr_db") == s)
    nr = agg(sub, "nrmse").sort("nrmse").with_row_index("rank")
    best = nr.row(0, named=True)
    cp_row = nr.filter(pl.col("filter_family") == "cpgf").row(0, named=True)
    rows.append(
        [
            snr_label(s),
            pretty(best["filter_config_id"]),
            f(best["nrmse"]),
            pretty(cp_row["filter_config_id"]),
            f(cp_row["nrmse"]),
            str(cp_row["rank"] + 1),
        ]
    )
write(
    "appendix_awgn_leaderboard.csv",
    ["SNR (dB)", "Best filter", "NRMSE", "Best CPGF", "NRMSE ", "CPGF rank"],
    rows,
)

# ===== Table 6: across-seed confidence half-widths ==========================
# Per SNR, the 95% Student-t CI half-width as a fraction of the mean across the
# eight seeds, summarized over the competitive CPGF degree-1 radii (r >= 11).
per_seed = (
    cp.filter((pl.col("metric") == "nrmse") & (~pl.col("is_nan")))
    .group_by("filter_config_id", "snr_db", "seed")
    .agg(pl.col("value").mean().alias("seed_mean"))
)
ci = (
    per_seed.group_by("filter_config_id", "snr_db")
    .agg(
        pl.col("seed_mean").mean().alias("mean"),
        pl.col("seed_mean").std(ddof=1).alias("sd"),
        pl.len().alias("n"),
    )
    .with_columns(
        pl.col("filter_config_id")
        .map_elements(lambda s: rd(s)[0], return_dtype=pl.Int64)
        .alias("r"),
        pl.col("filter_config_id")
        .map_elements(lambda s: rd(s)[1], return_dtype=pl.Int64)
        .alias("d"),
    )
    .with_columns((T_975_7 * pl.col("sd") / pl.col("n").sqrt() / pl.col("mean")).alias("frac"))
    .filter((pl.col("d") == 1) & (pl.col("r") >= 11))
)
rows = []
for s in SNRS:
    sub = ci.filter(pl.col("snr_db") == s)
    median_frac = sub["frac"].median()
    max_frac = sub["frac"].max()
    rows.append(
        [
            snr_label(s),
            f(pct(median_frac), 3),
            f(pct(max_frac), 3),
        ]
    )
write(
    "appendix_awgn_ci.csv",
    ["SNR (dB)", "Median CI (%)", "Max CI (%)"],
    rows,
)

# ===== Table 7: full per-family catalog at representative SNRs ==============
# Every configuration's NRMSE at five SNR levels spanning the ladder, split by
# family (CPGF first), the best value in each column set bold within its family.
CAT_SNRS = [0.0, 5.0, 10.0, 20.0, 30.0]
gcat = agg(df, "nrmse").with_columns(pl.col("filter_config_id"))
recs: list[tuple[str, str, list[str | NumericCell]]] = []
for cid in sorted(df["filter_config_id"].unique().to_list(), key=natkey):
    sub = gcat.filter(pl.col("filter_config_id") == cid)
    vals: list[NumericCell] = []
    for s in CAT_SNRS:
        cell = sub.filter(pl.col("snr_db") == s)
        vals.append(float(cell["nrmse"][0]) if cell.height else None)
    recs.append((family(cid), cid, [pretty(cid)] + vals))

# emit one booktabs block per family, bolding best (min) per numeric column
counts = Counter(fam for fam, _, _ in recs)
recs = [("Other" if counts[fam] < 3 else fam, cid, cells) for fam, cid, cells in recs]
cat_rows: list[list[str]] = []
for fam in sorted({fam for fam, _, _ in recs}, key=fam_order):
    grp = sorted(((cid, cells) for fm, cid, cells in recs if fm == fam), key=lambda t: natkey(t[0]))
    best = {}
    for ci_idx in range(1, 1 + len(CAT_SNRS)):
        vals: list[tuple[int, float]] = []
        for i, (_, c) in enumerate(grp):
            cell_value = c[ci_idx]
            if isinstance(cell_value, (int, float)):
                vals.append((i, float(cell_value)))
        if not vals:
            continue
        target = round(min(vals, key=lambda t: t[1])[1], 2)
        best[ci_idx] = {i for i, v in vals if round(v, 2) == target}
    for ri, (_, cells) in enumerate(grp):
        row: list[str] = [fam, str(cells[0])]
        for ci_idx in range(1, 1 + len(CAT_SNRS)):
            v = cells[ci_idx]
            numeric_v: NumericCell = v if isinstance(v, (int, float)) else None
            s = f(numeric_v, 2)
            if numeric_v is not None and ri in best.get(ci_idx, set()):
                s = "*" + s + "*"
            row.append(s)
        cat_rows.append(row)
write(
    "appendix_awgn_catalog.csv",
    ["Family", "Filter"] + [f"{snr_label(s)} dB" for s in CAT_SNRS],
    cat_rows,
)

print("\nall AWGN appendix table CSVs written to", OUT.resolve())
