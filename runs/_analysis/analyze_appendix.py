"""Detailed-results tables for the benchmark appendix.

Emits display-ready CSVs (numbers pre-formatted as strings) for the appendix
tables. Each CSV's first row is the header, matching the `csvbooktabs` Typst
helper. Reductions follow the AGFB protocol used in analyze.py: per-image metric
values are averaged over cells and seeds, except noise_gain, which uses the
outlier-robust median. Output lands in ../PGF_paper/figures/tables relative to
the AGFB working directory.
"""

import glob
import re
from collections import Counter
from pathlib import Path

import polars as pl

OUT = Path("../PGF_paper/figures/tables")
OUT.mkdir(parents=True, exist_ok=True)


def load(subdir: str) -> pl.DataFrame:
    fs = sorted(glob.glob(f"runs/{subdir}/*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in fs], how="diagonal")


def write(name: str, header: list[str], rows: list[list]) -> None:
    frame = pl.DataFrame({header[i]: [r[i] for r in rows] for i in range(len(header))})
    frame.write_csv(OUT / name)
    print(f"wrote {name} ({len(rows)} rows)")


# ----- filter display names -------------------------------------------------
# Capture the full scale, including a decimal part, so "sigma0.5" renders as one
# math run "$sigma 0.5$" rather than "$sigma 0$.5", which leaves ".5" as body
# text and opens a gap before the decimal.
_SUB = re.compile(r"sigma(\d+(?:\.\d+)?)")
_RD = re.compile(r"radius(\d+)_degree(\d+)")


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
    # Render the support scale inside a math run, e.g. "sigma4" -> "$sigma 4$".
    out = _SUB.sub(lambda m: f"$sigma {m.group(1)}$", out)
    out = (
        out.replace("radius", "r")
        .replace("degree", "d")
        .replace("iterations", "it")
        .replace("kappa", "$kappa$")
    )
    out = out.replace("_", " ").strip()
    # stencil sizes: "Sobel 5" -> "Sobel $5 times 5$"
    out = re.sub(
        r"\b(Sobel|Scharr|Prewitt|Ando|Farid-Simoncelli) (\d+)\b",
        lambda m: f"{m.group(1)} ${m.group(2)} times {m.group(2)}$",
        out,
    )
    return out


# ----- coarse filter-type grouping ------------------------------------------
# The stored filter_family column splits stencils by size (sobel_3, sobel_5),
# which is too fine for the appendix. These rules fold each operator into a
# single display family. Order matters: longer prefixes precede their stems.
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


def is_even_degree_redundant(cid: str) -> bool:
    """True for an even-degree local-polynomial gradient configuration (CPGF or
    Savitzky-Golay). On the symmetric support the even polynomial terms are
    orthogonal to the linear gradient term, so an even degree gives the same
    gradient estimate as the preceding odd degree (e.g. degree 2 == degree 1).
    These parity-redundant rows are dropped from the per-configuration tables."""
    m = _RD.search(cid)
    return (
        m is not None
        and (cid.startswith("cpgf") or cid.startswith("savitzky_golay"))
        and int(m.group(2)) % 2 == 0
    )


def natkey(s: str) -> list[str]:
    """Natural-ordering key. Digit runs are zero-padded so radius 3 precedes
    radius 11 and so the key stays all-strings (safe to compare across the
    mixed config ids that land in the merged "Other" table)."""
    return [t.zfill(8) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def fam_order(fam: str) -> tuple:
    """CPGF first, the merged bucket last, everything else alphabetical."""
    if fam == "CPGF":
        return (0, "")
    if fam == "Other":
        return (2, "")
    return (1, fam)


def emit(fname: str, header: list[str], recs: list[tuple], cols: list[dict], min_keep: int = 3):
    """Render one booktabs table per filter family.

    `recs` are (family, config_id, cells) where cells line up with `cols` and
    with `header` after the leading Family column. Families with fewer than
    `min_keep` configurations are merged into a single "Other" table. Within
    each family rows are sorted by name (natural order on the config id) and the
    best value in every numeric column is wrapped in Typst strong markup so the
    `csvfamilies` helper renders it bold.

    Column descriptors: {"kind": "name"} for the label column, {"kind": "text"}
    for a plain string column, and {"kind": "num", "p": <prec>, "dir": <min|max
    |zero>, "signed": <bool>} for a metric column.
    """
    counts = Counter(fam for fam, _, _ in recs)
    recs = [("Other" if counts[fam] < min_keep else fam, cid, cells) for fam, cid, cells in recs]
    out = []
    for fam in sorted({fam for fam, _, _ in recs}, key=fam_order):
        grp = sorted(
            ((cid, cells) for f, cid, cells in recs if f == fam), key=lambda t: natkey(t[0])
        )
        best = {}
        for ci, col in enumerate(cols):
            if col["kind"] != "num":
                continue
            key = (lambda v: abs(v)) if col["dir"] == "zero" else (lambda v: v)
            vals = [(i, c[ci]) for i, (_, c) in enumerate(grp) if isinstance(c[ci], (int, float))]
            if not vals:
                continue
            pick = min if col["dir"] in ("min", "zero") else max
            target = round(key(pick(vals, key=lambda t: key(t[1]))[1]), col["p"])
            best[ci] = {i for i, v in vals if round(key(v), col["p"]) == target}
        for ri, (_, cells) in enumerate(grp):
            row = [fam]
            for ci, col in enumerate(cols):
                v = cells[ci]
                if col["kind"] == "name":
                    s = v
                elif col["kind"] == "text":
                    s = v if v is not None else "--"
                elif v is None:
                    s = "--"
                else:
                    s = f(v, col["p"])
                    if col.get("signed") and v >= 0:
                        s = "+" + s
                    if ri in best.get(ci, set()):
                        s = "*" + s + "*"
                row.append(s)
            out.append(row)
    write(fname, ["Family"] + header, out)


def agg(df: pl.DataFrame, metric: str, fn: str = "mean") -> pl.DataFrame:
    d = df.filter((pl.col("metric") == metric) & (~pl.col("is_nan")))
    e = pl.col("value").median() if fn == "median" else pl.col("value").mean()
    return d.group_by("filter_config_id", "filter_family").agg(e.alias(metric))


def f(x, p=3):
    return "--" if x is None else f"{x:.{p}f}"


# ===== Study A: full clean-field catalog ====================================
# noise_gain is undefined on the clean field (SNR=inf), so the clean catalog
# reports accuracy metrics only; noise gain appears in the CPGF grid below.
A = load("synthetic/clean_accuracy")
metrics_A = [
    "nrmse",
    "angular_mae",
    "magnitude_bias",
    "tail_vector_error",
    "localization_offset",
    "edge_fwhm",
]
cat = agg(A, metrics_A[0])
for m in metrics_A[1:]:
    cat = cat.join(agg(A, m).select("filter_config_id", m), on="filter_config_id", how="left")
recs = [
    (
        family(r["filter_config_id"]),
        r["filter_config_id"],
        [
            pretty(r["filter_config_id"]),
            r["nrmse"],
            r["angular_mae"],
            r["magnitude_bias"],
            r["tail_vector_error"],
            r["localization_offset"],
            r["edge_fwhm"],
        ],
    )
    for r in cat.iter_rows(named=True)
    if not is_even_degree_redundant(r["filter_config_id"])
]
emit(
    "appendix_clean_catalog.csv",
    ["Filter", "NRMSE", "Ang. MAE", "Mag. bias", "Tail vec.", "Localiz.", "FWHM"],
    recs,
    [
        {"kind": "name"},
        {"kind": "num", "p": 3, "dir": "min"},
        {"kind": "num", "p": 2, "dir": "min"},
        {"kind": "num", "p": 3, "dir": "zero"},
        {"kind": "num", "p": 3, "dir": "min"},
        {"kind": "num", "p": 2, "dir": "min"},
        {"kind": "num", "p": 2, "dir": "min"},
    ],
)

# ===== Study A: per-structure-class breakdown ===============================
# Two tables, one ranked by NRMSE and one by angular MAE, each self-consistent:
# the leader and the leading CPGF are selected by the table's own metric.
classes = sorted(A["structure_class"].unique().to_list())


def per_class(metric: str, prec: int):
    rows = []
    for cls in classes:
        sub = A.filter(pl.col("structure_class") == cls)
        j = agg(sub, metric)
        if j.height == 0:  # smooth-surface class has no signal pixels to score
            continue
        best = j.sort(metric).row(0, named=True)
        cp = j.filter(pl.col("filter_family") == "cpgf").sort(metric).row(0, named=True)
        rows.append(
            [
                cls.capitalize(),
                pretty(best["filter_config_id"]),
                f(best[metric], prec),
                pretty(cp["filter_config_id"]),
                f(cp[metric], prec),
            ]
        )
    return rows


write(
    "appendix_per_class_nrmse.csv",
    ["Class", "Best filter", "NRMSE", "Best CPGF", "NRMSE "],
    per_class("nrmse", 3),
)
write(
    "appendix_per_class_mae.csv",
    ["Class", "Best filter", "Ang. MAE", "Best CPGF", "Ang. MAE "],
    per_class("angular_mae", 2),
)

# ===== Study C: per-noise-model leaderboard =================================
C = load("synthetic/noise_breadth")
models = sorted(C["noise_model"].unique().to_list())
rows = []
for mdl in models:
    sub = C.filter(pl.col("noise_model") == mdl)
    nr = agg(sub, "nrmse").sort("nrmse").with_row_index("rank")
    best = nr.row(0, named=True)
    cp = nr.filter(pl.col("filter_family") == "cpgf").row(0, named=True)
    label = mdl.replace("add_", "").replace("_", " ")
    rows.append(
        [
            label,
            pretty(best["filter_config_id"]),
            f(best["nrmse"]),
            pretty(cp["filter_config_id"]),
            f(cp["nrmse"]),
            str(cp["rank"] + 1),
        ]
    )
write(
    "appendix_noise_leaderboard.csv",
    ["Noise model", "Best filter", "NRMSE", "Best CPGF", "NRMSE ", "CPGF rank"],
    rows,
)

# ===== Study CG: CPGF radius x degree grids =================================
CG = load("synthetic/cpgf_grid")


def rd(s):
    m = _RD.search(s)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def grid(metric, fn, p=3):
    g = agg(CG.filter(pl.col("filter_family") == "cpgf"), metric, fn)
    g = g.with_columns(
        pl.col("filter_config_id")
        .map_elements(lambda s: rd(s)[0], return_dtype=pl.Int64)
        .alias("r"),
        pl.col("filter_config_id")
        .map_elements(lambda s: rd(s)[1], return_dtype=pl.Int64)
        .alias("d"),
    )
    radii = sorted(g["r"].unique().to_list())
    degs = sorted(g["d"].unique().to_list())
    rows = []
    for r in radii:
        row = [f"$r {r}$"]
        for d in degs:
            cell = g.filter((pl.col("r") == r) & (pl.col("d") == d))
            row.append(f(cell[metric][0], p) if cell.height else "--")
        rows.append(row)
    header = ["Radius"] + [f"$d {d}$" for d in degs]
    return header, rows


h, r = grid("nrmse", "mean")
write("appendix_cpgf_grid_nrmse.csv", h, r)
h, r = grid("noise_gain", "median")
write("appendix_cpgf_grid_noisegain.csv", h, r)

# ===== Study D: full wall-clock timing ======================================
D = load("timing/walltime_scaling")
recs = [
    (
        family(r["filter_config_id"]),
        r["filter_config_id"],
        [
            pretty(r["filter_config_id"]),
            r["filter_path"].replace("_", " ").title(),
            r["ms_per_call"],
        ],
    )
    for r in D.iter_rows(named=True)
]
emit(
    "appendix_timing_full.csv",
    ["Filter", "Execution path", "ms/call"],
    recs,
    [{"kind": "name"}, {"kind": "text"}, {"kind": "num", "p": 2, "dir": "min"}],
)

# ===== backend_timing: backend x radius matrix (4096, degree 1) =============
E = pl.read_parquet("runs/timing/backend_timing/backend_timing_sweep.parquet")
e = E.filter(
    (pl.col("filter_family") == "cpgf")
    & (pl.col("status") == "ok")
    & (pl.col("image_size") == 4096)
    & pl.col("filter_config_id").str.contains("degree1")
).with_columns(
    pl.col("filter_config_id").map_elements(lambda s: rd(s)[0], return_dtype=pl.Int64).alias("r")
)
# ANTIPODAL_PAIRS is omitted: it degenerates at the largest radii (r63 returns
# in 0.26 ms for low degree) and is not a default backend.
paths_E = ["FFT", "SPATIAL_DENSE", "SPARSE_OFFSETS"]
labels_E = ["FFT", "Dense spatial", "Sparse offsets"]
radii = sorted(e["r"].unique().to_list())
rows = []
for r in radii:
    row = [f"$r {r}$"]
    for p in paths_E:
        cell = e.filter((pl.col("r") == r) & (pl.col("forced_path") == p))
        row.append(f(cell["ms_per_call"][0], 2) if cell.height else "--")
    rows.append(row)
write("appendix_backend_matrix.csv", ["Radius"] + labels_E, rows)

# ===== realimg/edges: per-dataset full rankings =============================
R = pl.concat([pl.read_parquet(p) for p in glob.glob("runs/realimg/edges/*.parquet")])


def real_table(dataset):
    sub = R.filter((pl.col("dataset") == dataset) & (pl.col("status") == "ok"))

    def pivot(mode):
        return sub.filter(pl.col("mode") == mode).pivot(
            values="value", index="filter_config_id", on="metric"
        )

    raw = pivot("raw")
    nms = pivot("nms").select("filter_config_id", pl.col("ods").alias("ods_nms"))
    t = raw.join(nms, on="filter_config_id", how="left")
    return [
        (
            family(r["filter_config_id"]),
            r["filter_config_id"],
            [
                pretty(r["filter_config_id"]),
                r.get("ods"),
                r.get("ois"),
                r.get("ap"),
                r.get("orientation_mae"),
                r.get("ods_nms"),
            ],
        )
        for r in t.iter_rows(named=True)
        if not is_even_degree_redundant(r["filter_config_id"])
    ]


for ds in ["bsds500", "drive", "bbbc039"]:
    emit(
        f"appendix_real_{ds}.csv",
        ["Filter", "ODS", "OIS", "AP", "Orient.", "ODS (nms)"],
        real_table(ds),
        [
            {"kind": "name"},
            {"kind": "num", "p": 3, "dir": "max"},
            {"kind": "num", "p": 3, "dir": "max"},
            {"kind": "num", "p": 3, "dir": "max"},
            {"kind": "num", "p": 1, "dir": "min"},
            {"kind": "num", "p": 3, "dir": "max"},
        ],
    )

# ===== realimg/supersampled: native vs supersampled =========================
Rs = pl.concat([pl.read_parquet(p) for p in glob.glob("runs/realimg/supersampled/*.parquet")])


def best_ods(df, dataset):
    return df.filter(
        (pl.col("dataset") == dataset)
        & (pl.col("mode") == "raw")
        & (pl.col("metric") == "ods")
        & (pl.col("status") == "ok")
    ).select("filter_config_id", "value")


nat = best_ods(R, "bsds500").rename({"value": "native"})
ss8 = best_ods(Rs, "bsds500").rename({"value": "ss8"})
m = nat.join(ss8, on="filter_config_id").with_columns(
    (pl.col("ss8") - pl.col("native")).alias("delta")
)
recs = [
    (
        family(r["filter_config_id"]),
        r["filter_config_id"],
        [pretty(r["filter_config_id"]), r["native"], r["ss8"], r["delta"]],
    )
    for r in m.iter_rows(named=True)
    if not is_even_degree_redundant(r["filter_config_id"])
]
emit(
    "appendix_real_ss8.csv",
    ["Filter", "Native ODS", "$8 times$ ODS", "$Delta$"],
    recs,
    [
        {"kind": "name"},
        {"kind": "num", "p": 3, "dir": "max"},
        {"kind": "num", "p": 3, "dir": "max"},
        {"kind": "num", "p": 3, "dir": "max", "signed": True},
    ],
)

print("\nall appendix table CSVs written to", OUT.resolve())
