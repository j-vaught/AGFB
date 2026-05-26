"""Study R - real-image edge-detection benchmark.

Studies A-E score gradient *fields* against an analytic ground truth: the scene
is synthesised, so the true gradient vector is known at every pixel and the
metrics are field-accuracy metrics (nRMSE, angular error, ...). Real photographs
and micrographs have no gradient ground truth, only human boundary / region
annotations. Study R therefore pivots to *boundary-detection* evaluation. Each
filter produces a gradient magnitude; a threshold is swept, the thresholded
edges are matched to the annotations within a small spatial tolerance, and the
precision/recall summary is reported (the BSDS500 protocol: ODS, OIS, AP).

Two edge-extraction modes are compared head to head:

``raw``
    Threshold the gradient magnitude directly (tolerant matching absorbs the
    edge thickness). Simple, and indifferent to how well a filter localises.
``nms``
    Oriented non-maximum suppression first, thinning ridges to one-pixel lines
    using the same ``(gx, gy)`` the filter returns, then threshold. Rewards
    localisation, which is where sharp operators (CPGF, DoG) should separate
    from box / heavily-smoothing filters.

Datasets (no synthetic noise is injected; the images are scored as shipped):

``bsds500``
    Natural-image boundaries, five annotators per image (``.mat``). The five
    boundary maps are unioned into one ground-truth edge map.
``drive``
    Retinal vessels. The *edge* ground truth is the boundary of the binary
    vessel mask (``find_boundaries(vessel) & fov``), exactly as BSDS500 derives
    its ``Boundaries`` map from a filled ``Segmentation``; this is scored with
    the same ODS/OIS/AP protocol, restricted to the circular field-of-view. The
    original vessel mask is also kept and scored as a detection task (ROC-AUC,
    best-threshold Dice) since vessel segmentation is a legitimate target in its
    own right.
``bbbc039``
    Fluorescence-microscopy nuclei. The supplied mask is a *filled* semantic
    segmentation (class 1 = nucleus interior), not an edge map, so the boundary
    class cannot be used directly. The edge ground truth is derived by labelling
    the interior into per-nucleus instances and taking their boundaries
    (``find_boundaries(instances, mode="thick")``), which yields a closed rim per
    nucleus including the walls between touching cells.

The tolerant-matching ODS/OIS core (distance-transform near-GT masks, a
vectorised threshold sweep instead of per-threshold dilation) is adapted from
the ``edgecritic`` evaluation harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agfb_bench.filters import FilterConfig, build_backend_sweep_grid
from agfb_bench.progress import ProgressWriter, progress_path

# A prediction edge pixel counts as a true positive if a ground-truth edge lies
# within this many pixels (the standard BSDS500 localisation tolerance).
MATCH_RADIUS = 3
# Threshold sweep resolution for the precision/recall curve.
N_THRESHOLDS = 100

DATASETS = ("bsds500", "drive", "bbbc039")

ROW_COLUMNS = (
    "study",
    "dataset",
    "filter_family",
    "filter_config_id",
    "filter_path",
    "mode",
    "metric",
    "value",
    "n_images",
    "status",
    "ms_per_image",
)


# -- samples ------------------------------------------------------------------
@dataclass
class Sample:
    """One real image plus its boundary ground truth.

    ``gray`` is the intensity in ``[0, 1]`` as ``(H, W)`` float32. ``gt`` is the
    boolean *edge* ground truth (a region boundary for every dataset). ``roi``
    restricts scoring to a region (the DRIVE FOV); a ``None`` ROI means score
    everywhere. ``vessel`` is the optional filled segmentation mask kept for
    DRIVE's secondary detection metrics; ``None`` for the edge-only datasets.
    """

    name: str
    gray: np.ndarray
    gt: np.ndarray
    roi: np.ndarray | None = None
    vessel: np.ndarray | None = None


# -- dataset loaders ----------------------------------------------------------
def _load_gray(path: Path) -> np.ndarray:
    """Load an image as float32 luminance in ``[0, 1]``."""
    from PIL import Image

    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return array / 255.0


def _listing(directory: Path, pattern: str) -> list[Path]:
    """Sorted glob that skips macOS ``._`` AppleDouble sidecars (fuse mounts)."""
    return sorted(p for p in directory.glob(pattern) if not p.name.startswith("._"))


def _binary_edge(mask: np.ndarray) -> np.ndarray:
    """Inner boundary of a binary region: foreground pixels touching background.

    Equivalent to ``skimage.segmentation.find_boundaries(mask, mode="inner")``
    for a binary mask (connectivity 1), but implemented with ``scipy`` so the
    real-image extra needs no extra dependency.
    """
    from scipy.ndimage import binary_erosion

    mask = mask.astype(bool)
    return mask & ~binary_erosion(mask)


def _instance_edge(labels: np.ndarray) -> np.ndarray:
    """Thick boundaries of a label image: pixels where the label varies locally.

    Matches ``skimage.segmentation.find_boundaries(labels, mode="thick")`` - a
    pixel is a boundary if any 4-neighbour carries a different label (including
    the background label 0), so adjacent instances are separated by their shared
    wall. Implemented as a label disagreement between a grey dilation and erosion
    over the connectivity-1 cross footprint.
    """
    from scipy.ndimage import grey_dilation, grey_erosion

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    hi = grey_dilation(labels, footprint=cross)
    lo = grey_erosion(labels, footprint=cross)
    return hi != lo


def load_bsds500(root: Path, split: str = "test") -> list[Sample]:
    """BSDS500: ``.jpg`` images + ``.mat`` ground truth (5 annotators unioned)."""
    from scipy.io import loadmat

    image_dir = root / "BSDS500" / "BSDS500" / "data" / "images" / split
    gt_dir = root / "BSDS500" / "BSDS500" / "data" / "groundTruth" / split
    samples: list[Sample] = []
    for image_path in _listing(image_dir, "*.jpg"):
        mat = loadmat(gt_dir / f"{image_path.stem}.mat")
        annotators = mat["groundTruth"][0]
        boundary = np.zeros(annotators[0]["Boundaries"][0, 0].shape, dtype=bool)
        for annotator in annotators:  # union over the five human boundary maps
            boundary |= annotator["Boundaries"][0, 0] > 0
        samples.append(Sample(image_path.stem, _load_gray(image_path), boundary))
    return samples


def load_drive(root: Path, split: str = "test") -> list[Sample]:
    """DRIVE: ``.tif`` retinal images + ``1st_manual`` vessel GT, FOV ``mask``.

    The edge ground truth is the inner boundary of the vessel mask, clipped to
    the field of view. The filled vessel mask is kept for the detection metrics.
    """
    from PIL import Image

    base = root / "DRIVE" / split
    samples: list[Sample] = []
    for image_path in _listing(base / "images", "*.tif"):
        index = image_path.stem.split("_")[0]
        vessel = np.asarray(Image.open(base / "1st_manual" / f"{index}_manual1.gif")) > 0
        fov = np.asarray(Image.open(base / "mask" / f"{index}_{split}_mask.gif")) > 0
        edge = _binary_edge(vessel) & fov
        samples.append(
            Sample(image_path.stem, _load_gray(image_path), edge, roi=fov, vessel=vessel)
        )
    return samples


def load_bbbc039(root: Path, split: str = "test", split_list: str = "test.txt") -> list[Sample]:
    """BBBC039: ``.tif`` micrographs + filled semantic mask (class 1 = interior).

    The edge ground truth is derived by labelling the nucleus interiors into
    instances and taking their thick boundaries, so touching nuclei are split by
    their shared wall. The supplied "boundary" class is *not* used: it is a
    filled training artifact, not an edge annotation.
    """
    from PIL import Image
    from scipy.ndimage import label

    base = root / "BBBC039"
    listed = (base / "metadata" / split_list).read_text().split()
    wanted = {Path(name).stem for name in listed} if listed else None
    samples: list[Sample] = []
    for image_path in _listing(base / "images", "*.tif"):
        if wanted is not None and image_path.stem not in wanted:
            continue
        mask = np.asarray(Image.open(base / "masks" / f"{image_path.stem}.png"))
        classes = mask[..., 0] if mask.ndim == 3 else mask
        instances, _ = label(classes == 1)
        edge = _instance_edge(instances)
        samples.append(Sample(image_path.stem, _load_gray(image_path), edge))
    return samples


def load_dataset(name: str, root: Path) -> list[Sample]:
    """Dispatch to the loader for ``name`` (one of :data:`DATASETS`)."""
    if name == "bsds500":
        return load_bsds500(root)
    if name == "drive":
        return load_drive(root)
    if name == "bbbc039":
        return load_bbbc039(root)
    raise ValueError(f"unknown dataset {name!r}")


# -- edge extraction ----------------------------------------------------------
def oriented_nms(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Thin ridges with Canny-style oriented non-maximum suppression.

    Each pixel is compared to its two neighbours along the gradient direction
    (quantised into the four standard sectors); pixels that are not a local
    maximum across that direction are zeroed. The surviving ridge is one pixel
    wide, which is what the BSDS protocol expects before matching.
    """
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    padded = np.pad(magnitude, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    # Neighbour pairs along each quantised orientation.
    east, west = padded[1:-1, 2:], padded[1:-1, :-2]
    north, south = padded[:-2, 1:-1], padded[2:, 1:-1]
    ne, sw = padded[:-2, 2:], padded[2:, :-2]
    nw, se = padded[:-2, :-2], padded[2:, 2:]

    keep = np.zeros_like(magnitude, dtype=bool)
    horizontal = (angle < 22.5) | (angle >= 157.5)
    diag_45 = (angle >= 22.5) & (angle < 67.5)
    vertical = (angle >= 67.5) & (angle < 112.5)
    diag_135 = (angle >= 112.5) & (angle < 157.5)

    keep |= horizontal & (center >= east) & (center >= west)
    keep |= diag_45 & (center >= ne) & (center >= sw)
    keep |= vertical & (center >= north) & (center >= south)
    keep |= diag_135 & (center >= nw) & (center >= se)
    return np.where(keep, magnitude, 0.0).astype(np.float32)


# -- metrics ------------------------------------------------------------------
def _near_gt_mask(gt_bool: np.ndarray, match_radius: int) -> np.ndarray:
    """Pixels within ``match_radius`` of a GT edge (tolerant-match region)."""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~gt_bool) <= match_radius


def compute_ods_ois_ap(
    magnitudes: list[np.ndarray],
    ground_truths: list[np.ndarray],
    rois: list[np.ndarray | None] | None = None,
    *,
    n_thresholds: int = N_THRESHOLDS,
    match_radius: int = MATCH_RADIUS,
) -> dict[str, float]:
    """BSDS500 ODS / OIS / AP via tolerant matching (adapted from edgecritic).

    Returns ``{"ods", "ois", "ap"}``. ODS is the best F-score at a single global
    threshold across all images; OIS averages each image's own best F-score; AP
    is the area under the dataset precision/recall curve. When ``rois`` is given,
    only pixels inside each image's ROI are scored (the DRIVE FOV border would
    otherwise be counted as a wall of false-positive edges).
    """
    from scipy.ndimage import maximum_filter

    max_val = max((float(m.max()) for m in magnitudes), default=0.0)
    if max_val <= 0.0:
        return {"ods": 0.0, "ois": 0.0, "ap": 0.0}
    thresholds = np.linspace(0.0, max_val, n_thresholds)
    if rois is None:
        rois = [None] * len(magnitudes)

    all_tp = np.zeros(n_thresholds)
    all_fp = np.zeros(n_thresholds)
    all_fn = np.zeros(n_thresholds)
    per_image_best_f: list[float] = []

    for mag, gt, roi in zip(magnitudes, ground_truths, rois, strict=True):
        gt_bool = gt > 0
        if not gt_bool.any():
            per_image_best_f.append(0.0)
            continue
        roi_bool = np.ones(mag.shape, dtype=bool) if roi is None else roi.astype(bool)
        near = _near_gt_mask(gt_bool, match_radius) & roi_bool
        if match_radius > 0:
            local_max = maximum_filter(mag, size=2 * match_radius + 1)
            gt_local_max = np.sort(local_max[gt_bool])
        else:
            gt_local_max = np.sort(mag[gt_bool])
        near_sorted = np.sort(mag[near])
        far_sorted = np.sort(mag[~near & roi_bool])

        tp = len(near_sorted) - np.searchsorted(near_sorted, thresholds, side="right")
        fp = len(far_sorted) - np.searchsorted(far_sorted, thresholds, side="right")
        fn = np.searchsorted(gt_local_max, thresholds, side="right")
        all_tp += tp
        all_fp += fp
        all_fn += fn

        prec = np.divide(tp, tp + fp, out=np.zeros(n_thresholds), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros(n_thresholds), where=(tp + fn) > 0)
        denom = prec + rec
        image_f = np.divide(2 * prec * rec, denom, out=np.zeros(n_thresholds), where=denom > 0)
        per_image_best_f.append(float(image_f.max()))

    prec = np.divide(
        all_tp, all_tp + all_fp, out=np.zeros(n_thresholds), where=(all_tp + all_fp) > 0
    )
    rec = np.divide(
        all_tp, all_tp + all_fn, out=np.zeros(n_thresholds), where=(all_tp + all_fn) > 0
    )
    denom = prec + rec
    f_scores = np.divide(2 * prec * rec, denom, out=np.zeros(n_thresholds), where=denom > 0)

    # AP = area under the precision/recall curve, integrated over recall.
    order = np.argsort(rec)
    ap = float(np.trapezoid(prec[order], rec[order]))
    return {"ods": float(f_scores.max()), "ois": float(np.mean(per_image_best_f)), "ap": ap}


# -- orientation accuracy on real edges ---------------------------------------
def edge_normal_orientation(gt_bool: np.ndarray) -> np.ndarray:
    """Local boundary-normal axis (radians, mod pi) at every pixel from the GT.

    A smoothed edge map gives a ridge whose gradient points across the contour;
    the structure tensor of that gradient, averaged over a small neighbourhood,
    has a dominant eigenvector aligned with the edge normal. Its angle is the
    real-data analogue of the synthetic true gradient orientation, defined here
    purely from the human annotation (no analytic field required).
    """
    from scipy.ndimage import gaussian_filter

    e = gaussian_filter(gt_bool.astype(np.float32), sigma=1.0)
    ey, ex = np.gradient(e)
    jxx = gaussian_filter(ex * ex, 1.5)
    jyy = gaussian_filter(ey * ey, 1.5)
    jxy = gaussian_filter(ex * ey, 1.5)
    # angle of the dominant gradient (normal) direction, an axis modulo pi.
    return (0.5 * np.arctan2(2.0 * jxy, jxx - jyy)).astype(np.float32)


def orientation_error_sum(
    gx: np.ndarray, gy: np.ndarray, normal: np.ndarray, eval_mask: np.ndarray
) -> tuple[float, int]:
    """Summed acute angle (deg) between the filter gradient and the edge normal.

    Both the gradient and the normal are treated as undirected lines, so the
    error folds into ``[0, 90]`` (0 = gradient perfectly across the contour, 45 =
    chance). Evaluated only at ``eval_mask`` pixels (GT edges inside any ROI).
    Returns ``(sum_of_errors_deg, n_pixels)`` so callers can pool across images.
    """
    theta_g = np.arctan2(gy, gx)
    d = np.abs(theta_g - normal) % np.pi
    d = np.minimum(d, np.pi - d)
    deg = np.rad2deg(d)[eval_mask]
    return float(deg.sum()), int(deg.size)


def compute_vessel_scores(
    magnitudes: list[np.ndarray],
    ground_truths: list[np.ndarray],
    rois: list[np.ndarray | None],
    *,
    n_thresholds: int = N_THRESHOLDS,
) -> dict[str, float]:
    """DRIVE vessel detection: ROC-AUC and best-threshold Dice, inside the FOV.

    The magnitude is treated as a per-pixel vessel score. Only pixels inside the
    FOV ``roi`` are scored (the dark circular border would otherwise dominate).
    """
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for mag, gt, roi in zip(magnitudes, ground_truths, rois, strict=True):
        mask = np.ones(mag.shape, dtype=bool) if roi is None else roi
        scores.append(mag[mask].ravel())
        labels.append((gt[mask] > 0).ravel())
    score = np.concatenate(scores)
    label = np.concatenate(labels)
    positives = int(label.sum())
    if positives == 0 or positives == label.size:
        return {"auc": 0.0, "dice": 0.0}

    # ROC-AUC via the rank-sum (Mann-Whitney U) identity.
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    # Average ranks within ties so the AUC is unbiased for quantised magnitudes.
    _assign_tied_ranks(score[order], ranks, order)
    auc = (ranks[label].sum() - positives * (positives + 1) / 2.0) / (
        positives * (label.size - positives)
    )

    max_val = float(score.max())
    thresholds = np.linspace(0.0, max_val, n_thresholds)
    best_dice = 0.0
    label_sum = float(label.sum())
    for threshold in thresholds:
        pred = score > threshold
        tp = float(np.count_nonzero(pred & label))
        denom = float(np.count_nonzero(pred)) + label_sum
        if denom > 0:
            best_dice = max(best_dice, 2.0 * tp / denom)
    return {"auc": float(auc), "dice": float(best_dice)}


def _assign_tied_ranks(sorted_score: np.ndarray, ranks: np.ndarray, order: np.ndarray) -> None:
    """Replace ranks of equal scores with their average (in place)."""
    n = sorted_score.size
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_score[end] == sorted_score[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end


# -- supersampled (anti-aliased) filtering ------------------------------------
def _gaussian_1d(device: torch.device, taps: int = 9, sigma: float = 1.5) -> torch.Tensor:
    """A normalised 1D Gaussian of ``taps`` samples (default 9-tap, sigma 1.5)."""
    r = (taps - 1) // 2
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def _upscale_blur(image: torch.Tensor, factor: int, k1d: torch.Tensor) -> torch.Tensor:
    """Bilinearly upscale a ``(1, H, W)`` image ``factor``x, then blur it with a
    separable Gaussian (``k1d``). Returns the ``(1, factor*H, factor*W)`` image."""
    import torch.nn.functional as F

    up = F.interpolate(
        image.unsqueeze(0), scale_factor=factor, mode="bilinear", align_corners=False
    )
    r = (k1d.numel() - 1) // 2
    up = F.conv2d(F.pad(up, (r, r, 0, 0), mode="reflect"), k1d.view(1, 1, 1, -1))
    up = F.conv2d(F.pad(up, (0, 0, r, r), mode="reflect"), k1d.view(1, 1, -1, 1))
    return up.squeeze(0)


def _downscale_grad(g: torch.Tensor, factor: int) -> torch.Tensor:
    """Area-average a ``(1, factor*H, factor*W)`` gradient component back to
    ``(1, H, W)`` (the linear inverse of the bilinear upscale)."""
    import torch.nn.functional as F

    return F.avg_pool2d(g.unsqueeze(0), kernel_size=factor).squeeze(0)


# -- orchestration ------------------------------------------------------------
def _frames_for(samples: list[Sample], device: torch.device) -> list[torch.Tensor]:
    """Stage each sample's intensity as a ``(1, H, W)`` float32 tensor on device."""
    return [
        torch.from_numpy(sample.gray).to(device=device, dtype=torch.float32).unsqueeze(0)
        for sample in samples
    ]


def run_realimg(
    *,
    device: torch.device,
    out_dir: Path,
    data_root: Path,
    datasets: tuple[str, ...] = DATASETS,
    filters: list[FilterConfig] | None = None,
    modes: tuple[str, ...] = ("raw", "nms"),
    shard_index: int = 0,
    shard_count: int = 1,
    supersample: int = 1,
) -> dict:
    """Study R: score the filter catalog on real-image boundary annotations.

    For each ``(dataset, filter, mode)`` every image is filtered into a gradient
    magnitude, optionally thinned by oriented NMS, and scored against the human
    annotations. One Parquet row is written per metric. ``shard_index`` /
    ``shard_count`` slice the filter list so several GPUs can split the catalog.

    When ``supersample > 1`` each image is bilinearly upscaled by that factor,
    blurred with a 9-tap Gaussian, filtered at the higher resolution, and the
    gradient components are area-averaged back to native resolution before the
    magnitude is recomputed and scored. This anti-aliased path is written under a
    distinct study label (``supersampled``) so it never clobbers the baseline
    (``edges``).
    """
    import agfb_filters
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    catalog = filters if filters is not None else build_backend_sweep_grid()
    shard = catalog[shard_index::shard_count]
    study_label = "edges" if supersample <= 1 else "supersampled"
    gauss = _gaussian_1d(device) if supersample > 1 else None

    loaded = {name: load_dataset(name, data_root) for name in datasets}
    frames = {name: _frames_for(samples, device) for name, samples in loaded.items()}
    # Per-sample GT edge normal axis + the pixels (GT edges inside any ROI) where
    # the gradient orientation error is evaluated. Computed once, reused per filter.
    normals = {
        name: [
            (
                edge_normal_orientation(s.gt > 0),
                (s.gt > 0) & (np.ones_like(s.gt, dtype=bool) if s.roi is None else s.roi),
            )
            for s in samples
        ]
        for name, samples in loaded.items()
    }

    total_units = len(shard) * len(datasets)
    suffix = f"_shard{shard_index:02d}" if shard_count > 1 else ""
    reporter = ProgressWriter(
        progress_path(out_dir, f"{study_label}{suffix}", shard_index),
        # Distinct heartbeat files per shard, but one shared study name so the
        # dashboard merges the shards into a single row (it groups by study and
        # keys shards by host+seed), matching how studies A-E report.
        study=study_label,
        seed=shard_index,
        device=str(device),
        n_cells=total_units,
        n_conditions=len(modes),
        n_filters=len(shard),
    )

    rows: list[dict] = []
    units_done = 0
    try:
        for config in shard:
            definition = agfb_filters.get_filter_definition(config.family, **config.params)
            path = agfb_filters.ExecutionPath[config.path]
            for dataset in datasets:
                samples = loaded[dataset]
                raw_mags: list[np.ndarray] = []
                nms_mags: list[np.ndarray] = []
                ori_sum = 0.0
                ori_n = 0
                status = "ok"
                start = time.perf_counter()
                try:
                    for image, (normal, eval_mask) in zip(
                        frames[dataset], normals[dataset], strict=True
                    ):
                        if supersample > 1:
                            hi = _upscale_blur(image, supersample, gauss)
                            gx, gy = agfb_filters.run_filter(
                                definition, hi, path=path, boundary=definition.default_boundary
                            )
                            gx = _downscale_grad(gx, supersample)
                            gy = _downscale_grad(gy, supersample)
                        else:
                            gx, gy = agfb_filters.run_filter(
                                definition, image, path=path, boundary=definition.default_boundary
                            )
                        gx_np = gx[0].detach().cpu().numpy()
                        gy_np = gy[0].detach().cpu().numpy()
                        mag = np.hypot(gx_np, gy_np).astype(np.float32)
                        raw_mags.append(mag)
                        if "nms" in modes:
                            nms_mags.append(oriented_nms(mag, gx_np, gy_np))
                        # orientation error is gradient-direction-based, so it is
                        # the same for raw/nms; accumulate it once here.
                        err_sum, err_n = orientation_error_sum(gx_np, gy_np, normal, eval_mask)
                        ori_sum += err_sum
                        ori_n += err_n
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError:
                    status = "oom"
                    gc_cuda(device)
                except (ValueError, NotImplementedError, RuntimeError) as error:
                    status = f"error:{type(error).__name__}"
                ms_per_image = (time.perf_counter() - start) / max(len(samples), 1) * 1e3
                orientation_mae = ori_sum / ori_n if ori_n > 0 else float("nan")

                for mode in modes:
                    mags = raw_mags if mode == "raw" else nms_mags
                    metrics = _score(dataset, mags, samples) if status == "ok" else {}
                    # emit orientation MAE once, on the raw rows (mode-independent)
                    if status == "ok" and mode == "raw":
                        metrics = {**metrics, "orientation_mae": orientation_mae}
                    if not metrics:
                        metrics = {"status_only": float("nan")}
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "study": study_label,
                                "dataset": dataset,
                                "filter_family": config.family,
                                "filter_config_id": config.config_id,
                                "filter_path": config.path,
                                "mode": mode,
                                "metric": metric,
                                "value": value,
                                "n_images": len(samples),
                                "status": status,
                                "ms_per_image": round(ms_per_image, 4),
                            }
                        )
                units_done += 1
                reporter.update(cells_done=units_done, rows=len(rows))
    except Exception as error:  # noqa: BLE001 - record then re-raise so the shard fails loudly
        reporter.finish(status="error", error=f"{type(error).__name__}: {error}")
        raise

    parquet = out_dir / f"{study_label}{suffix}.parquet"
    pl.DataFrame(rows).write_parquet(parquet)
    reporter.finish(status="done", rows=len(rows))
    return {
        "study": study_label,
        "datasets": list(datasets),
        "n_filters": len(shard),
        "n_rows": len(rows),
        "shard": f"{shard_index}/{shard_count}",
    }


def _score(dataset: str, mags: list[np.ndarray], samples: list[Sample]) -> dict[str, float]:
    """Score one dataset's magnitudes with the protocol(s) that fit its GT.

    Every dataset gets edge ODS/OIS/AP against its region-boundary GT. DRIVE
    additionally reports vessel-detection ROC-AUC and Dice against the filled
    mask, since vessel segmentation is a distinct task worth keeping.
    """
    rois = [s.roi for s in samples]
    edge = compute_ods_ois_ap(mags, [s.gt for s in samples], rois)
    if dataset == "drive":
        vessel = compute_vessel_scores(mags, [s.vessel for s in samples], rois)
        return {**edge, **vessel}
    return edge


def gc_cuda(device: torch.device) -> None:
    """Release cached CUDA memory after an OOM so the sweep can continue."""
    import gc

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
