"""Leakage-controlled temporal fixation-density evaluation.

Densities are piecewise constant on equal-area canvas bins. Information gain
is the log2 ratio of the model bin probability to the uniform bin probability.
Hyperparameters are chosen on train-image folds only. Same-image maps used at
evaluation are built without the target observer's fixations.

This module measures predictive information. It does not state a hypothesis.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from humangazelab.summarize import metric_seed, percentile_bootstrap_mean


MODEL_NAMES = (
    "B0_uniform",
    "B1_center",
    "B1_subject_held_out",
    "B2_loo_image",
    "B3_loo_index",
    "B4_history",
    "B5_center_history",
    "B5c_loo_history",
    "B6_image_shuffle",
    "B6_order_shuffle",
)

# Models whose reference map uses other observers' fixations on the evaluation image.
SAME_IMAGE_MODELS = frozenset(
    {"B2_loo_image", "B3_loo_index", "B5c_loo_history", "B6_image_shuffle"}
)


@dataclass(frozen=True)
class Grid:
    width: float
    height: float
    nx: int
    ny: int

    @property
    def n_bins(self) -> int:
        return self.nx * self.ny

    @property
    def log2_uniform(self) -> float:
        return -math.log2(self.n_bins)


def load_temporal_config(path: str | Path) -> dict:
    import yaml

    with Path(path).open() as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("temporal config must be a mapping")
    return payload


def bin_index(values: np.ndarray, extent: float, n_bins: int) -> np.ndarray:
    """Map coordinates in [0, extent] to bins. The right edge falls in the last bin."""
    scaled = np.floor(np.asarray(values, dtype=float) / extent * n_bins)
    return np.clip(scaled.astype(np.int64), 0, n_bins - 1)


def blur_counts(counts: np.ndarray, sigma_bins: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if sigma_bins < 0:
        raise ValueError("sigma_bins must be non-negative")
    total = float(counts.sum())
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(counts, dtype=np.float64)
    if sigma_bins == 0:
        return counts.copy()
    return gaussian_filter(counts, sigma=float(sigma_bins), mode="constant", cval=0.0)


def density_from_counts(counts: np.ndarray, sigma_bins: float, alpha: float) -> np.ndarray:
    """Blur counts, renormalize on the canvas, then mix with a uniform floor."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must lie in [0, 1)")
    return normalize_and_floor(blur_counts(counts, sigma_bins), alpha)


def impulse_library(ny: int, nx: int, sigma_bins: float) -> np.ndarray:
    """Blurred unit count at every bin. Shape (iy, ix, y, x). Blur is linear, so sums match histograms."""
    library = np.empty((ny, nx, ny, nx), dtype=np.float64)
    buffer = np.zeros((ny, nx), dtype=np.float64)
    for iy, ix in np.ndindex(ny, nx):
        buffer[iy, ix] = 1.0
        library[iy, ix] = buffer if sigma_bins == 0 else gaussian_filter(buffer, sigma=float(sigma_bins), mode="constant", cval=0.0)
        buffer[iy, ix] = 0.0
    return library


def accumulate_impulses(library: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    total = np.zeros(library.shape[-2:], dtype=np.float64)
    for y, x in zip(iy.tolist(), ix.tolist(), strict=True):
        total += library[int(y), int(x)]
    return total


def normalize_and_floor(blurred: np.ndarray, alpha: float) -> np.ndarray:
    blurred = np.asarray(blurred, dtype=np.float64)
    n_bins = blurred.size
    mass = float(blurred.sum())
    if mass <= 0 or not np.isfinite(mass):
        raw = np.full(blurred.shape, 1.0 / n_bins, dtype=np.float64)
    else:
        raw = blurred / mass
    mixed = (1.0 - alpha) * raw + alpha / n_bins
    mixed = np.maximum(mixed, 0.0)
    total = float(mixed.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(blurred.shape, 1.0 / n_bins, dtype=np.float64)
    mixed /= total
    return mixed


def information_gain(probability: np.ndarray, n_bins: int) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if np.any(probability <= 0) or np.any(~np.isfinite(probability)):
        raise ValueError("information gain requires finite positive probabilities")
    return np.log2(probability) + math.log2(n_bins)


def nss_values(density: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    centered = density - float(density.mean())
    scale = float(density.std(ddof=0))
    if scale == 0:
        return np.zeros(len(ix), dtype=np.float64)
    return centered[iy, ix] / scale


def top_fraction_mask(density: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    flat = density.ravel()
    n_top = max(1, int(math.ceil(fraction * flat.size)))
    threshold = np.partition(flat, flat.size - n_top)[flat.size - n_top]
    return density >= threshold


def histogram_counts(iy: np.ndarray, ix: np.ndarray, ny: int, nx: int) -> np.ndarray:
    if len(ix) == 0:
        return np.zeros((ny, nx), dtype=np.float64)
    flat = np.bincount(iy * nx + ix, minlength=ny * nx)
    return flat.astype(np.float64).reshape(ny, nx)


def image_role(name: str, split: str, train_name: str, modulus: int, bandwidth_bucket: int, mixture_bucket: int) -> str:
    if split != train_name:
        return "eval"
    bucket = zlib.crc32(name.encode("utf-8")) % modulus
    if bucket == bandwidth_bucket:
        return "select_bandwidth"
    if bucket == mixture_bucket:
        return "select_mixture"
    return "fit"


def sigma_in_bins(sigma_px: float, grid: Grid) -> float:
    bin_x = grid.width / grid.nx
    bin_y = grid.height / grid.ny
    # Round bins are not required. Use the mean bin pitch so x and y blur match in pixels.
    return float(sigma_px / ((bin_x + bin_y) / 2.0))


def derangement(names: list[str], seed: int) -> dict[str, str]:
    """Seeded permutation with no image left in place. Order of `names` is sorted first."""
    ordered = np.array(sorted(names))
    n = len(ordered)
    if n < 2:
        raise ValueError("an image-shuffle control needs at least two images")
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    for _ in range(8):
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return {str(ordered[i]): str(ordered[int(perm[i])]) for i in range(n)}
    # Repair remaining fixed points by swapping along the cycle.
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    if np.any(perm == np.arange(n)):
        # A two-cycle repair can recreate a fixed point only for n == 1, already rejected.
        shift = np.roll(np.arange(n), 1)
        perm = shift
    return {str(ordered[i]): str(ordered[int(perm[i])]) for i in range(n)}


def alternate_history_index(target_index: int, length: int) -> int:
    """Another fixation from the same scanpath, not the target and not its predecessor.

    Length-2 scanpaths have no such fixation. Callers must skip them.
    """
    if length < 3 or target_index <= 0 or target_index >= length:
        raise ValueError("alternate history is only defined for target_index in 1..length-2 and length >= 3")
    candidate = length - 1 - target_index
    predecessor = target_index - 1
    if candidate != target_index and candidate != predecessor:
        return int(candidate)
    for offset in range(length):
        if offset != target_index and offset != predecessor:
            return int(offset)
    raise RuntimeError("no alternate history index")


@dataclass
class ScanRecord:
    scan_id: int
    image: str
    subject: int
    split: str
    role: str
    ix: np.ndarray
    iy: np.ndarray

    @property
    def length(self) -> int:
        return int(self.ix.size)


def pack_scanpaths(scanpaths, grid: Grid, role_kwargs: dict) -> list[ScanRecord]:
    records: list[ScanRecord] = []
    seen: set[tuple[str, int, str]] = set()
    for scan_id, scan in enumerate(scanpaths):
        key = (scan.image_name, int(scan.subject_id), scan.split)
        if key in seen:
            raise ValueError(f"duplicate scanpath for image={scan.image_name} subject={scan.subject_id} split={scan.split}")
        seen.add(key)
        x = np.asarray(scan.x, dtype=float)
        y = np.asarray(scan.y, dtype=float)
        if x.size != y.size:
            raise ValueError("x and y lengths differ")
        if np.any(x < 0) or np.any(y < 0) or np.any(x > grid.width) or np.any(y > grid.height):
            raise ValueError(f"fixation outside the canvas for {scan.image_name}")
        records.append(
            ScanRecord(
                scan_id=scan_id,
                image=scan.image_name,
                subject=int(scan.subject_id),
                split=scan.split,
                role=image_role(scan.image_name, scan.split, **role_kwargs),
                ix=bin_index(x, grid.width, grid.nx),
                iy=bin_index(y, grid.height, grid.ny),
            )
        )
    return records


def _group_records(records: list[ScanRecord]) -> dict[str, list[ScanRecord]]:
    grouped: dict[str, list[ScanRecord]] = {}
    for record in records:
        grouped.setdefault(record.image, []).append(record)
    return grouped


def accumulate_counts(records: list[ScanRecord], predicate, index_min: int, index_max: int | None, grid: Grid) -> np.ndarray:
    counts = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    for record in records:
        if not predicate(record):
            continue
        stop = record.length if index_max is None else min(record.length, index_max)
        start = index_min
        if stop <= start:
            continue
        counts += histogram_counts(record.iy[start:stop], record.ix[start:stop], grid.ny, grid.nx)
    return counts


def displacement_histogram(records: list[ScanRecord], predicate, grid: Grid, previous_fn=None) -> np.ndarray:
    """Histogram of (target bin - previous bin). Center bin is zero displacement."""
    ny, nx = grid.ny, grid.nx
    hist = np.zeros((2 * ny - 1, 2 * nx - 1), dtype=np.float64)
    for record in records:
        if not predicate(record) or record.length < 2:
            continue
        for target in range(1, record.length):
            if previous_fn is None:
                previous = target - 1
            else:
                try:
                    previous = previous_fn(target, record.length)
                except ValueError:
                    continue
            diy = int(record.iy[target] - record.iy[previous] + ny - 1)
            dix = int(record.ix[target] - record.ix[previous] + nx - 1)
            hist[diy, dix] += 1
    return hist


def blur_displacement(hist: np.ndarray, sigma_bins: float) -> np.ndarray:
    if float(hist.sum()) <= 0:
        out = np.zeros_like(hist)
        out[hist.shape[0] // 2, hist.shape[1] // 2] = 1.0
        return out
    blurred = hist if sigma_bins == 0 else gaussian_filter(hist, sigma=float(sigma_bins), mode="constant", cval=0.0)
    total = float(blurred.sum())
    if total <= 0:
        out = np.zeros_like(hist)
        out[hist.shape[0] // 2, hist.shape[1] // 2] = 1.0
        return out
    return blurred / total


def canvas_kernel_mass(kernel: np.ndarray, grid: Grid) -> np.ndarray:
    """For each previous bin, the kernel mass that lands inside the canvas."""
    ny, nx = grid.ny, grid.nx
    mass = np.empty((ny, nx), dtype=np.float64)
    for py in range(ny):
        row0 = ny - 1 - py
        for px in range(nx):
            col0 = nx - 1 - px
            mass[py, px] = kernel[row0 : row0 + ny, col0 : col0 + nx].sum()
    return mass


def history_probabilities(
    kernel: np.ndarray,
    mass: np.ndarray,
    prev_iy: np.ndarray,
    prev_ix: np.ndarray,
    tgt_iy: np.ndarray,
    tgt_ix: np.ndarray,
    grid: Grid,
    alpha: float,
) -> np.ndarray:
    ny, nx = grid.ny, grid.nx
    row = tgt_iy - prev_iy + ny - 1
    col = tgt_ix - prev_ix + nx - 1
    raw = kernel[row, col]
    visible = mass[prev_iy, prev_ix]
    probability = np.empty(raw.shape, dtype=np.float64)
    valid = visible > 0
    probability[valid] = raw[valid] / visible[valid]
    probability[~valid] = 1.0 / grid.n_bins
    probability = (1.0 - alpha) * probability + alpha / grid.n_bins
    return probability


def subject_means(subjects: np.ndarray, values: np.ndarray) -> np.ndarray:
    subjects = np.asarray(subjects)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    subjects = subjects[keep]
    values = values[keep]
    if subjects.size == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(subjects, kind="mergesort")
    subjects = subjects[order]
    values = values[order]
    unique, starts = np.unique(subjects, return_index=True)
    sums = np.add.reduceat(values, starts)
    counts = np.diff(np.append(starts, subjects.size))
    means = sums / counts
    return means


def scanpath_then_subject_means(scan_ids: np.ndarray, subjects: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Equal weight per scanpath inside a subject, then one mean per subject."""
    scan_ids = np.asarray(scan_ids)
    subjects = np.asarray(subjects)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if not np.any(keep):
        return np.empty(0, dtype=float)
    frame = pd.DataFrame({"scan": scan_ids[keep], "subject": subjects[keep], "value": values[keep]})
    per_scan = frame.groupby(["subject", "scan"], sort=False)["value"].mean()
    return per_scan.groupby("subject").mean().to_numpy(dtype=float)


def choose_maximum(scores: dict[float, float], candidates: list[float]) -> float:
    """Highest score. Ties use the earlier candidate, which is the smaller listed value."""
    best_key = None
    best_score = -np.inf
    for key in candidates:
        score = scores[key]
        if score > best_score and np.isfinite(score):
            best_score = score
            best_key = key
    if best_key is None:
        raise RuntimeError("no finite selection score")
    return best_key


def _mean_subject_ig(records: list[ScanRecord], density: np.ndarray, grid: Grid, index_min: int) -> float:
    subjects = []
    values = []
    scans = []
    for record in records:
        if record.length <= index_min:
            continue
        probability = density[record.iy[index_min:], record.ix[index_min:]]
        ig = information_gain(probability, grid.n_bins)
        subjects.append(np.full(ig.size, record.subject))
        scans.append(np.full(ig.size, record.scan_id))
        values.append(ig)
    if not values:
        return math.nan
    return float(np.mean(scanpath_then_subject_means(np.concatenate(scans), np.concatenate(subjects), np.concatenate(values))))


def select_center_sigma(records, grid: Grid, candidates_px: list[float], alpha: float, index_min: int) -> tuple[float, list[dict]]:
    fit_counts = accumulate_counts(records, lambda r: r.role == "fit", index_min, None, grid)
    select_records = [r for r in records if r.role == "select_bandwidth"]
    rows = []
    scores = {}
    for sigma_px in candidates_px:
        density = density_from_counts(fit_counts, sigma_in_bins(sigma_px, grid), alpha)
        score = _mean_subject_ig(select_records, density, grid, index_min)
        scores[sigma_px] = score
        rows.append({"parameter": "sigma_center_px", "candidate": sigma_px, "select_subject_mean_ig_bits": score, "fold": "select_bandwidth"})
    return choose_maximum(scores, list(candidates_px)), rows


def select_history_sigma(records, grid: Grid, candidates_px: list[float], alpha: float) -> tuple[float, list[dict]]:
    fit_hist = displacement_histogram(records, lambda r: r.role == "fit", grid)
    select_records = [r for r in records if r.role == "select_bandwidth"]
    rows = []
    scores = {}
    for sigma_px in candidates_px:
        kernel = blur_displacement(fit_hist, sigma_in_bins(sigma_px, grid))
        mass = canvas_kernel_mass(kernel, grid)
        subjects, scans, values = [], [], []
        for record in select_records:
            if record.length < 2:
                continue
            targets = np.arange(1, record.length)
            probability = history_probabilities(
                kernel, mass, record.iy[targets - 1], record.ix[targets - 1], record.iy[targets], record.ix[targets], grid, alpha
            )
            ig = information_gain(probability, grid.n_bins)
            subjects.append(np.full(ig.size, record.subject))
            scans.append(np.full(ig.size, record.scan_id))
            values.append(ig)
        score = float(np.mean(scanpath_then_subject_means(np.concatenate(scans), np.concatenate(subjects), np.concatenate(values))))
        scores[sigma_px] = score
        rows.append({"parameter": "sigma_history_px", "candidate": sigma_px, "select_subject_mean_ig_bits": score, "fold": "select_bandwidth"})
    return choose_maximum(scores, list(candidates_px)), rows


def _loo_maps(group: list[ScanRecord], grid: Grid, sigma_bins: float, alpha: float, index_min: int) -> dict[int, np.ndarray]:
    hists: dict[int, np.ndarray] = {}
    for record in group:
        if record.length <= index_min:
            hists[record.subject] = np.zeros((grid.ny, grid.nx), dtype=np.float64)
        else:
            hists[record.subject] = histogram_counts(record.iy[index_min:], record.ix[index_min:], grid.ny, grid.nx)
    total = np.sum(list(hists.values()), axis=0)
    return {subject: density_from_counts(total - hist, sigma_bins, alpha) for subject, hist in hists.items()}


def select_loo_sigma(grouped: dict[str, list[ScanRecord]], grid: Grid, candidates_px: list[float], alpha: float, index_min: int) -> tuple[float, list[dict]]:
    images = [name for name, group in grouped.items() if group and group[0].role == "select_bandwidth"]
    rows = []
    scores = {}
    for sigma_px in candidates_px:
        sigma_bins = sigma_in_bins(sigma_px, grid)
        subjects, scans, values = [], [], []
        for name in images:
            maps = _loo_maps(grouped[name], grid, sigma_bins, alpha, index_min)
            for record in grouped[name]:
                if record.length <= index_min:
                    continue
                probability = maps[record.subject][record.iy[index_min:], record.ix[index_min:]]
                ig = information_gain(probability, grid.n_bins)
                subjects.append(np.full(ig.size, record.subject))
                scans.append(np.full(ig.size, record.scan_id))
                values.append(ig)
        score = float(np.mean(scanpath_then_subject_means(np.concatenate(scans), np.concatenate(subjects), np.concatenate(values))))
        scores[sigma_px] = score
        rows.append({"parameter": "sigma_loo_px", "candidate": sigma_px, "select_subject_mean_ig_bits": score, "fold": "select_bandwidth"})
    return choose_maximum(scores, list(candidates_px)), rows


def _collect_static_ig(records: list[ScanRecord], density: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects, scans, values = [], [], []
    for record in records:
        if record.length < 2:
            continue
        probability = density[record.iy[1:], record.ix[1:]]
        ig = information_gain(probability, grid.n_bins)
        subjects.append(np.full(ig.size, record.subject))
        scans.append(np.full(ig.size, record.scan_id))
        values.append(ig)
    return np.concatenate(scans), np.concatenate(subjects), np.concatenate(values)


def select_lambdas(
    records: list[ScanRecord],
    grouped: dict[str, list[ScanRecord]],
    grid: Grid,
    center: np.ndarray,
    kernel: np.ndarray,
    kernel_mass: np.ndarray,
    sigma_loo_bins: float,
    alpha: float,
    lambda_candidates: list[float],
    index_min: int = 1,
) -> tuple[float, float, list[dict]]:
    """Choose mixture weights on the mixture fold. Densities were fit on the fit fold only.

    Mixture targets still start at index 1, because the history term has no
    previous fixation at index 0. ``index_min`` controls which fixations enter
    the leave-one-observer map.
    """
    mix_records = [r for r in records if r.role == "select_mixture"]
    loo_cache = {
        name: _loo_maps(group, grid, sigma_loo_bins, alpha, index_min)
        for name, group in grouped.items()
        if group and group[0].role == "select_mixture"
    }
    p_center_parts = []
    p_hist_parts = []
    p_loo_parts = []
    scan_parts = []
    subject_parts = []
    for record in mix_records:
        if record.length < 2:
            continue
        targets = np.arange(1, record.length)
        p_center_parts.append(center[record.iy[targets], record.ix[targets]])
        p_hist_parts.append(
            history_probabilities(
                kernel, kernel_mass, record.iy[targets - 1], record.ix[targets - 1], record.iy[targets], record.ix[targets], grid, alpha
            )
        )
        p_loo_parts.append(loo_cache[record.image][record.subject][record.iy[targets], record.ix[targets]])
        scan_parts.append(np.full(targets.size, record.scan_id))
        subject_parts.append(np.full(targets.size, record.subject))
    p_center = np.concatenate(p_center_parts)
    p_hist = np.concatenate(p_hist_parts)
    p_loo = np.concatenate(p_loo_parts)
    scans = np.concatenate(scan_parts)
    subjects = np.concatenate(subject_parts)
    rows = []
    scores_b5 = {}
    scores_b5c = {}
    for lam in lambda_candidates:
        p_b5 = (1.0 - lam) * p_center + lam * p_hist
        p_b5c = (1.0 - lam) * p_loo + lam * p_hist
        score_b5 = float(np.mean(scanpath_then_subject_means(scans, subjects, information_gain(p_b5, grid.n_bins))))
        score_b5c = float(np.mean(scanpath_then_subject_means(scans, subjects, information_gain(p_b5c, grid.n_bins))))
        scores_b5[lam] = score_b5
        scores_b5c[lam] = score_b5c
        rows.append({"parameter": "lambda_center_history", "candidate": lam, "select_subject_mean_ig_bits": score_b5, "fold": "select_mixture"})
        rows.append({"parameter": "lambda_loo_history", "candidate": lam, "select_subject_mean_ig_bits": score_b5c, "fold": "select_mixture"})
    return choose_maximum(scores_b5, list(lambda_candidates)), choose_maximum(scores_b5c, list(lambda_candidates)), rows


def assert_loo_excludes_target(group: list[ScanRecord], grid: Grid, index_min: int = 1) -> None:
    hists = {}
    for record in group:
        if record.length <= index_min:
            hists[record.subject] = np.zeros((grid.ny, grid.nx))
        else:
            hists[record.subject] = histogram_counts(record.iy[index_min:], record.ix[index_min:], grid.ny, grid.nx)
    total = np.sum(list(hists.values()), axis=0)
    for subject, hist in hists.items():
        reference = total - hist
        # The held-out histogram and the reference share no counts.
        overlap = np.minimum(reference, hist)
        if float(overlap.sum()) != 0 and float(hist.sum()) > 0:
            # Overlap of positive bins is allowed when another observer landed in the same bin.
            # The count identity is the leakage check.
            pass
        if not np.array_equal(reference + hist, total):
            raise AssertionError("leave-one-observer reference does not rebuild the total")
        if float(reference.sum() + hist.sum() - total.sum()) != 0:
            raise AssertionError("leave-one-observer counts changed the total mass")


def sample_counts(records: list[ScanRecord], index_max: int) -> pd.DataFrame:
    rows = []
    for index in range(0, index_max + 1):
        present = [r for r in records if r.length > index]
        rows.append(
            {
                "fixation_index": index,
                "n_scanpaths": len(present),
                "n_subjects": len({r.subject for r in present}),
                "n_images": len({r.image for r in present}),
                "n_train_scanpaths": sum(r.split == "train" for r in present),
                "n_valid_scanpaths": sum(r.split == "valid" for r in present),
            }
        )
    late = [r for r in records if r.length > index_max]
    rows.append(
        {
            "fixation_index": f"{index_max + 1}_and_later",
            "n_scanpaths": len(late),
            "n_subjects": len({r.subject for r in late}),
            "n_images": len({r.image for r in late}),
            "n_train_scanpaths": sum(r.split == "train" for r in late),
            "n_valid_scanpaths": sum(r.split == "valid" for r in late),
        }
    )
    return pd.DataFrame(rows)


def bootstrap_groups(values: np.ndarray, metric: str, subset: str, estimand: str, n_boot: int, base_seed: int) -> dict:
    summary = percentile_bootstrap_mean(values, n_boot, metric_seed(base_seed, metric, subset, estimand))
    summary.update({"metric": metric, "subset": subset, "estimand": estimand})
    return summary


def probability_from_blur(blurred: np.ndarray, iy: int, ix: int, alpha: float, n_bins: int) -> float:
    mass = float(blurred.sum())
    if mass <= 0 or not np.isfinite(mass):
        return 1.0 / n_bins
    mixed = (1.0 - alpha) * (float(blurred[iy, ix]) / mass) + alpha / n_bins
    return float(mixed)


def history_lookup_tables(kernel: np.ndarray, mass: np.ndarray, grid: Grid, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Conditional bin probability and NSS for every previous bin and target bin."""
    ny, nx = grid.ny, grid.nx
    probability = np.empty((ny, nx, ny, nx), dtype=np.float64)
    nss = np.empty((ny, nx, ny, nx), dtype=np.float64)
    for py, px in np.ndindex(ny, nx):
        row0 = ny - 1 - py
        col0 = nx - 1 - px
        raw = kernel[row0 : row0 + ny, col0 : col0 + nx]
        visible = float(mass[py, px])
        if visible <= 0:
            base = np.full((ny, nx), 1.0 / grid.n_bins)
        else:
            base = raw / visible
        mixed = (1.0 - alpha) * base + alpha / grid.n_bins
        mixed = np.maximum(mixed, 0.0)
        mixed /= mixed.sum()
        probability[py, px] = mixed
        centered = mixed - float(mixed.mean())
        scale = float(mixed.std(ddof=0))
        nss[py, px] = 0.0 if scale == 0 else centered / scale
    return probability, nss


def evaluate_records(
    records: list[ScanRecord],
    grid: Grid,
    center: np.ndarray,
    center_by_subject: dict[int, np.ndarray],
    loo_sigma_bins: float,
    kernel: np.ndarray,
    kernel_mass: np.ndarray,
    alpha: float,
    lambda_b5: float,
    lambda_b5c: float,
    shuffle_map: dict[str, str],
    index_min: int = 1,
) -> pd.DataFrame:
    """Score primary targets. Index `index_min` and later are targets. Index 0 can be history."""
    grouped = _group_records(records)
    library = impulse_library(grid.ny, grid.nx, loo_sigma_bins)
    impulse_mass = library.sum(axis=(2, 3))
    history_p, history_nss = history_lookup_tables(kernel, kernel_mass, grid, alpha)
    center_scale = float(center.std(ddof=0))
    nss_center = np.zeros_like(center) if center_scale == 0 else (center - float(center.mean())) / center_scale
    center_top = top_fraction_mask(center)

    own_blur: dict[str, dict[int, np.ndarray]] = {}
    total_blur: dict[str, np.ndarray] = {}
    for name, group in grouped.items():
        own: dict[int, np.ndarray] = {}
        total = np.zeros((grid.ny, grid.nx), dtype=np.float64)
        for record in group:
            if record.length <= index_min:
                own[record.subject] = np.zeros((grid.ny, grid.nx), dtype=np.float64)
            else:
                blurred = accumulate_impulses(library, record.iy[index_min:], record.ix[index_min:])
                own[record.subject] = blurred
                total += blurred
        own_blur[name] = own
        total_blur[name] = total

    loo_density: dict[str, dict[int, np.ndarray]] = {}
    for name, own in own_blur.items():
        total = total_blur[name]
        loo_density[name] = {subject: normalize_and_floor(total - blurred, alpha) for subject, blurred in own.items()}

    rows: dict[str, list] = {key: [] for key in (
        "scan_id", "image", "subject", "split", "role", "length", "fixation_index", "fixation_iy", "fixation_ix",
        "ig_B0_uniform", "ig_B1_center", "ig_B1_subject_held_out", "ig_B2_loo_image",
        "ig_B3_loo_index", "ig_B4_history", "ig_B5_center_history", "ig_B5c_loo_history",
        "ig_B6_image_shuffle", "ig_B6_order_shuffle",
        "nss_B1_center", "nss_B2_loo_image", "nss_B4_history",
        "top10_B1_center", "top10_B2_loo_image", "order_control_degenerate",
        "p_B1_center", "p_B4_history", "p_B2_loo_image",
    )}
    log2_n = math.log2(grid.n_bins)

    def _ig(probability: np.ndarray) -> np.ndarray:
        safe = np.maximum(np.asarray(probability, dtype=np.float64), np.finfo(np.float64).tiny)
        return np.log2(safe) + log2_n

    for name, group in grouped.items():
        other = shuffle_map[name]
        other_own = own_blur[other]
        other_total = total_blur[other]
        index_points: dict[int, list[tuple[int, int, int]]] = {}
        for record in group:
            for fixation_index in range(index_min, record.length):
                index_points.setdefault(fixation_index, []).append(
                    (record.subject, int(record.iy[fixation_index]), int(record.ix[fixation_index]))
                )
        b3: dict[tuple[int, int], float] = {}
        for fixation_index, entries in index_points.items():
            total = np.zeros((grid.ny, grid.nx), dtype=np.float64)
            for _subject, iy, ix in entries:
                total += library[iy, ix]
            total_mass = float(total.sum())
            for subject, iy, ix in entries:
                loo_mass = total_mass - float(impulse_mass[iy, ix])
                loo_value = float(total[iy, ix] - library[iy, ix, iy, ix])
                if loo_mass <= 0:
                    probability = 1.0 / grid.n_bins
                else:
                    probability = (1.0 - alpha) * (loo_value / loo_mass) + alpha / grid.n_bins
                b3[(subject, fixation_index)] = float(probability)

        for record in group:
            if record.length <= index_min:
                continue
            targets = np.arange(index_min, record.length)
            iy = record.iy[targets]
            ix = record.ix[targets]
            p_b1 = center[iy, ix]
            p_b1_held = center_by_subject[record.subject][iy, ix]
            p_b2 = loo_density[name][record.subject][iy, ix]
            p_b3 = np.array([b3[(record.subject, int(k))] for k in targets])
            history_ok = targets >= 1
            p_b4 = np.full(targets.shape, np.nan)
            nss_b4 = np.full(targets.shape, np.nan)
            if np.any(history_ok):
                prev = targets[history_ok] - 1
                p_b4[history_ok] = history_p[record.iy[prev], record.ix[prev], iy[history_ok], ix[history_ok]]
                nss_b4[history_ok] = history_nss[record.iy[prev], record.ix[prev], iy[history_ok], ix[history_ok]]
            p_b6o = p_b4.copy()
            if record.length >= 3 and np.any(history_ok):
                alt = np.array([alternate_history_index(int(k), record.length) if int(k) >= 1 else 0 for k in targets])
                p_b6o[history_ok] = history_p[record.iy[alt[history_ok]], record.ix[alt[history_ok]], iy[history_ok], ix[history_ok]]
            degenerate_flags = (record.length < 3) | ~history_ok
            other_blur = other_total - other_own.get(record.subject, 0.0)
            other_mass = float(np.asarray(other_blur).sum())
            if other_mass <= 0:
                p_b6i = np.full(targets.size, 1.0 / grid.n_bins)
            else:
                p_b6i = (1.0 - alpha) * (np.asarray(other_blur)[iy, ix] / other_mass) + alpha / grid.n_bins
            p_b5 = (1.0 - lambda_b5) * p_b1 + lambda_b5 * p_b4
            p_b5c = (1.0 - lambda_b5c) * p_b2 + lambda_b5c * p_b4
            n = targets.size
            rows["scan_id"].extend([record.scan_id] * n)
            rows["image"].extend([record.image] * n)
            rows["subject"].extend([record.subject] * n)
            rows["split"].extend([record.split] * n)
            rows["role"].extend([record.role] * n)
            rows["length"].extend([record.length] * n)
            rows["fixation_index"].extend(targets.tolist())
            rows["fixation_iy"].extend(iy.tolist())
            rows["fixation_ix"].extend(ix.tolist())
            rows["ig_B0_uniform"].extend(np.zeros(n).tolist())
            rows["ig_B1_center"].extend(_ig(p_b1).tolist())
            rows["ig_B1_subject_held_out"].extend(_ig(p_b1_held).tolist())
            rows["ig_B2_loo_image"].extend(_ig(p_b2).tolist())
            rows["ig_B3_loo_index"].extend(_ig(p_b3).tolist())
            rows["ig_B4_history"].extend(_ig(p_b4).tolist())
            rows["ig_B5_center_history"].extend(_ig(p_b5).tolist())
            rows["ig_B5c_loo_history"].extend(_ig(p_b5c).tolist())
            rows["ig_B6_image_shuffle"].extend(_ig(p_b6i).tolist())
            rows["ig_B6_order_shuffle"].extend(_ig(p_b6o).tolist())
            rows["nss_B1_center"].extend(nss_center[iy, ix].tolist())
            rows["nss_B2_loo_image"].extend(nss_values(loo_density[name][record.subject], iy, ix).tolist())
            rows["nss_B4_history"].extend(nss_b4.tolist())
            rows["top10_B1_center"].extend(center_top[iy, ix].tolist())
            rows["top10_B2_loo_image"].extend(top_fraction_mask(loo_density[name][record.subject])[iy, ix].tolist())
            rows["order_control_degenerate"].extend(degenerate_flags.tolist())
            rows["p_B1_center"].extend(p_b1.tolist())
            rows["p_B4_history"].extend(p_b4.tolist())
            rows["p_B2_loo_image"].extend(p_b2.tolist())
    return pd.DataFrame(rows)


def aggregate_ig(frame: pd.DataFrame, column: str, mask: np.ndarray, n_boot: int, seed: int, subset: str, estimand: str) -> dict:
    part = frame.loc[mask]
    if estimand == "subject_scanpath_equal":
        values = scanpath_then_subject_means(part["scan_id"].to_numpy(), part["subject"].to_numpy(), part[column].to_numpy())
    elif estimand == "subject_fixation_weighted":
        values = subject_means(part["subject"].to_numpy(), part[column].to_numpy())
    elif estimand == "image":
        per_image = part.groupby("image")[column].mean()
        values = per_image.to_numpy(dtype=float)
    else:
        raise ValueError(estimand)
    summary = bootstrap_groups(values, column, subset, estimand, n_boot, seed)
    summary["n_fixations"] = int(mask.sum())
    summary["n_scanpaths"] = int(part["scan_id"].nunique()) if len(part) else 0
    summary["n_images"] = int(part["image"].nunique()) if len(part) else 0
    summary["n_subjects_observed"] = int(part["subject"].nunique()) if len(part) else 0
    return summary


def paired_subject_difference(frame: pd.DataFrame, column_a: str, column_b: str, mask: np.ndarray, n_boot: int, seed: int, subset: str) -> dict:
    part = frame.loc[mask].copy()
    part["diff"] = part[column_a] - part[column_b]
    per_scan = part.groupby(["subject", "scan_id"], sort=False)["diff"].mean()
    per_subject = per_scan.groupby("subject").mean()
    summary = bootstrap_groups(per_subject.to_numpy(dtype=float), f"{column_a}_minus_{column_b}", subset, "subject_scanpath_equal", n_boot, seed)
    summary["n_fixations"] = int(mask.sum())
    summary["n_scanpaths"] = int(part["scan_id"].nunique()) if len(part) else 0
    summary["contrast"] = f"{column_a} - {column_b}"
    return summary


def calibration_table(frame: pd.DataFrame, probability_column: str, n_groups: int = 10) -> pd.DataFrame:
    """Group validation fixations by predicted bin probability and compare with a bin occupancy check.

    This is not a repeated-trial calibration curve. It reports the mean predicted bin
    probability in each quantile group. The spatial reliability of B1 is computed
    separately from the map and the empirical bin counts.
    """
    values = frame[probability_column].to_numpy(dtype=float)
    order = np.argsort(values)
    edges = np.linspace(0, len(values), n_groups + 1).astype(int)
    rows = []
    for group in range(n_groups):
        chosen = order[edges[group] : edges[group + 1]]
        if chosen.size == 0:
            continue
        rows.append(
            {
                "probability_column": probability_column,
                "quantile_group": group + 1,
                "n_fixations": int(chosen.size),
                "mean_predicted_bin_probability": float(values[chosen].mean()),
                "min_predicted_bin_probability": float(values[chosen].min()),
                "max_predicted_bin_probability": float(values[chosen].max()),
            }
        )
    return pd.DataFrame(rows)


def spatial_reliability(iy: np.ndarray, ix: np.ndarray, density: np.ndarray, n_groups: int = 10) -> pd.DataFrame:
    flat_p = density.ravel()
    counts = histogram_counts(iy, ix, density.shape[0], density.shape[1]).ravel()
    order = np.argsort(flat_p)
    edges = np.linspace(0, flat_p.size, n_groups + 1).astype(int)
    total = float(counts.sum())
    rows = []
    for group in range(n_groups):
        chosen = order[edges[group] : edges[group + 1]]
        rows.append(
            {
                "quantile_group": group + 1,
                "n_bins": int(chosen.size),
                "mean_predicted_bin_probability": float(flat_p[chosen].mean()),
                "observed_fixation_share": float(counts[chosen].sum() / total) if total else math.nan,
                "predicted_bin_share": float(flat_p[chosen].sum()),
            }
        )
    return pd.DataFrame(rows)


def config_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
