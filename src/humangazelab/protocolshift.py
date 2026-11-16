"""Exploratory controls for the index-conditioned density.

These functions do not write into outputs/temporal_v1, outputs/osie_v1,
outputs/audit_checkpoint, or outputs/temporal_v2. Coordinate scaling here
is a fraction of the released canvas. It is not a conversion to degrees
of visual angle.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from humangazelab.adapter import Scanpath
from humangazelab.temporal import (
    Grid,
    _group_records,
    blur_displacement,
    canvas_kernel_mass,
    displacement_histogram,
    history_probabilities,
    impulse_library,
    information_gain,
    normalize_and_floor,
    scanpath_then_subject_means,
    sigma_in_bins,
)

FROZEN_OUTPUTS = (
    "outputs/temporal_v1",
    "outputs/osie_v1",
    "outputs/audit_checkpoint",
    "outputs/temporal_v2",
    "outputs/figures",
    "outputs/tables",
)


def assert_new_output(root, output_dir) -> None:
    resolved = output_dir.resolve()
    for relative in FROZEN_OUTPUTS:
        if resolved == (root / relative).resolve():
            raise RuntimeError(f"refusing to write protocol-shift results into {relative}")


def scale_scanpaths(scanpaths: list[Scanpath], width: float, height: float) -> list[Scanpath]:
    """Map released pixels onto the unit square. This is not a visual-angle map."""
    if width <= 0 or height <= 0:
        raise ValueError("canvas size must be positive")
    scaled = []
    for scan in scanpaths:
        x = np.asarray(scan.x, dtype=float) / width
        y = np.asarray(scan.y, dtype=float) / height
        if np.any(x < 0) or np.any(y < 0) or np.any(x > 1) or np.any(y > 1):
            raise ValueError(f"normalized coordinate left the unit square for {scan.image_name}")
        scaled.append(replace(scan, x=x, y=y))
    return scaled


def keep_subjects(scanpaths: list[Scanpath], subjects: set[int]) -> list[Scanpath]:
    return [scan for scan in scanpaths if int(scan.subject_id) in subjects]


def seeded_subject_subsets(subjects: list[int], size: int, n_subsets: int, seed: int) -> list[tuple[int, ...]]:
    """Distinct subsets. The seed fixes the draw; it is not a search over outcomes."""
    if size > len(subjects) or n_subsets < 1:
        raise ValueError("subset request is larger than the subject list or empty")
    rng = np.random.default_rng(seed)
    pool = np.array(sorted(subjects))
    chosen: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(100000):
        pick = tuple(sorted(int(v) for v in rng.choice(pool, size=size, replace=False)))
        if pick not in seen:
            seen.add(pick)
            chosen.append(pick)
        if len(chosen) == n_subsets:
            return chosen
    raise RuntimeError("could not draw the requested number of distinct subsets")


_LIBRARY_CACHE: dict[tuple[int, int, float], np.ndarray] = {}


def _cached_library(ny: int, nx: int, sigma_bins: float) -> np.ndarray:
    key = (ny, nx, round(float(sigma_bins), 8))
    if key not in _LIBRARY_CACHE:
        _LIBRARY_CACHE[key] = impulse_library(ny, nx, sigma_bins)
    return _LIBRARY_CACHE[key]


def score_index_density(
    records,
    grid: Grid,
    sigma_bins: float,
    alpha: float,
    index_min: int,
    index_max: int | None = None,
    with_map_diagnostics: bool = False,
) -> pd.DataFrame:
    """Leave-one-observer density at one released index.

    The predictive entropy and the highest-density region describe the map
    that was scored. They are properties of that discrete distribution, not
    a confidence interval for a continuous density.
    """
    library = _cached_library(grid.ny, grid.nx, sigma_bins)
    grouped = _group_records(records)
    rows: dict[str, list] = {key: [] for key in (
        "scan_id", "image", "subject", "split", "length", "fixation_index",
        "ig", "n_other", "entropy_bits", "in_half_mass_region",
    )}
    n_bins = grid.n_bins
    for image, group in grouped.items():
        points: dict[int, list[tuple]] = {}
        for record in group:
            stop = record.length if index_max is None else min(record.length, index_max + 1)
            for index in range(index_min, stop):
                points.setdefault(index, []).append((record, int(record.iy[index]), int(record.ix[index])))
        for index, entries in points.items():
            total = np.zeros((grid.ny, grid.nx), dtype=np.float64)
            for _record, iy, ix in entries:
                total += library[iy, ix]
            for record, iy, ix in entries:
                density = normalize_and_floor(total - library[iy, ix], alpha)
                probability = float(density[iy, ix])
                rows["scan_id"].append(record.scan_id)
                rows["image"].append(image)
                rows["subject"].append(record.subject)
                rows["split"].append(record.split)
                rows["length"].append(record.length)
                rows["fixation_index"].append(index)
                rows["ig"].append(float(information_gain(np.array([probability]), n_bins)[0]))
                rows["n_other"].append(len(entries) - 1)
                if with_map_diagnostics:
                    entropy = float(-np.sum(density * np.log2(density)))
                    rows["entropy_bits"].append(entropy)
                    rows["in_half_mass_region"].append(_in_highest_mass(density, iy, ix, 0.5))
                else:
                    rows["entropy_bits"].append(math.nan)
                    rows["in_half_mass_region"].append(math.nan)
    return pd.DataFrame(rows)


def _in_highest_mass(density: np.ndarray, iy: int, ix: int, mass: float) -> float:
    flat = density.ravel()
    if np.max(flat) - np.min(flat) <= 1e-12:
        return math.nan
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    keep = int(np.searchsorted(cumulative, mass, side="left") + 1)
    target = iy * density.shape[1] + ix
    return float(target in set(order[:keep].tolist()))


def subject_equal_ig(frame: pd.DataFrame, valid_name: str = "valid") -> dict:
    part = frame[frame["split"].eq(valid_name)]
    values = scanpath_then_subject_means(part["scan_id"].to_numpy(), part["subject"].to_numpy(), part["ig"].to_numpy())
    return {
        "estimate": float(np.mean(values)) if len(values) else math.nan,
        "n_subjects": int(len(values)),
        "n_fixations": int(np.isfinite(part["ig"].to_numpy()).sum()),
        "n_scanpaths": int(part["scan_id"].nunique()) if len(part) else 0,
        "n_images": int(part["image"].nunique()) if len(part) else 0,
        "subject_sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
    }


def image_equal_ig(frame: pd.DataFrame, valid_name: str = "valid") -> dict:
    part = frame[frame["split"].eq(valid_name)]
    if part.empty:
        return {"estimate": math.nan, "image_sd": math.nan, "n_images": 0}
    per_image = part.groupby("image")["ig"].mean()
    values = per_image.to_numpy(dtype=float)
    return {
        "estimate": float(np.mean(values)),
        "image_sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
        "n_images": int(len(values)),
    }


def confident_but_poor(frame: pd.DataFrame, low_quantile: float, poor_below: float, valid_name: str = "valid") -> dict:
    """Low predictive entropy together with negative information gain.

    The entropy quantile is computed on the same rows. It describes those
    rows. It is not a threshold selected on a separate confirmation set.
    """
    part = frame[frame["split"].eq(valid_name)].dropna(subset=["entropy_bits", "ig"])
    entropy = part["entropy_bits"].to_numpy(dtype=float)
    ig = part["ig"].to_numpy(dtype=float)
    if len(part) == 0:
        return {"n": 0}
    cutoff = float(np.quantile(entropy, low_quantile))
    mask = (entropy <= cutoff) & (ig < poor_below)
    return {
        "n_fixations": int(len(part)),
        "entropy_cutoff_bits": cutoff,
        "n_confident_and_poor": int(mask.sum()),
        "fraction_confident_and_poor": float(mask.mean()),
        "mean_ig_confident_and_poor": float(np.mean(ig[mask])) if mask.any() else math.nan,
        "mean_ig_remaining": float(np.mean(ig[~mask])) if (~mask).any() else math.nan,
        "fraction_inside_half_mass": float(np.nanmean(part["in_half_mass_region"].to_numpy(dtype=float))),
        "mean_entropy_bits": float(np.mean(entropy)),
    }


def observer_spread(scanpaths, index_min: int, index_max: int, width: float, height: float, valid_name: str) -> pd.DataFrame:
    """Spread of human fixations at one index. This is not a model uncertainty."""
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for scan in scanpaths:
        if scan.split != valid_name:
            continue
        stop = min(len(scan.x), index_max + 1)
        for index in range(index_min, stop):
            grouped.setdefault((scan.image_name, index), []).append((float(scan.x[index]), float(scan.y[index])))
    rows = []
    for (image, index), points in grouped.items():
        if len(points) < 2:
            continue
        xs = np.array([point[0] for point in points])
        ys = np.array([point[1] for point in points])
        rows.append({
            "image": image,
            "fixation_index": index,
            "n_observers": len(points),
            "sd_x_px": float(xs.std(ddof=1)),
            "sd_y_px": float(ys.std(ddof=1)),
            "sd_x_fraction_of_width": float((xs / width).std(ddof=1)),
            "sd_y_fraction_of_height": float((ys / height).std(ddof=1)),
        })
    return pd.DataFrame(rows)


def history_transfer_ig(source_records, target_records, grid: Grid, sigma_bins: float, alpha: float) -> dict:
    """Score a history kernel fit on source train images, on target validation images.

    Target validation fixations are not used to fit the kernel.
    """
    kernel = blur_displacement(
        displacement_histogram(source_records, lambda record: record.split == "train", grid),
        sigma_bins,
    )
    mass = canvas_kernel_mass(kernel, grid)
    subjects, scans, values = [], [], []
    n_fixations = 0
    for record in target_records:
        if record.split != "valid" or record.length < 2:
            continue
        targets = np.arange(1, record.length)
        probability = history_probabilities(
            kernel, mass, record.iy[targets - 1], record.ix[targets - 1],
            record.iy[targets], record.ix[targets], grid, alpha,
        )
        ig = information_gain(probability, grid.n_bins)
        subjects.append(np.full(ig.size, record.subject))
        scans.append(np.full(ig.size, record.scan_id))
        values.append(ig)
        n_fixations += int(ig.size)
    if not values:
        return {"estimate": math.nan, "n_fixations": 0, "n_subjects": 0}
    means = scanpath_then_subject_means(np.concatenate(scans), np.concatenate(subjects), np.concatenate(values))
    return {
        "estimate": float(np.mean(means)),
        "n_subjects": int(len(means)),
        "n_fixations": n_fixations,
        "fit_split": "train",
        "eval_split": "valid",
    }


def width_fraction_to_sigma_px(fraction: float, grid: Grid) -> float:
    """Sigma expressed as a fraction of canvas width, in the grid's x units."""
    return float(fraction * grid.width)
