"""Verification checks for the frozen temporal_v1 and osie_v1 runs.

These functions do not write into outputs/temporal_v1 or outputs/osie_v1.
Bandwidth curves in temporal_v2 are train-fold scores. They do not read
validation information gain and they do not replace the frozen model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from humangazelab.adapter import load_scanpaths
from humangazelab.osie import load_osie_scanpaths
from humangazelab.summarize import metric_seed, percentile_bootstrap_mean
from humangazelab.temporal import (
    Grid,
    _group_records,
    accumulate_counts,
    aggregate_ig,
    blur_displacement,
    bootstrap_groups,
    canvas_kernel_mass,
    choose_maximum,
    density_from_counts,
    derangement,
    displacement_histogram,
    evaluate_records,
    pack_scanpaths,
    paired_subject_difference,
    scanpath_then_subject_means,
    select_center_sigma,
    select_history_sigma,
    select_lambdas,
    select_loo_sigma,
    sigma_in_bins,
)
from humangazelab.temporal_run import _paired_epoch

FROZEN_DIRS = ("outputs/temporal_v1", "outputs/osie_v1", "outputs/figures", "outputs/tables")
EXTENDED_SIGMAS = (4.0, 8.0, 12.0, 16.0, 24.0, 40.0, 64.0, 96.0, 128.0)


def assert_not_frozen(root: Path, output_dir: Path) -> None:
    resolved = output_dir.resolve()
    for relative in FROZEN_DIRS:
        if resolved == (root / relative).resolve():
            raise RuntimeError(f"refusing to write an audit into frozen directory {relative}")


def two_way_subject_bootstrap(subjects: np.ndarray, images: np.ndarray, values: np.ndarray, n_boot: int, seed: int) -> dict:
    """Resample subjects and images, then recompute the subject-equal mean.

    The point estimate is the unweighted mean of per-subject means. The interval
    resamples both factors with replacement. It is a sensitivity for crossed
    sampling, not a mixed model and not a causal estimate.
    """
    frame = pd.DataFrame({"subject": subjects, "image": images, "value": values})
    frame = frame[np.isfinite(frame["value"].to_numpy())]
    cell = frame.groupby(["subject", "image"], sort=False)["value"].mean()
    wide = cell.unstack("image")
    matrix = wide.to_numpy(dtype=float)
    subject_ids = wide.index.to_numpy()
    point_values = []
    for row in matrix:
        if np.any(np.isfinite(row)):
            point_values.append(float(np.nanmean(row)))
    point = percentile_bootstrap_mean(np.array(point_values), n_boot, seed)
    rng = np.random.default_rng(seed)
    n_subjects, n_images = matrix.shape
    draws = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        subject_index = rng.integers(0, n_subjects, n_subjects)
        image_index = rng.integers(0, n_images, n_images)
        means = []
        for row in matrix[subject_index][:, image_index]:
            if np.any(np.isfinite(row)):
                means.append(float(np.nanmean(row)))
        draws[draw] = float(np.mean(means)) if means else math.nan
    low, high = (float(v) for v in np.quantile(draws[np.isfinite(draws)], (0.025, 0.975)))
    point.update({
        "ci_low_crossed": low,
        "ci_high_crossed": high,
        "n_subjects": int(len(subject_ids)),
        "n_images": int(n_images),
        "estimand": "subject_equal_mean",
    })
    return point


def reference_geometry(scanpaths, index_min: int) -> pd.DataFrame:
    """Other observers present at the same released index, and the nearest one's distance."""
    grouped: dict[str, dict[int, list[tuple[int, float, float]]]] = {}
    for scan in scanpaths:
        bucket = grouped.setdefault(scan.image_name, {})
        for index in range(index_min, len(scan.x)):
            bucket.setdefault(index, []).append((int(scan.subject_id), float(scan.x[index]), float(scan.y[index])))
    rows = []
    for image, by_index in grouped.items():
        for index, points in by_index.items():
            for position, (subject, x_value, y_value) in enumerate(points):
                others = [point for other_position, point in enumerate(points) if other_position != position and point[0] != subject]
                if others:
                    distance = min(math.hypot(x_value - other[1], y_value - other[2]) for other in others)
                else:
                    distance = math.nan
                rows.append({
                    "image": image,
                    "subject": subject,
                    "fixation_index": index,
                    "n_other": len(others),
                    "nearest_other_px": distance,
                })
    return pd.DataFrame(rows)


def _fit_frozen(records, grid, alpha, sigma_center, sigma_history, sigma_loo, lambda_b5, lambda_b5c, shuffle_map, index_min):
    center = density_from_counts(
        accumulate_counts(records, lambda record: record.split == "train", index_min, None, grid),
        sigma_in_bins(sigma_center, grid),
        alpha,
    )
    subjects = sorted({record.subject for record in records})
    heldout = {}
    for subject in subjects:
        counts = accumulate_counts(
            records,
            lambda record, held=subject: record.split == "train" and record.subject != held,
            index_min,
            None,
            grid,
        )
        heldout[subject] = density_from_counts(counts, sigma_in_bins(sigma_center, grid), alpha)
    kernel = blur_displacement(
        displacement_histogram(records, lambda record: record.split == "train", grid),
        sigma_in_bins(sigma_history, grid),
    )
    return evaluate_records(
        records,
        grid,
        center,
        heldout,
        sigma_in_bins(sigma_loo, grid),
        kernel,
        canvas_kernel_mass(kernel, grid),
        alpha,
        lambda_b5,
        lambda_b5c,
        shuffle_map,
        index_min=index_min,
    )


def _shuffle(records, seed: int) -> dict[str, str]:
    by_split: dict[str, list[str]] = {}
    for record in records:
        by_split.setdefault(record.split, []).append(record.image)
    mapping: dict[str, str] = {}
    for split, names in by_split.items():
        mapping.update(derangement(sorted(set(names)), seed + (0 if split == "train" else 1)))
    return mapping


def _subject_table(frame: pd.DataFrame, column: str, mask: np.ndarray, n_boot: int, seed: int, label: str) -> dict:
    part = frame.loc[mask]
    values = scanpath_then_subject_means(part["scan_id"].to_numpy(), part["subject"].to_numpy(), part[column].to_numpy())
    summary = percentile_bootstrap_mean(values, n_boot, metric_seed(seed, column, label, "subject_scanpath_equal"))
    summary.update({
        "metric": column,
        "subset": label,
        "estimand": "subject_scanpath_equal",
        "n_fixations_finite": int(np.isfinite(part[column].to_numpy()).sum()),
        "n_scanpaths": int(part["scan_id"].nunique()) if len(part) else 0,
    })
    return summary


def _image_table(frame: pd.DataFrame, column: str, mask: np.ndarray, n_boot: int, seed: int, label: str) -> dict:
    part = frame.loc[mask]
    part = part[np.isfinite(part[column].to_numpy())]
    if part.empty:
        values = np.empty(0)
    else:
        values = part.groupby("image")[column].mean().to_numpy(dtype=float)
    summary = percentile_bootstrap_mean(values, n_boot, metric_seed(seed, column, label, "image"))
    summary.update({"metric": column, "subset": label, "estimand": "image_equal_mean", "n_images": int(len(values))})
    return summary


def length_rows(frame: pd.DataFrame, index_min: int, index_max: int, n_boot: int, seed: int, valid_name: str) -> list[dict]:
    valid = frame["split"].eq(valid_name).to_numpy()
    length = frame["length"].to_numpy()
    index = frame["fixation_index"].to_numpy()
    subsets = {
        "all_valid": valid,
        "length_at_least_12": valid & (length >= 12),
        "length_at_least_16": valid & (length >= 16),
        "length_below_8": valid & (length < 8),
        "reaches_index_15": valid & (length > 15),
    }
    rows = []
    columns = ("ig_B1_center", "ig_B2_loo_image", "ig_B3_loo_index", "ig_B4_history")
    for name, base in subsets.items():
        index_masks = {"all_in_subset": base}
        for fixation_index in range(index_min, index_max + 1):
            index_masks[f"index_{fixation_index}"] = base & (index == fixation_index)
        for epoch, mask in index_masks.items():
            if int(mask.sum()) == 0:
                continue
            for column in columns:
                rows.append(_subject_table(frame, column, mask, n_boot, seed, f"{name}|{epoch}"))
    return rows


def b3_diagnostic(frame: pd.DataFrame, geometry: pd.DataFrame, alpha: float, n_bins: int, valid_name: str) -> pd.DataFrame:
    part = frame[frame["split"].eq(valid_name)][
        ["image", "subject", "fixation_index", "ig_B3_loo_index", "length", "scan_id"]
    ].copy()
    merged = part.merge(geometry, on=["image", "subject", "fixation_index"], how="left")
    floor_ig = math.log2(alpha)
    merged["near_floor"] = merged["ig_B3_loo_index"] <= floor_ig + 0.15
    rows = []
    for index, group in merged.groupby("fixation_index"):
        finite = group["ig_B3_loo_index"].to_numpy()
        rows.append({
            "fixation_index": int(index),
            "n_fixations": int(len(group)),
            "mean_n_other": float(group["n_other"].mean()),
            "median_n_other": float(group["n_other"].median()),
            "min_n_other": int(group["n_other"].min()),
            "fraction_no_other": float((group["n_other"] == 0).mean()),
            "median_nearest_other_px": (
                float(np.nanmedian(group["nearest_other_px"]))
                if np.isfinite(group["nearest_other_px"].to_numpy(dtype=float)).any()
                else math.nan
            ),
            "fraction_near_floor": float(group["near_floor"].mean()),
            "fixation_mean_ig": float(np.nanmean(finite)),
            "floor_ig_bits": floor_ig,
            "n_bins": n_bins,
        })
    return pd.DataFrame(rows)


def uncertainty_rows(frame: pd.DataFrame, valid_name: str, n_boot: int, seed: int, subset: str) -> list[dict]:
    """Subject-cluster, image-cluster, and crossed intervals for one published subset.

    The subject and image intervals use the same estimator as temporal_v1.
    The crossed interval resamples subjects and images, then recomputes the
    subject-equal mean. It is a sensitivity, not a replacement for the subject interval.
    """
    valid = frame["split"].eq(valid_name).to_numpy()
    rows = []
    for column in ("ig_B1_center", "ig_B2_loo_image", "ig_B3_loo_index", "ig_B4_history", "ig_B5c_loo_history"):
        subject_row = aggregate_ig(frame, column, valid, n_boot, seed, subset, "subject_scanpath_equal")
        image_row = aggregate_ig(frame, column, valid, n_boot, seed, subset, "image")
        part = frame.loc[valid]
        finite = np.isfinite(part[column].to_numpy())
        crossed = two_way_subject_bootstrap(
            part.loc[finite, "subject"].to_numpy(),
            part.loc[finite, "image"].to_numpy(),
            part.loc[finite, column].to_numpy(),
            n_boot,
            metric_seed(seed, column, subset, "crossed_subject_equal"),
        )
        rows.append(_uncertainty_record(column, subject_row, image_row, crossed))
    subject_diff = paired_subject_difference(
        frame, "ig_B5c_loo_history", "ig_B2_loo_image", valid, n_boot, seed, subset
    )
    part = frame.loc[valid].copy()
    part["diff"] = part["ig_B5c_loo_history"] - part["ig_B2_loo_image"]
    image_values = part.groupby("image")["diff"].mean().to_numpy(dtype=float)
    image_diff = bootstrap_groups(
        image_values,
        "ig_B5c_loo_history_minus_ig_B2_loo_image",
        subset,
        "image",
        n_boot,
        seed,
    )
    finite = np.isfinite(part["diff"].to_numpy())
    crossed = two_way_subject_bootstrap(
        part.loc[finite, "subject"].to_numpy(),
        part.loc[finite, "image"].to_numpy(),
        part.loc[finite, "diff"].to_numpy(),
        n_boot,
        metric_seed(seed, "ig_B5c_loo_history_minus_ig_B2_loo_image", subset, "crossed_subject_equal"),
    )
    row = _uncertainty_record("ig_B5c_loo_history_minus_ig_B2_loo_image", subject_diff, image_diff, crossed)
    row["same_image_human_reference_in_both"] = True
    row["b5c_also_uses_own_previous_fixation"] = True
    row["note"] = (
        "B2 is the leave-one-observer density on the same image. "
        "B5c mixes that density with the target observer's own previous fixation. "
        "Neither model is an image-only predictor."
    )
    rows.append(row)
    return rows


def _uncertainty_record(metric: str, subject_row: dict, image_row: dict, crossed: dict) -> dict:
    return {
        "metric": metric,
        "subject_equal_estimate": subject_row["estimate"],
        "subject_ci_low": subject_row["ci_low"],
        "subject_ci_high": subject_row["ci_high"],
        "subject_ci_width": subject_row["ci_high"] - subject_row["ci_low"],
        "image_equal_estimate": image_row["estimate"],
        "image_ci_low": image_row["ci_low"],
        "image_ci_high": image_row["ci_high"],
        "image_ci_width": image_row["ci_high"] - image_row["ci_low"],
        "crossed_estimate": crossed["estimate"],
        "crossed_ci_low": crossed["ci_low_crossed"],
        "crossed_ci_high": crossed["ci_high_crossed"],
        "crossed_ci_width": crossed["ci_high_crossed"] - crossed["ci_low_crossed"],
        "n_subjects": crossed["n_subjects"],
        "n_images": crossed["n_images"],
    }


def survival_contrasts(frame: pd.DataFrame, valid_name: str, early: tuple[int, int], late_start: int, n_boot: int, seed: int) -> list[dict]:
    """Early-minus-late within scanpaths that survive to a late index.

    A drop that remains inside long scanpaths is not explained only by short
    scanpaths leaving the later indices.
    """
    valid = frame["split"].eq(valid_name).to_numpy()
    length = frame["length"].to_numpy()
    index = frame["fixation_index"].to_numpy()
    rows = []
    subsets = {
        "all_valid": valid,
        "length_at_least_12": valid & (length >= 12),
        "reaches_index_15": valid & (length > 15),
    }
    for name, base in subsets.items():
        early_mask = base & (index >= early[0]) & (index <= early[1])
        late_mask = base & (index >= late_start)
        if int(early_mask.sum()) == 0 or int(late_mask.sum()) == 0:
            continue
        for column in ("ig_B1_center", "ig_B2_loo_image", "ig_B3_loo_index", "ig_B4_history"):
            rows.append(_paired_epoch(
                frame, column, early_mask, late_mask, n_boot, seed,
                f"{name}|early_minus_late", f"{column}_early_minus_late",
            ))
    return rows


def train_bandwidth_curve(records, grid: Grid, alpha: float, index_min: int, dataset: str) -> pd.DataFrame:
    """Score candidate bandwidths on the train selection fold only."""
    train_records = [record for record in records if record.split == "train"]
    if any(record.split != "train" for record in train_records):
        raise AssertionError("bandwidth selection received a non-train scanpath")
    grouped = _group_records(train_records)
    candidates = list(EXTENDED_SIGMAS)
    _, center_rows = select_center_sigma(train_records, grid, candidates, alpha, index_min)
    _, history_rows = select_history_sigma(train_records, grid, candidates, alpha)
    _, loo_rows = select_loo_sigma(grouped, grid, candidates, alpha, index_min)
    sigma_center = choose_maximum({row["candidate"]: row["select_subject_mean_ig_bits"] for row in center_rows}, candidates)
    sigma_history = choose_maximum({row["candidate"]: row["select_subject_mean_ig_bits"] for row in history_rows}, candidates)
    sigma_loo = choose_maximum({row["candidate"]: row["select_subject_mean_ig_bits"] for row in loo_rows}, candidates)
    fit_center = density_from_counts(
        accumulate_counts(train_records, lambda record: record.role == "fit", index_min, None, grid),
        sigma_in_bins(sigma_center, grid),
        alpha,
    )
    fit_kernel = blur_displacement(
        displacement_histogram(train_records, lambda record: record.role == "fit", grid),
        sigma_in_bins(sigma_history, grid),
    )
    lambda_b5, lambda_b5c, lambda_rows = select_lambdas(
        train_records,
        grouped,
        grid,
        fit_center,
        fit_kernel,
        canvas_kernel_mass(fit_kernel, grid),
        sigma_in_bins(sigma_loo, grid),
        alpha,
        [0.0, 0.25, 0.5, 0.75, 1.0],
        index_min=index_min,
    )
    rows = []
    for source in (center_rows, history_rows, loo_rows, lambda_rows):
        for row in source:
            copied = dict(row)
            copied["dataset"] = dataset
            copied["validation_accessed"] = False
            copied["selected_sigma_center_px"] = sigma_center
            copied["selected_sigma_history_px"] = sigma_history
            copied["selected_sigma_loo_px"] = sigma_loo
            copied["selected_lambda_center_history"] = lambda_b5
            copied["selected_lambda_loo_history"] = lambda_b5c
            rows.append(copied)
    return pd.DataFrame(rows)


def checksum_against_published(frame: pd.DataFrame, published: pd.DataFrame, subset: str, metrics: tuple[str, ...]) -> list[dict]:
    rows = []
    valid = frame["split"].eq("valid")
    for metric in metrics:
        fresh = scanpath_then_subject_means(
            frame.loc[valid, "scan_id"].to_numpy(),
            frame.loc[valid, "subject"].to_numpy(),
            frame.loc[valid, metric].to_numpy(),
        )
        fresh_mean = float(np.mean(fresh)) if len(fresh) else math.nan
        match = published[(published["subset"] == subset) & (published["metric"] == metric) & (published["estimand"] == "subject_scanpath_equal")]
        published_mean = float(match["estimate"].iloc[0]) if len(match) == 1 else math.nan
        rows.append({
            "metric": metric,
            "published_estimate": published_mean,
            "recomputed_estimate": fresh_mean,
            "absolute_difference": abs(fresh_mean - published_mean) if np.isfinite(fresh_mean) and np.isfinite(published_mean) else math.nan,
            "match": bool(np.isfinite(fresh_mean) and np.isfinite(published_mean) and abs(fresh_mean - published_mean) < 1e-9),
        })
    return rows


def _early_late(frame, column, early_mask, late_mask, n_boot, seed, label):
    return _paired_epoch(frame, column, early_mask, late_mask, n_boot, seed, label, f"{column}_early_minus_late")


def plot_length_curves(length_table: pd.DataFrame, path: Path, index_min: int) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    for subset, color in (("all_valid|index_", "#1b9e77"), ("reaches_index_15|index_", "#d95f02")):
        rows = length_table[(length_table["metric"] == "ig_B3_loo_index") & length_table["subset"].str.startswith(subset)].copy()
        if rows.empty:
            continue
        rows["index"] = rows["subset"].str.extract(r"index_(\d+)$").astype(float)
        rows = rows.dropna(subset=["index"]).sort_values("index")
        ax.plot(rows["index"], rows["estimate"], color=color, label=subset.split("|")[0])
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Released fixation index")
    ax.set_ylabel("B3 information gain (bits)")
    ax.set_title("Same-index reference, all scanpaths and long scanpaths")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def overlay_osie(scanpaths, image_dir: Path, names: list[str], path: Path) -> pd.DataFrame:
    rows = []
    present = [name for name in names if (image_dir / name).exists()]
    if not present:
        return pd.DataFrame(rows)
    columns = 4
    rows_n = int(math.ceil(len(present) / columns))
    fig, axes = plt.subplots(rows_n, columns, figsize=(3.2 * columns, 2.8 * rows_n), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    by_name: dict[str, list] = {}
    for scan in scanpaths:
        by_name.setdefault(scan.image_name, []).append(scan)
    for axis in axes:
        axis.axis("off")
    for position, name in enumerate(present):
        axis = axes[position]
        axis.axis("on")
        image = Image.open(image_dir / name).convert("RGB")
        axis.imshow(image, extent=[0, image.width, image.height, 0], origin="upper")
        inside = 0
        total = 0
        for scan in by_name.get(name, []):
            total += len(scan.x)
            inside += int(np.sum((scan.x >= 0) & (scan.x <= image.width) & (scan.y >= 0) & (scan.y <= image.height)))
            axis.plot(scan.x, scan.y, color="yellow", linewidth=0.6, alpha=0.7)
            if len(scan.x):
                axis.scatter(scan.x[0], scan.y[0], s=12, facecolors="none", edgecolors="red", linewidths=0.6)
        axis.set_xlim(0, image.width)
        axis.set_ylim(image.height, 0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{name} {image.width}x{image.height}", fontsize=8)
        rows.append({
            "image": name,
            "width_px": image.width,
            "height_px": image.height,
            "n_fixations": total,
            "n_fixations_inside_image": inside,
            "coordinate_frame_checked": "released x,y drawn on the JPEG with y increasing downward",
        })
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return pd.DataFrame(rows)
