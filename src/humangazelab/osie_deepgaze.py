"""Eligibility and paired summaries for the frozen DeepGaze III OSIE run.

The metric, the 35/24 scale, and the four-fixation history are fixed in
configs/osie_deepgaze3_v2.yaml. This module does not load a network and does
not read model scores in order to choose images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from humangazelab.audit_checkpoint import two_way_subject_bootstrap
from humangazelab.summarize import metric_seed, percentile_bootstrap_mean

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
MINIMUM_TARGET_INDEX = 4


def scaled_coordinate(values, source: int, resized: int) -> np.ndarray:
    """Map canvas coordinates onto the resized grid.

    Multiply by the resized size, then divide by the source size. Factoring the
    ratio first changes values that land on a half pixel.
    """
    return np.asarray(values, dtype=np.float64) * resized / source


def scaled_pixel(coordinate: float, source: int, resized: int) -> int:
    """Nearest pixel on the resized grid, using the same product as the nine-image run."""
    return int(np.clip(np.rint(scaled_coordinate(coordinate, source, resized)), 0, resized - 1))


def previously_scored_stems(path: str | Path) -> tuple[str, ...]:
    """Image stems already present in the committed nine-image table."""
    frame = pd.read_csv(path)
    return tuple(sorted(frame["image"].astype(str).unique()))


def _inside(x: float, y: float) -> bool:
    return bool(np.isfinite(x) and np.isfinite(y) and 0 <= x < CANVAS_WIDTH and 0 <= y < CANVAS_HEIGHT)


def paired_targets(scanpaths) -> tuple[list[dict], dict[str, int]]:
    """Targets with four preceding fixations, paired to a wrong image when one exists.

    The partner is the next image in sorted-name order, for the same subject,
    whose released scanpath contains that index. A target with no such image is
    counted and dropped, so the true-image rows and the control rows are the
    same set. Index 0 stays in the history and is not a target.
    """
    lengths: dict[tuple[str, int], int] = {}
    coordinates: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for scan in scanpaths:
        key = (str(scan.image_name), int(scan.subject_id))
        if key in lengths:
            raise ValueError(f"duplicate scanpath for {key}")
        lengths[key] = int(len(scan.x))
        coordinates[key] = (np.asarray(scan.x, dtype=float), np.asarray(scan.y, dtype=float))
    names = sorted({key[0] for key in lengths})
    by_subject: dict[int, list[str]] = {}
    for image_name, subject in lengths:
        by_subject.setdefault(subject, []).append(image_name)
    for subject, images in by_subject.items():
        images.sort()
        by_subject[subject] = images

    counts = {
        "n_fixations": 0,
        "n_index_below_4": 0,
        "n_index_0": 0,
        "n_outside_canvas": 0,
        "n_partner_outside_canvas": 0,
        "n_no_partner": 0,
        "n_scored": 0,
    }
    rows: list[dict] = []
    for image_name, subject in sorted(lengths):
        x_values, y_values = coordinates[(image_name, subject)]
        images = by_subject[subject]
        position = images.index(image_name)
        for index in range(len(x_values)):
            counts["n_fixations"] += 1
            if index == 0:
                counts["n_index_0"] += 1
            if index < MINIMUM_TARGET_INDEX:
                counts["n_index_below_4"] += 1
                continue
            if not _inside(float(x_values[index]), float(y_values[index])):
                counts["n_outside_canvas"] += 1
                continue
            partner = None
            for step in range(1, len(images)):
                candidate = images[(position + step) % len(images)]
                if lengths[(candidate, subject)] > index:
                    partner = candidate
                    break
            if partner is None:
                counts["n_no_partner"] += 1
                continue
            partner_x, partner_y = coordinates[(partner, subject)]
            if not _inside(float(partner_x[index]), float(partner_y[index])):
                counts["n_partner_outside_canvas"] += 1
                continue
            counts["n_scored"] += 1
            rows.append(
                {
                    "image": Path(image_name).stem,
                    "image_name": image_name,
                    "subject": int(subject),
                    "target_index": int(index),
                    "x": float(x_values[index]),
                    "y": float(y_values[index]),
                    "partner_image": Path(partner).stem,
                    "partner_x": float(partner_x[index]),
                    "partner_y": float(partner_y[index]),
                }
            )
    if counts["n_scored"] != len(rows):
        raise RuntimeError("scored count does not match the paired rows")
    return rows, counts


def group_means(frame: pd.DataFrame, value: str, group: str) -> pd.Series:
    return frame.groupby(group, sort=True)[value].mean()


def summarize_paired(frame: pd.DataFrame, n_boot: int, seed: int, subset: str) -> list[dict]:
    """Image-level, observer-level, and crossed intervals for paired differences.

    Fixations are averaged inside each image or observer before the bootstrap.
    The crossed interval resamples images and observers together.
    """
    rows = []
    for column in ("information_gain_bits", "wrong_image_information_gain_bits", "paired_difference_bits"):
        image_values = group_means(frame, column, "image").to_numpy()
        image_stats = percentile_bootstrap_mean(image_values, n_boot, metric_seed(seed, column, subset, "image"))
        image_stats.update({"subset": subset, "column": column, "estimand": "image_equal_mean"})
        subject_values = group_means(frame, column, "subject").to_numpy()
        subject_stats = percentile_bootstrap_mean(
            subject_values, n_boot, metric_seed(seed, column, subset, "subject")
        )
        subject_stats.update({"subset": subset, "column": column, "estimand": "subject_equal_mean"})
        crossed = two_way_subject_bootstrap(
            frame["subject"].to_numpy(),
            frame["image"].to_numpy(),
            frame[column].to_numpy(dtype=float),
            n_boot,
            metric_seed(seed, column, subset, "crossed"),
        )
        crossed_row = {
            "subset": subset,
            "column": column,
            "estimand": "subject_equal_mean_crossed_resample",
            "n_groups": crossed["n_subjects"],
            "estimate": crossed["estimate"],
            "sd": crossed["sd"],
            "sem": crossed["sem"],
            "ci_low": crossed["ci_low_crossed"],
            "ci_high": crossed["ci_high_crossed"],
            "n_images": crossed["n_images"],
            "n_fixations": int(len(frame)),
        }
        for stats in (image_stats, subject_stats):
            stats["n_fixations"] = int(len(frame))
            stats["n_images"] = int(frame["image"].nunique())
            stats["n_subjects"] = int(frame["subject"].nunique())
        rows.extend([image_stats, subject_stats, crossed_row])
    return rows
