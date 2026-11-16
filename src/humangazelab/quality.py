"""Integrity counts for a loaded COCO-FreeView release.

Counts are computed from the objects passed in. This module does not download
or reinterpret missing fields.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from humangazelab.adapter import Scanpath

README_FIELDS = (
    "trial_ind",
    "name",
    "subject",
    "task",
    "condition",
    "X",
    "Y",
    "T",
    "length",
    "split",
)
FILE_FIELDS = (
    "name",
    "subject",
    "condition",
    "X",
    "Y",
    "T",
    "length",
    "split",
    "fixOnTarget",
    "correct",
)


def quality_report(scanpaths: list[Scanpath], canvas_width: float, canvas_height: float) -> dict:
    if not scanpaths:
        raise ValueError("quality_report received no scanpaths")
    cx = canvas_width / 2.0
    cy = canvas_height / 2.0
    n_fixations = 0
    outside = 0
    not_strictly_increasing = 0
    nondecreasing = 0
    nonpositive_duration = 0
    duration_gt_5000 = 0
    sum_gt_5000 = 0
    sum_gt_10000 = 0
    length_one = 0
    xmin = math.inf
    xmax = -math.inf
    ymin = math.inf
    ymax = -math.inf
    tmin = math.inf
    tmax = -math.inf
    duration_sums = []
    first_radii = []
    subjects = Counter()
    splits = Counter()
    conditions = Counter()
    corrects = Counter()
    fix_on_target = Counter()
    images = set()
    images_by_split: dict[str, set[str]] = {}
    subjects_by_image: dict[str, set[int]] = {}
    pairs = set()
    duplicate_pairs = 0
    split_fixations = Counter()
    subject_fixations = Counter()

    for scanpath in scanpaths:
        n = len(scanpath.x)
        n_fixations += n
        subjects[scanpath.subject_id] += 1
        splits[scanpath.split] += 1
        conditions[scanpath.condition] += 1
        corrects[scanpath.correct] += 1
        fix_on_target[scanpath.fix_on_target] += 1
        images.add(scanpath.image_name)
        images_by_split.setdefault(scanpath.split, set()).add(scanpath.image_name)
        subjects_by_image.setdefault(scanpath.image_name, set()).add(scanpath.subject_id)
        pair = (scanpath.image_name, scanpath.subject_id, scanpath.split)
        if pair in pairs:
            duplicate_pairs += 1
        pairs.add(pair)
        split_fixations[scanpath.split] += n
        subject_fixations[scanpath.subject_id] += n
        if n == 1:
            length_one += 1
        duration = scanpath.duration
        if n >= 2 and np.any(np.diff(duration) <= 0):
            not_strictly_increasing += 1
        if n == 1 or np.all(np.diff(duration) >= 0):
            nondecreasing += 1
        nonpositive_duration += int(np.sum(duration <= 0))
        duration_gt_5000 += int(np.sum(duration > 5000))
        total = float(duration.sum())
        duration_sums.append(total)
        if total > 5000:
            sum_gt_5000 += 1
        if total > 10000:
            sum_gt_10000 += 1
        xmin = min(xmin, float(scanpath.x.min()))
        xmax = max(xmax, float(scanpath.x.max()))
        ymin = min(ymin, float(scanpath.y.min()))
        ymax = max(ymax, float(scanpath.y.max()))
        tmin = min(tmin, float(duration.min()))
        tmax = max(tmax, float(duration.max()))
        outside += int(np.sum(
            (scanpath.x < 0) | (scanpath.x > canvas_width) | (scanpath.y < 0) | (scanpath.y > canvas_height)
        ))
        first_radii.append(math.hypot(float(scanpath.x[0] - cx), float(scanpath.y[0] - cy)))

    sums = np.asarray(duration_sums)
    radii = np.asarray(first_radii)
    coverage = Counter(len(v) for v in subjects_by_image.values())
    return {
        "n_scanpaths": len(scanpaths),
        "n_fixations": n_fixations,
        "n_images": len(images),
        "n_subjects": len(subjects),
        "subject_ids": sorted(subjects),
        "subject_scanpath_counts": dict(sorted(subjects.items())),
        "subject_fixation_counts": dict(sorted(subject_fixations.items())),
        "split_scanpath_counts": dict(splits),
        "split_fixation_counts": dict(split_fixations),
        "split_image_counts": {key: len(value) for key, value in images_by_split.items()},
        "condition_counts": dict(conditions),
        "correct_counts": {str(key): value for key, value in corrects.items()},
        "fix_on_target_counts": {str(key): value for key, value in fix_on_target.items()},
        "duplicate_image_subject_split": duplicate_pairs,
        "x_min": xmin,
        "x_max": xmax,
        "y_min": ymin,
        "y_max": ymax,
        "duration_min": tmin,
        "duration_max": tmax,
        "n_fixations_outside_canvas": outside,
        "n_sequences_not_strictly_increasing_duration": not_strictly_increasing,
        "n_sequences_nondecreasing_duration": nondecreasing,
        "n_nonpositive_durations": nonpositive_duration,
        "n_durations_gt_5000": duration_gt_5000,
        "n_scanpaths_duration_sum_gt_5000": sum_gt_5000,
        "n_scanpaths_duration_sum_gt_10000": sum_gt_10000,
        "duration_sum_min": float(sums.min()),
        "duration_sum_median": float(np.median(sums)),
        "duration_sum_max": float(sums.max()),
        "n_length_1": length_one,
        "first_fixation_radius_median": float(np.median(radii)),
        "fraction_first_fixation_within_50px": float(np.mean(radii <= 50)),
        "fraction_first_fixation_within_100px": float(np.mean(radii <= 100)),
        "images_by_n_subjects": dict(sorted(coverage.items())),
    }


def quality_table(report: dict) -> pd.DataFrame:
    rows = []
    for key, value in report.items():
        rows.append({"key": key, "value": _format_value(value)})
    return pd.DataFrame(rows)


def _format_value(value) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def schema_comparison(report: dict) -> pd.DataFrame:
    """Compare readme v1.0 field names with the fields observed in this release."""
    observed = set(FILE_FIELDS)
    documented = set(README_FIELDS)
    notes = {
        "trial_ind": "Named in readme v1.0. Absent from the train+validation JSON.",
        "task": "Named in readme v1.0 as the COCO-Search18 target category. Absent from the JSON.",
        "condition": (
            "Readme v1.0 describes target-present or target-absent labels from COCO-Search18. "
            f"Observed value counts: {report['condition_counts']}."
        ),
        "fixOnTarget": (
            "Not named in readme v1.0. Present in the JSON. "
            f"Observed value counts: {report['fix_on_target_counts']}."
        ),
        "correct": (
            "Not named in readme v1.0. Present in the JSON. "
            f"Observed value counts: {report['correct_counts']}."
        ),
        "T": (
            "Readme v1.0 calls T fixation durations ordered from first to last, and does not name the unit. "
            f"{report['n_sequences_not_strictly_increasing_duration']} of {report['n_scanpaths']} "
            "sequences are not strictly increasing, which is expected for durations and not for timestamps."
        ),
        "X": f"Observed range [{report['x_min']}, {report['x_max']}] on the readme canvas width.",
        "Y": f"Observed range [{report['y_min']}, {report['y_max']}] on the readme canvas height.",
        "length": "Matches len(X) for every loaded record; the loader rejects mismatches.",
        "split": f"Observed scanpath counts: {report['split_scanpath_counts']}.",
        "name": "COCO-style JPEG file name.",
        "subject": f"Observed ids: {report['subject_ids']}.",
    }
    rows = []
    for field in list(README_FIELDS) + [field for field in FILE_FIELDS if field not in documented]:
        rows.append(
            {
                "field": field,
                "in_readme_v1_0": field in documented,
                "in_trainval_json": field in observed,
                "note": notes.get(field, ""),
            }
        )
    return pd.DataFrame(rows)


def subject_coverage_table(scanpaths: list[Scanpath]) -> pd.DataFrame:
    rows = []
    subjects = sorted({sp.subject_id for sp in scanpaths})
    for subject in subjects:
        owned = [sp for sp in scanpaths if sp.subject_id == subject]
        rows.append(
            {
                "subject_id": subject,
                "n_scanpaths": len(owned),
                "n_images": len({sp.image_name for sp in owned}),
                "n_train": sum(sp.split == "train" for sp in owned),
                "n_valid": sum(sp.split == "valid" for sp in owned),
                "n_fixations": int(sum(len(sp.x) for sp in owned)),
            }
        )
    return pd.DataFrame(rows)


def split_counts_table(scanpaths: list[Scanpath]) -> pd.DataFrame:
    rows = []
    for split in sorted({sp.split for sp in scanpaths}):
        owned = [sp for sp in scanpaths if sp.split == split]
        rows.append(
            {
                "split": split,
                "n_scanpaths": len(owned),
                "n_images": len({sp.image_name for sp in owned}),
                "n_fixations": int(sum(len(sp.x) for sp in owned)),
                "n_subjects": len({sp.subject_id for sp in owned}),
            }
        )
    return pd.DataFrame(rows)
