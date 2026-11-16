"""Image-level luminance and gradient comparison with a radius-matched control.

The 18 COCO-FreeView images and 9 OSIE images were chosen before this control.
The result is a description of those files. It is not a confirmatory test and
not a gaze model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from humangazelab.adapter import load_scanpaths
from humangazelab.image_features import canvas_radius, radius_matched_points, window_features
from humangazelab.osie import load_osie_scanpaths
from humangazelab.protocolshift import assert_new_output
from humangazelab.stimuli import align_fixations, load_grayscale

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "protocolshift_prep_v1"
RADIUS = 8
SEED = 20261114


def _points(scanpaths, image_name):
    xs, ys = [], []
    for scan in scanpaths:
        if scan.image_name != image_name:
            continue
        xs.extend(float(value) for value in scan.x)
        ys.extend(float(value) for value in scan.y)
    return np.asarray(xs), np.asarray(ys)


def _image_row(dataset, image_name, gray, xs, ys, rng):
    height, width = gray.shape
    align_fixations(gray, xs, ys, height, width)
    fixation = window_features(gray, xs, ys, RADIUS)
    matched_x, matched_y = radius_matched_points(xs, ys, width, height, rng, RADIUS)
    matched = window_features(gray, matched_x, matched_y, RADIUS)
    random_x = rng.uniform(RADIUS, width - RADIUS, size=xs.size)
    random_y = rng.uniform(RADIUS, height - RADIUS, size=ys.size)
    random = window_features(gray, random_x, random_y, RADIUS)
    placed = np.isfinite(matched_x)
    row = {
        "dataset": dataset,
        "image": image_name,
        "n_fixations": int(xs.size),
        "n_radius_matched": int(placed.sum()),
        "mean_fixation_radius_px": float(canvas_radius(xs, ys, width, height).mean()),
        "mean_random_radius_px": float(canvas_radius(random_x, random_y, width, height).mean()),
    }
    for name in ("luminance", "gradient", "variance"):
        fix = fixation[name]
        row[f"fixation_minus_unmatched_{name}"] = float(np.nanmean(fix - random[name]))
        row[f"fixation_minus_radius_matched_{name}"] = float(np.nanmean(fix[placed] - matched[name][placed]))
    return row


def _dataset(name, directory, scanpaths, width, height, rng):
    rows = []
    for path in sorted(Path(directory).glob("*.jpg")):
        gray = load_grayscale(path)
        align_fixations(gray, *_points(scanpaths, path.name), height, width)
        xs, ys = _points(scanpaths, path.name)
        rows.append(_image_row(name, path.name, gray, xs, ys, rng))
    return rows


def main() -> None:
    assert_new_output(ROOT, OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    coco = load_scanpaths(ROOT / "data" / "raw" / "COCOFreeView_fixations_trainval.json")
    osie = load_osie_scanpaths(ROOT / "data" / "raw" / "external" / "osie_fixations.mat")
    rows = _dataset("COCO-FreeView", ROOT / "data" / "raw" / "images_subset", coco, 1680, 1050, rng)
    rows += _dataset("OSIE", ROOT / "data" / "raw" / "external" / "osie_stimuli", osie, 800, 600, rng)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "center_distance_control.csv", index=False)
    summary = []
    for dataset, part in frame.groupby("dataset"):
        item = {"dataset": dataset, "n_images": int(len(part)), "confirmatory": False}
        for column in part.columns:
            if column.startswith("fixation_minus_"):
                values = part[column].to_numpy(dtype=float)
                item[column + "_image_mean"] = float(values.mean())
                item[column + "_image_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        summary.append(item)
    pd.DataFrame(summary).to_csv(OUT / "center_distance_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
