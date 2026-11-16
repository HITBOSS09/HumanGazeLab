"""Low-level features on MIT1003 tuning images, after the official fixation conversion.

The reserved final partition is not read. Windows are in pixels. They are not
degrees of visual angle. No object masks are in the local MIT1003 release, so
no object descriptor is computed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from humangazelab.check_fixations import assert_open_partition
from humangazelab.image_features import (
    center_antipode,
    directional_energy,
    gradient_energy,
    radius_matched_points,
    window_features,
    window_mean,
)
from humangazelab.protocolshift import assert_new_output
from humangazelab.stimuli import load_grayscale

ROOT = Path(__file__).resolve().parents[1]
STIMULI = Path("/tmp/ALLSTIMULI.zip")
FIXATIONS = ROOT / "outputs" / "baseline_v1" / "mit1003_tune_fixations.csv"
OUTPUT = ROOT / "outputs" / "baseline_v1"
RADIUS = 16
SEED = 20261114
FEATURES = ("luminance", "rms_contrast", "gradient", "orientation_energy", "orientation_anisotropy")


def _maps(gray: np.ndarray) -> dict[str, np.ndarray]:
    directions = directional_energy(gray)
    stacked = np.stack(list(directions.values()))
    return {
        "gradient": gradient_energy(gray),
        "orientation_energy": stacked.mean(axis=0),
        "orientation_anisotropy": stacked.max(axis=0) - stacked.min(axis=0),
    }


def _collect(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    height, width = gray.shape
    base = window_features(gray, xs, ys, RADIUS)
    values = {
        "luminance": base["luminance"],
        "rms_contrast": np.sqrt(np.maximum(base["variance"], 0.0)),
    }
    maps = _maps(gray)
    for name, field in maps.items():
        values[name] = window_mean(field, xs, ys, RADIUS)
    matched_x, matched_y = radius_matched_points(xs, ys, width, height, rng, margin=1)
    anti_x, anti_y = center_antipode(xs, ys, width, height)
    inside_anti = (anti_x >= 0) & (anti_x < width) & (anti_y >= 0) & (anti_y < height)
    anti_x = np.where(inside_anti, anti_x, np.nan)
    anti_y = np.where(inside_anti, anti_y, np.nan)
    controls = {}
    matched_base = window_features(gray, matched_x, matched_y, RADIUS)
    controls["luminance_radius_matched"] = matched_base["luminance"]
    controls["rms_contrast_radius_matched"] = np.sqrt(np.maximum(matched_base["variance"], 0.0))
    anti_base = window_features(gray, anti_x, anti_y, RADIUS)
    controls["luminance_antipode"] = anti_base["luminance"]
    controls["rms_contrast_antipode"] = np.sqrt(np.maximum(anti_base["variance"], 0.0))
    for name, field in maps.items():
        controls[f"{name}_radius_matched"] = window_mean(field, matched_x, matched_y, RADIUS)
        controls[f"{name}_antipode"] = window_mean(field, anti_x, anti_y, RADIUS)
    values.update(controls)
    return values


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def main() -> None:
    assert_new_output(ROOT, OUTPUT)
    if not FIXATIONS.is_file():
        raise FileNotFoundError(FIXATIONS)
    fixations = pd.read_csv(FIXATIONS)
    stems = sorted(fixations["image_stem"].astype(str).unique())
    for stem in stems:
        if assert_open_partition(stem) != "tune":
            raise RuntimeError(f"{stem} is not a tuning image")
    stimuli_dir = ROOT / "data" / "raw" / "external" / "mit1003_tune_stimuli"
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(STIMULI) as archive:
        names = set(archive.namelist())
        for stem in stems:
            member = f"ALLSTIMULI/{stem}.jpeg"
            if member not in names:
                raise FileNotFoundError(member)
            destination = stimuli_dir / f"{stem}.jpeg"
            if not destination.exists():
                destination.write_bytes(archive.read(member))
    rng = np.random.default_rng(SEED)
    image_rows = []
    observer_rows = []
    alignment = []
    for stem in stems:
        gray = load_grayscale(stimuli_dir / f"{stem}.jpeg")
        height, width = gray.shape
        subset = fixations.loc[fixations["image_stem"].eq(stem)]
        xs = subset["x"].to_numpy(dtype=float)
        ys = subset["y"].to_numpy(dtype=float)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        alignment.append(
            {
                "image_stem": stem,
                "width": width,
                "height": height,
                "n_fixations": int(len(subset)),
                "n_inside": int(inside.sum()),
                "n_outside": int((~inside).sum()),
                "max_x": float(np.nanmax(xs)) if len(xs) else float("nan"),
                "max_y": float(np.nanmax(ys)) if len(ys) else float("nan"),
            }
        )
        kept = subset.loc[inside].copy()
        if kept.empty:
            continue
        measured = _collect(gray, kept["x"].to_numpy(dtype=float), kept["y"].to_numpy(dtype=float), rng)
        kept = kept.reset_index(drop=True)
        for feature in FEATURES:
            difference_matched = measured[feature] - measured[f"{feature}_radius_matched"]
            difference_antipode = measured[feature] - measured[f"{feature}_antipode"]
            image_rows.append(
                {
                    "image_stem": stem,
                    "feature": feature,
                    "fixation_mean": _mean(measured[feature]),
                    "radius_matched_difference": _mean(difference_matched),
                    "antipode_difference": _mean(difference_antipode),
                    "n_fixations": int(np.isfinite(measured[feature]).sum()),
                }
            )
            frame = pd.DataFrame(
                {
                    "user": kept["user"].to_numpy(),
                    "matched": difference_matched,
                    "antipode": difference_antipode,
                }
            )
            for user, group in frame.groupby("user"):
                observer_rows.append(
                    {
                        "image_stem": stem,
                        "user": user,
                        "feature": feature,
                        "radius_matched_difference": _mean(group["matched"].to_numpy()),
                        "antipode_difference": _mean(group["antipode"].to_numpy()),
                    }
                )
    image_frame = pd.DataFrame(image_rows)
    observer_frame = pd.DataFrame(observer_rows)
    alignment_frame = pd.DataFrame(alignment)
    image_frame.to_csv(OUTPUT / "visual_feature_image_means.csv", index=False)
    observer_frame.to_csv(OUTPUT / "visual_feature_observer_means.csv", index=False)
    alignment_frame.to_csv(OUTPUT / "visual_feature_alignment.csv", index=False)
    summary_rows = []
    for feature in FEATURES:
        images = image_frame.loc[image_frame["feature"].eq(feature)]
        observers = (
            observer_frame.loc[observer_frame["feature"].eq(feature)]
            .groupby("user")[["radius_matched_difference", "antipode_difference"]]
            .mean()
        )
        summary_rows.append(
            {
                "feature": feature,
                "n_images": int(images["image_stem"].nunique()),
                "image_mean_radius_matched_difference": float(images["radius_matched_difference"].mean()),
                "image_sd_radius_matched_difference": float(images["radius_matched_difference"].std(ddof=1)),
                "image_mean_antipode_difference": float(images["antipode_difference"].mean()),
                "image_sd_antipode_difference": float(images["antipode_difference"].std(ddof=1)),
                "n_observers": int(len(observers)),
                "observer_mean_radius_matched_difference": float(observers["radius_matched_difference"].mean()),
                "observer_sd_radius_matched_difference": float(observers["radius_matched_difference"].std(ddof=1)),
                "observer_mean_antipode_difference": float(observers["antipode_difference"].mean()),
                "observer_sd_antipode_difference": float(observers["antipode_difference"].std(ddof=1)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "visual_feature_summary.csv", index=False)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    positions = np.arange(len(FEATURES))
    axis.errorbar(
        positions - 0.08,
        summary["image_mean_radius_matched_difference"],
        yerr=summary["image_sd_radius_matched_difference"],
        fmt="o",
        color="#1f4e79",
        label="Fixation minus radius-matched",
        capsize=3,
    )
    axis.errorbar(
        positions + 0.08,
        summary["image_mean_antipode_difference"],
        yerr=summary["image_sd_antipode_difference"],
        fmt="s",
        color="#b85c38",
        label="Fixation minus antipode",
        capsize=3,
    )
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(FEATURES, rotation=20, ha="right")
    axis.set_ylabel("Image-mean difference (pixel units)")
    axis.set_title("MIT1003 tune images: fixation versus same-radius controls")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "visual_feature_controls.png", dpi=140)
    plt.close(figure)
    report = {
        "partition": "tune",
        "final_partition_opened": False,
        "window_radius_px": RADIUS,
        "seed": SEED,
        "n_images": int(alignment_frame["image_stem"].nunique()),
        "n_fixations": int(alignment_frame["n_fixations"].sum()),
        "n_inside": int(alignment_frame["n_inside"].sum()),
        "n_outside": int(alignment_frame["n_outside"].sum()),
        "object_descriptors": "not computed; the local ALLSTIMULI release has no object masks or boxes",
        "visual_angle": "not applied; MIT1003 pixels per degree remains unresolved",
        "center": "canvas center is width/2, height/2 of the JPEG, not a reconstructed screen center",
    }
    (OUTPUT / "visual_feature_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
