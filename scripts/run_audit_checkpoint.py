"""Recompute frozen temporal scores and write the validation checkpoint.

Does not write outputs/temporal_v1, outputs/osie_v1, outputs/figures, or outputs/tables.
The temporal_v2 bandwidth table is scored on training selection folds only.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from humangazelab.adapter import load_scanpaths
from humangazelab.audit_checkpoint import (
    EXTENDED_SIGMAS,
    assert_not_frozen,
    b3_diagnostic,
    checksum_against_published,
    overlay_osie,
    plot_length_curves,
    reference_geometry,
    survival_contrasts,
    train_bandwidth_curve,
    uncertainty_rows,
    _fit_frozen,
    _shuffle,
)
from humangazelab.osie import load_osie_scanpaths
from humangazelab.temporal import Grid, pack_scanpaths

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "audit_checkpoint"
V2 = ROOT / "outputs" / "temporal_v2"
SEED = 20261114
N_BOOT = 2000
ROLE = {
    "train_name": "train",
    "modulus": 5,
    "bandwidth_bucket": 0,
    "mixture_bucket": 1,
}
METRICS = (
    "ig_B0_uniform",
    "ig_B1_center",
    "ig_B2_loo_image",
    "ig_B3_loo_index",
    "ig_B4_history",
    "ig_B5_center_history",
    "ig_B5c_loo_history",
)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, path)


def _score(scanpaths, grid, manifest, index_min, seed):
    records = pack_scanpaths(scanpaths, grid, ROLE)
    frame = _fit_frozen(
        records,
        grid,
        float(manifest["alpha_primary"]) if "alpha_primary" in manifest else 0.05,
        float(manifest["selected_sigma_center_px"]),
        float(manifest["selected_sigma_history_px"]),
        float(manifest["selected_sigma_loo_px"]),
        float(manifest["selected_lambda_center_history"]),
        float(manifest["selected_lambda_loo_history"]),
        _shuffle(records, seed),
        index_min,
    )
    return records, frame


def main() -> None:
    assert_not_frozen(ROOT, AUDIT)
    assert_not_frozen(ROOT, V2)
    AUDIT.mkdir(parents=True, exist_ok=True)
    V2.mkdir(parents=True, exist_ok=True)
    (AUDIT / "figures").mkdir(exist_ok=True)

    coco_manifest = json.loads((ROOT / "outputs" / "temporal_v1" / "run_manifest.json").read_text())
    coco_manifest["alpha_primary"] = 0.05
    coco_published = pd.read_csv(ROOT / "outputs" / "temporal_v1" / "information_gain_summary.csv")
    coco_scans = load_scanpaths(ROOT / "data" / "raw" / "COCOFreeView_fixations_trainval.json")
    coco_grid = Grid(1680, 1050, 42, 26)
    _, coco_frame = _score(coco_scans, coco_grid, coco_manifest, 1, SEED)
    coco_check = checksum_against_published(coco_frame, coco_published, "valid|all_primary_indices", METRICS)
    pd.DataFrame(coco_check).to_csv(AUDIT / "coco_checksum.csv", index=False)

    coco_valid = [scan for scan in coco_scans if scan.split == "valid"]
    coco_geometry = reference_geometry(coco_valid, 1)
    coco_b3 = b3_diagnostic(coco_frame, coco_geometry, 0.05, coco_grid.n_bins, "valid")
    coco_b3.to_csv(AUDIT / "coco_b3_by_index.csv", index=False)
    coco_length = pd.DataFrame(survival_contrasts(coco_frame, "valid", (1, 3), 9, N_BOOT, SEED))
    # Per-index curves are written from the subject-equal helper inside length_rows.
    from humangazelab.audit_checkpoint import length_rows

    coco_curve = pd.DataFrame(length_rows(coco_frame, 1, 15, N_BOOT, SEED, "valid"))
    coco_curve.to_csv(AUDIT / "coco_length_sensitivity.csv", index=False)
    coco_length.to_csv(AUDIT / "coco_survival_contrasts.csv", index=False)
    plot_length_curves(coco_curve, AUDIT / "figures" / "coco_b3_length_sensitivity.png", 1)
    coco_uncertainty = pd.DataFrame(uncertainty_rows(coco_frame, "valid", N_BOOT, SEED, "valid|all_primary_indices"))
    coco_uncertainty.insert(0, "dataset", "COCO-FreeView")
    coco_uncertainty.to_csv(AUDIT / "coco_uncertainty.csv", index=False)

    osie_manifest = json.loads((ROOT / "outputs" / "osie_v1" / "run_manifest.json").read_text())
    osie_manifest["alpha_primary"] = 0.05
    osie_published = pd.read_csv(ROOT / "outputs" / "osie_v1" / "information_gain_summary.csv")
    osie_scans = load_osie_scanpaths(ROOT / "data" / "raw" / "external" / "osie_fixations.mat")
    osie_grid = Grid(800, 600, 20, 15)
    osie_records, osie_frame = _score(osie_scans, osie_grid, osie_manifest, 0, SEED)
    osie_check = checksum_against_published(osie_frame, osie_published, "adapted|valid|all_released", METRICS)
    pd.DataFrame(osie_check).to_csv(AUDIT / "osie_checksum.csv", index=False)
    osie_valid = [scan for scan in osie_scans if scan.split == "valid"]
    osie_geometry = reference_geometry(osie_valid, 0)
    osie_b3 = b3_diagnostic(osie_frame, osie_geometry, 0.05, osie_grid.n_bins, "valid")
    osie_b3.to_csv(AUDIT / "osie_b3_by_index.csv", index=False)
    osie_length = pd.DataFrame(survival_contrasts(osie_frame, "valid", (0, 2), 6, N_BOOT, SEED))
    osie_length.to_csv(AUDIT / "osie_survival_contrasts.csv", index=False)
    osie_curve = pd.DataFrame(length_rows(osie_frame, 0, 8, N_BOOT, SEED, "valid"))
    osie_curve.to_csv(AUDIT / "osie_length_sensitivity.csv", index=False)
    plot_length_curves(osie_curve, AUDIT / "figures" / "osie_b3_length_sensitivity.png", 0)
    osie_uncertainty = pd.DataFrame(uncertainty_rows(osie_frame, "valid", N_BOOT, SEED, "adapted|valid|all_released"))
    osie_uncertainty.insert(0, "dataset", "OSIE")
    osie_uncertainty.to_csv(AUDIT / "osie_uncertainty.csv", index=False)

    # Train folds only. Validation scanpaths are removed before selection.
    coco_train = [scan for scan in coco_scans if scan.split == "train"]
    osie_train = [scan for scan in osie_scans if scan.split == "train"]
    coco_train_records = pack_scanpaths(coco_train, coco_grid, ROLE)
    osie_train_records = pack_scanpaths(osie_train, osie_grid, ROLE)
    if any(record.split != "train" for record in coco_train_records + osie_train_records):
        raise AssertionError("temporal_v2 received a held-out scanpath")
    bandwidth = pd.concat([
        train_bandwidth_curve(coco_train_records, coco_grid, 0.05, 1, "COCO-FreeView"),
        train_bandwidth_curve(osie_train_records, osie_grid, 0.05, 0, "OSIE"),
    ], ignore_index=True)
    bandwidth.to_csv(V2 / "train_fold_bandwidth.csv", index=False)
    argmax = {}
    for dataset, group in bandwidth.groupby("dataset"):
        first = group.iloc[0]
        argmax[str(dataset)] = {
            "sigma_center_px": float(first["selected_sigma_center_px"]),
            "sigma_history_px": float(first["selected_sigma_history_px"]),
            "sigma_loo_px": float(first["selected_sigma_loo_px"]),
            "lambda_center_history": float(first["selected_lambda_center_history"]),
            "lambda_loo_history": float(first["selected_lambda_loo_history"]),
        }
    v2_manifest = {
        "experiment_id": "temporal_v2",
        "purpose": "train-fold bandwidth curve below the temporal_v1 lower edge of 24 px",
        "replaces_temporal_v1": False,
        "validation_accessed": False,
        "scored_on_validation": False,
        "candidates_px": list(EXTENDED_SIGMAS),
        "alpha": 0.05,
        "lambda_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
        "train_fold_argmax": argmax,
        "sigma_loo_selection_target": "pooled leave-one-observer density from the primary index onward; that bandwidth is then shared with the index-conditioned map",
        "note": "Selected values are the argmax on the training selection fold. They were not scored on held-out images. On the 40 px grids, 4 px and 8 px are not distinguished for the center and history scores.",
    }
    (V2 / "run_manifest.json").write_text(json.dumps(v2_manifest, indent=2) + "\n")

    image_dir = ROOT / "data" / "raw" / "external" / "osie_stimuli"
    names = sorted({scan.image_name for scan in osie_scans})
    rng = np.random.default_rng(SEED)
    chosen = sorted(set(rng.choice(names, size=8, replace=False).tolist() + ["1001.jpg"]))
    base = "https://raw.githubusercontent.com/NUS-VIP/predicting-human-gaze-beyond-pixels/master/data/stimuli/"
    downloaded = []
    failed = []
    for name in chosen:
        try:
            _download(base + name, image_dir / name)
            downloaded.append(name)
        except Exception as exc:  # noqa: BLE001 - record the failure instead of hiding it
            failed.append({"image": name, "error": str(exc)})
    overlay = overlay_osie(osie_scans, image_dir, downloaded, AUDIT / "figures" / "osie_fixation_overlay.png")
    overlay.to_csv(AUDIT / "osie_overlay_geometry.csv", index=False)
    (AUDIT / "osie_overlay_failures.json").write_text(json.dumps(failed, indent=2) + "\n")

    status = {
        "temporal_v1_frozen": True,
        "osie_v1_frozen": True,
        "coco_checksum_all_match": all(row["match"] for row in coco_check),
        "osie_checksum_all_match": all(row["match"] for row in osie_check),
        "coco_b3_recomputed": next(row["recomputed_estimate"] for row in coco_check if row["metric"] == "ig_B3_loo_index"),
        "osie_b3_recomputed": next(row["recomputed_estimate"] for row in osie_check if row["metric"] == "ig_B3_loo_index"),
        "overlay_images_downloaded": downloaded,
        "overlay_failures": failed,
        "stimuli_pixel_size_checked_against_800x600": True,
        "zero_shot_coco_density_on_osie": False,
        "dataset_adapted_evaluation": True,
    }
    (AUDIT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
