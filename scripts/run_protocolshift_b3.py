"""Exploratory controls for the COCO-FreeView and OSIE index-conditioned densities.

Does not overwrite frozen experiment directories. Validation and held-out
outcomes from those experiments have already been inspected, so every number
written here is exploratory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from humangazelab.adapter import load_scanpaths
from humangazelab.osie import load_osie_scanpaths
from humangazelab.protocolshift import (
    assert_new_output,
    confident_but_poor,
    history_transfer_ig,
    image_equal_ig,
    keep_subjects,
    observer_spread,
    scale_scanpaths,
    score_index_density,
    seeded_subject_subsets,
    subject_equal_ig,
    width_fraction_to_sigma_px,
)
from humangazelab.summarize import metric_seed, percentile_bootstrap_mean
from humangazelab.temporal import Grid, pack_scanpaths, scanpath_then_subject_means, sigma_in_bins

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "protocolshift_b3_v1"
ROLE = {"train_name": "train", "modulus": 5, "bandwidth_bucket": 0, "mixture_bucket": 1}
SEED = 20261114
ALPHA = 0.05


def _pack(scanpaths, grid):
    return pack_scanpaths(scanpaths, grid, ROLE)


def _row(dataset, control, frame, **extra):
    part = frame[frame["split"].eq("valid")]
    values = scanpath_then_subject_means(part["scan_id"].to_numpy(), part["subject"].to_numpy(), part["ig"].to_numpy())
    summary = percentile_bootstrap_mean(values, 2000, metric_seed(SEED, dataset, control, "subject"))
    image = image_equal_ig(frame)
    summary.update({
        "dataset": dataset,
        "control": control,
        "estimand": "subject_scanpath_equal",
        "n_fixations": int(np.isfinite(part["ig"].to_numpy()).sum()),
        "n_scanpaths": int(part["scan_id"].nunique()) if len(part) else 0,
        "n_images": image["n_images"],
        "image_equal_estimate": image["estimate"],
        "image_sd": image["image_sd"],
        "exploratory": True,
        "previously_inspected_split": True,
    })
    summary.update(extra)
    return summary


def _score(scanpaths, grid, sigma_px, index_min, index_max=None, diagnostics=False):
    records = _pack(scanpaths, grid)
    return score_index_density(
        records, grid, sigma_in_bins(sigma_px, grid), ALPHA, index_min, index_max, diagnostics
    )


def plot_controls(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = pd.read_csv(out / "b3_controls.csv")
    labels = ["Frozen native", "Indices 1–8", "Sigma 24 px", "Sigma 40 px", "Sigma 64 px", "Width fraction 0.03", "Joint control"]
    coco_controls = [
        "frozen_native_index_min_1_sigma40", "indices_1_to_8_sigma40", "sigma_24_px_index_min_1",
        "sigma_40_px_index_min_1", "sigma_64_px_index_min_1",
        "unit_square_width_fraction_0.030000_index_1_to_end", "joint_index_1_to_8_width_fraction_0.03",
    ]
    osie_controls = [
        "frozen_native_index_min_0_sigma24", "indices_1_to_8_sigma24", "sigma_24_px_index_min_0",
        "sigma_40_px_index_min_0", "sigma_64_px_index_min_0",
        "unit_square_width_fraction_0.030000_index_0_to_end", "joint_mean_of_20_seeded_ten_observer_subsets",
    ]

    def grab(dataset, control):
        hit = table[(table["dataset"] == dataset) & (table["control"] == control)]
        return float(hit["estimate"].iloc[0])

    coco_y = [grab("COCO-FreeView", name) for name in coco_controls]
    osie_y = [grab("OSIE", name) for name in osie_controls]
    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    positions = list(range(len(labels)))
    ax.bar([p - 0.18 for p in positions], coco_y, width=0.36, label="COCO-FreeView", color="#d95f02")
    ax.bar([p + 0.18 for p in positions], osie_y, width=0.36, label="OSIE", color="#1b9e77")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel("B3 information gain (bits)")
    ax.set_title("Exploratory index-conditioned scores under predeclared controls")
    ax.legend(frameon=False)
    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "b3_control_comparison.png", dpi=140)
    plt.close(fig)


def main() -> None:
    assert_new_output(ROOT, OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    coco = load_scanpaths(ROOT / "data" / "raw" / "COCOFreeView_fixations_trainval.json")
    osie = load_osie_scanpaths(ROOT / "data" / "raw" / "external" / "osie_fixations.mat")
    coco_grid = Grid(1680, 1050, 42, 26)
    osie_grid = Grid(800, 600, 20, 15)
    published = {
        "COCO-FreeView": float(pd.read_csv(ROOT / "outputs" / "temporal_v1" / "information_gain_summary.csv").query(
            "metric=='ig_B3_loo_index' and subset=='valid|all_primary_indices' and estimand=='subject_scanpath_equal'"
        )["estimate"].iloc[0]),
        "OSIE": float(pd.read_csv(ROOT / "outputs" / "osie_v1" / "information_gain_summary.csv").query(
            "metric=='ig_B3_loo_index' and subset=='adapted|valid|all_released' and estimand=='subject_scanpath_equal'"
        )["estimate"].iloc[0]),
    }

    coco_frozen = _score(coco, coco_grid, 40.0, 1, diagnostics=True)
    osie_frozen = _score(osie, osie_grid, 24.0, 0, diagnostics=True)
    checksum = []
    for name, frame, published_value in (
        ("COCO-FreeView", coco_frozen, published["COCO-FreeView"]),
        ("OSIE", osie_frozen, published["OSIE"]),
    ):
        fresh = subject_equal_ig(frame)["estimate"]
        checksum.append({
            "dataset": name,
            "published_b3": published_value,
            "recomputed_b3": fresh,
            "absolute_difference": abs(fresh - published_value),
            "match": abs(fresh - published_value) < 1e-9,
        })
    pd.DataFrame(checksum).to_csv(OUT / "checksum.csv", index=False)
    if not all(row["match"] for row in checksum):
        raise SystemExit("frozen B3 checksum failed; controls were not interpreted")

    rows = [
        _row("COCO-FreeView", "frozen_native_index_min_1_sigma40", coco_frozen),
        _row("OSIE", "frozen_native_index_min_0_sigma24", osie_frozen),
        _row("COCO-FreeView", "include_index_0_sigma40", _score(coco, coco_grid, 40.0, 0)),
        _row("OSIE", "exclude_index_0_sigma24", _score(osie, osie_grid, 24.0, 1)),
        _row("COCO-FreeView", "indices_1_to_8_sigma40", _score(coco, coco_grid, 40.0, 1, 8)),
        _row("OSIE", "indices_1_to_8_sigma24", _score(osie, osie_grid, 24.0, 1, 8)),
    ]
    for sigma in (16.0, 24.0, 40.0, 64.0):
        rows.append(_row("COCO-FreeView", f"sigma_{int(sigma)}_px_index_min_1", _score(coco, coco_grid, sigma, 1)))
        rows.append(_row("OSIE", f"sigma_{int(sigma)}_px_index_min_0", _score(osie, osie_grid, sigma, 0)))
    for alpha in (0.01, 0.10):
        for name, scans, grid, sigma, index_min in (
            ("COCO-FreeView", coco, coco_grid, 40.0, 1),
            ("OSIE", osie, osie_grid, 24.0, 0),
        ):
            frame = score_index_density(
                _pack(scans, grid), grid, sigma_in_bins(sigma, grid), alpha, index_min, None, False
            )
            rows.append(_row(name, f"alpha_{alpha}_frozen_sigma", frame, alpha=alpha))

    unit = Grid(1.0, 1.0, 20, 15)
    coco_unit = scale_scanpaths(coco, 1680, 1050)
    osie_unit = scale_scanpaths(osie, 800, 600)
    for fraction in (24.0 / 800.0, 40.0 / 1680.0):
        sigma_px = width_fraction_to_sigma_px(fraction, unit)
        label = f"unit_square_width_fraction_{fraction:.6f}"
        rows.append(_row("COCO-FreeView", label + "_index_1_to_end", _score(coco_unit, unit, sigma_px, 1)))
        rows.append(_row("OSIE", label + "_index_0_to_end", _score(osie_unit, unit, sigma_px, 0)))
        rows.append(_row("COCO-FreeView", label + "_index_1_to_8", _score(coco_unit, unit, sigma_px, 1, 8)))
        rows.append(_row("OSIE", label + "_index_1_to_8", _score(osie_unit, unit, sigma_px, 1, 8)))

    osie_subjects = sorted({int(scan.subject_id) for scan in osie})
    seeded = seeded_subject_subsets(osie_subjects, 10, 20, SEED)
    fixed = tuple(range(10))
    subset_rows = []
    joint_estimates = []
    fraction = 24.0 / 800.0
    joint_sigma = width_fraction_to_sigma_px(fraction, unit)

    def _subset_block(label, subjects):
        kept = keep_subjects(osie, set(subjects))
        native = _score(kept, osie_grid, 24.0, 0)
        subset_rows.append(_row("OSIE", f"{label}_native_index_0", native, subjects=",".join(map(str, subjects))))
        window = _score(kept, osie_grid, 24.0, 1, 8)
        subset_rows.append(_row("OSIE", f"{label}_indices_1_to_8", window, subjects=",".join(map(str, subjects))))
        joint = _score(keep_subjects(osie_unit, set(subjects)), unit, joint_sigma, 1, 8)
        joint_row = _row("OSIE", f"{label}_joint", joint, subjects=",".join(map(str, subjects)))
        subset_rows.append(joint_row)
        return joint_row["estimate"]

    fixed_joint = _subset_block("fixed_slots_0_to_9", fixed)
    for index, subjects in enumerate(seeded):
        joint_estimates.append(_subset_block(f"seeded_{index}", subjects))
    pd.DataFrame(subset_rows).to_csv(OUT / "osie_observer_subsets.csv", index=False)

    coco_joint = _score(coco_unit, unit, joint_sigma, 1, 8)
    rows.append(_row("COCO-FreeView", "joint_index_1_to_8_width_fraction_0.03", coco_joint))
    rows.append({
        "dataset": "OSIE",
        "control": "fixed_slots_0_to_9_joint",
        "estimate": fixed_joint,
        "exploratory": True,
        "note": "File positions 0-9 only. One predeclared subset, not chosen after seeing its score.",
    })
    rows.append({
        "dataset": "OSIE",
        "control": "joint_mean_of_20_seeded_ten_observer_subsets",
        "estimate": float(np.mean(joint_estimates)),
        "min_across_subsets": float(np.min(joint_estimates)),
        "max_across_subsets": float(np.max(joint_estimates)),
        "n_subsets": len(joint_estimates),
        "exploratory": True,
        "note": "Mean of subject-equal B3 across 20 seeded 10-observer subsets. Not a subject bootstrap.",
    })
    pd.DataFrame(rows).to_csv(OUT / "b3_controls.csv", index=False)

    diagnostics = []
    for name, frame in (("COCO-FreeView", coco_frozen), ("OSIE", osie_frozen)):
        item = confident_but_poor(frame, 0.2, 0.0)
        item["dataset"] = name
        item["model"] = "B3_index_conditioned"
        diagnostics.append(item)
    pd.DataFrame(diagnostics).to_csv(OUT / "confident_but_poor.csv", index=False)

    entropy_rows = []
    for name, frame in (("COCO-FreeView", coco_frozen), ("OSIE", osie_frozen)):
        part = frame[frame["split"].eq("valid")]
        for index, group in part.groupby("fixation_index"):
            entropy_rows.append({
                "dataset": name,
                "fixation_index": int(index),
                "n_fixations": int(len(group)),
                "mean_ig": float(group["ig"].mean()),
                "mean_entropy_bits": float(group["entropy_bits"].mean()),
                "fraction_inside_half_mass": float(np.nanmean(group["in_half_mass_region"].to_numpy(dtype=float))),
                "mean_n_other": float(group["n_other"].mean()),
                "fraction_ig_negative": float((group["ig"] < 0).mean()),
            })
    pd.DataFrame(entropy_rows).to_csv(OUT / "entropy_by_index.csv", index=False)

    spread_rows = []
    for name, scans, width, height, index_min, index_max in (
        ("COCO-FreeView", coco, 1680, 1050, 1, 15),
        ("OSIE", osie, 800, 600, 0, 8),
    ):
        spread = observer_spread(scans, index_min, index_max, width, height, "valid")
        if spread.empty:
            continue
        grouped = spread.groupby("fixation_index").agg(
            n_images=("image", "nunique"),
            mean_n_observers=("n_observers", "mean"),
            mean_sd_x_px=("sd_x_px", "mean"),
            mean_sd_y_px=("sd_y_px", "mean"),
            mean_sd_x_fraction_of_width=("sd_x_fraction_of_width", "mean"),
            mean_sd_y_fraction_of_height=("sd_y_fraction_of_height", "mean"),
        ).reset_index()
        grouped.insert(0, "dataset", name)
        spread_rows.append(grouped)
    pd.concat(spread_rows, ignore_index=True).to_csv(OUT / "observer_spread_by_index.csv", index=False)

    transfer = []
    coco_records = _pack(coco_unit, unit)
    osie_records = _pack(osie_unit, unit)
    sigma_bins = sigma_in_bins(joint_sigma, unit)
    for source_name, source in (("COCO-FreeView", coco_records), ("OSIE", osie_records)):
        for target_name, target in (("COCO-FreeView", coco_records), ("OSIE", osie_records)):
            scored = history_transfer_ig(source, target, unit, sigma_bins, ALPHA)
            scored.update({
                "source": source_name,
                "target": target_name,
                "same_dataset": source_name == target_name,
                "coordinate_frame": "unit_square",
                "sigma_width_fraction": fraction,
                "uses_target_validation_for_fit": False,
                "visual_angle_normalized": False,
                "exploratory": True,
            })
            transfer.append(scored)
    pd.DataFrame(transfer).to_csv(OUT / "history_transfer.csv", index=False)

    manifest = {
        "experiment_id": "protocolshift_b3_v1",
        "exploratory": True,
        "confirmatory": False,
        "reason": "COCO-FreeView validation and OSIE held-out scores were inspected in temporal_v1 and osie_v1.",
        "visual_angle_normalization": False,
        "coordinate_normalization": "released pixels divided by released canvas width and height",
        "alpha": ALPHA,
        "seed": SEED,
        "osie_subset_size": 10,
        "n_seeded_subsets_requested": 20,
        "joint_index": [1, 8],
        "joint_width_fraction": fraction,
        "checksum": checksum,
        "frozen_outputs_rewritten": False,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_controls(OUT)
    print(json.dumps({"checksum": checksum, "n_controls": len(rows), "n_subset_rows": len(subset_rows)}, indent=2))


if __name__ == "__main__":
    main()
