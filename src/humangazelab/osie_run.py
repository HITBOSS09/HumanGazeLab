"""Within-OSIE replication of the temporal density evaluation.

Densities are fit on OSIE images only. COCO-FreeView fixations are not used.
A second block refits OSIE densities with the pixel bandwidths and mixture
weights that were selected on COCO-FreeView. That block is labeled frozen
bandwidth transfer, not zero-shot transfer of a COCO density.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from humangazelab.osie import load_osie_scanpaths
from humangazelab.temporal import (
    Grid,
    accumulate_counts,
    aggregate_ig,
    blur_displacement,
    bootstrap_groups,
    canvas_kernel_mass,
    density_from_counts,
    derangement,
    displacement_histogram,
    evaluate_records,
    file_sha256,
    pack_scanpaths,
    sample_counts,
    select_center_sigma,
    select_history_sigma,
    select_lambdas,
    select_loo_sigma,
    sigma_in_bins,
    _group_records,
)
from humangazelab.temporal import load_temporal_config as load_config
from humangazelab.temporal_run import _paired_epoch


IG_COLUMNS = [
    "ig_B0_uniform",
    "ig_B1_center",
    "ig_B1_subject_held_out",
    "ig_B2_loo_image",
    "ig_B3_loo_index",
    "ig_B4_history",
    "ig_B5_center_history",
    "ig_B5c_loo_history",
    "ig_B6_image_shuffle",
    "ig_B6_order_shuffle",
]


def _fit_and_score(records, grid, alpha, sigma_center, sigma_history, sigma_loo, lambda_b5, lambda_b5c, shuffle_map):
    center = density_from_counts(
        accumulate_counts(records, lambda r: r.split == "train", 0, None, grid),
        sigma_in_bins(sigma_center, grid),
        alpha,
    )
    n_train = sum(record.length for record in records if record.split == "train")
    if int(accumulate_counts(records, lambda r: r.split == "train", 0, None, grid).sum()) != n_train:
        raise AssertionError("OSIE train counts do not match the train scanpaths")
    subjects = sorted({record.subject for record in records})
    center_by_subject = {}
    for subject in subjects:
        counts = accumulate_counts(records, lambda r, held=subject: r.split == "train" and r.subject != held, 0, None, grid)
        center_by_subject[subject] = density_from_counts(counts, sigma_in_bins(sigma_center, grid), alpha)
    kernel = blur_displacement(
        displacement_histogram(records, lambda r: r.split == "train", grid),
        sigma_in_bins(sigma_history, grid),
    )
    frame = evaluate_records(
        records,
        grid,
        center,
        center_by_subject,
        sigma_in_bins(sigma_loo, grid),
        kernel,
        canvas_kernel_mass(kernel, grid),
        alpha,
        lambda_b5,
        lambda_b5c,
        shuffle_map,
        index_min=0,
    )
    if frame.loc[frame["fixation_index"] == 0, "ig_B4_history"].notna().any():
        raise AssertionError("history was scored without a previous fixation")
    return frame


def _summarize(frame, cfg, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_boot = int(cfg["uncertainty"]["n_bootstrap"])
    seed = int(cfg["uncertainty"]["bootstrap_seed"])
    valid = frame["split"].eq("valid").to_numpy()
    index = frame["fixation_index"].to_numpy()
    temporal = cfg["temporal"]
    balanced = valid & frame["length"].ge(int(temporal["balanced_min_length"])).to_numpy()
    masks = {
        f"{label}|valid|all_released": valid,
        f"{label}|valid|early_indices_0_to_2": valid & (index >= temporal["early_start"]) & (index <= temporal["early_end"]),
        f"{label}|valid|late_indices_6_and_later": valid & (index >= temporal["late_start"]),
        f"{label}|valid_balanced|early_indices_0_to_2": balanced & (index >= temporal["early_start"]) & (index <= temporal["early_end"]),
        f"{label}|valid_balanced|late_indices_6_and_later": balanced & (index >= temporal["late_start"]),
    }
    for fixation_index in range(0, int(temporal["index_curve_max"]) + 1):
        masks[f"{label}|valid|index_{fixation_index}"] = valid & (index == fixation_index)
    rows = []
    for subset, mask in masks.items():
        for column in IG_COLUMNS:
            rows.append(aggregate_ig(frame, column, mask, n_boot, seed, subset, "subject_scanpath_equal"))
    contrasts = []
    specs = [
        ("ig_B1_center", "ig_B0_uniform", valid),
        ("ig_B2_loo_image", "ig_B1_center", valid),
        ("ig_B4_history", "ig_B1_center", valid),
        ("ig_B2_loo_image", "ig_B6_image_shuffle", valid),
        ("ig_B4_history", "ig_B6_order_shuffle", valid & ~frame["order_control_degenerate"].to_numpy()),
        ("ig_B3_loo_index", "ig_B2_loo_image", valid),
    ]
    for left, right, mask in specs:
        part = frame.loc[mask].copy()
        part["diff"] = part[left] - part[right]
        per_subject = part.groupby(["subject", "scan_id"])["diff"].mean().groupby("subject").mean()
        summary = bootstrap_groups(
            per_subject.to_numpy(dtype=float),
            f"{left}_minus_{right}",
            f"{label}|valid",
            "subject_scanpath_equal",
            n_boot,
            seed,
        )
        summary["contrast"] = f"{left} - {right}"
        summary["n_fixations"] = int(np.isfinite(part["diff"]).sum())
        contrasts.append(summary)
    early = balanced & (index >= temporal["early_start"]) & (index <= temporal["early_end"])
    late = balanced & (index >= temporal["late_start"])
    for column in ("ig_B1_center", "ig_B2_loo_image", "ig_B4_history"):
        contrasts.append(
            _paired_epoch(frame, column, early, late, n_boot, seed, f"{label}|valid_balanced|early_minus_late", f"{column}_early_minus_late")
        )
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    adapted = summary[summary["subset"].str.startswith("adapted|valid|index_")]
    fig, ax = plt.subplots(figsize=(8.8, 4.4), constrained_layout=True)
    for metric, color in (
        ("ig_B1_center", "#d95f02"),
        ("ig_B2_loo_image", "#1b9e77"),
        ("ig_B4_history", "#7570b3"),
        ("ig_B3_loo_index", "#66a61e"),
    ):
        rows = adapted[adapted["metric"] == metric].copy()
        rows["index"] = rows["subset"].str.extract(r"index_(\d+)$").astype(float)
        rows = rows.sort_values("index")
        ax.plot(rows["index"], rows["estimate"], color=color, label=metric.removeprefix("ig_"))
        ax.fill_between(rows["index"], rows["ci_low"], rows["ci_high"], color=color, alpha=0.15, linewidth=0)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Released OSIE fixation index")
    ax.set_ylabel("Information gain (bits per fixation)")
    ax.set_title("OSIE held-out images, adapted bandwidths")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(figure_dir / "osie_temporal_information_gain.png", dpi=140)
    plt.close(fig)
    (figure_dir / "captions.txt").write_text(
        "osie_temporal_information_gain.png\n"
        "Held-out OSIE images under bandwidths selected on other OSIE images. "
        "Subject-equal means of the 15 file-order subject indices. "
        "Index 0 is a released fixation, not a labeled forced fixation. "
        "History is undefined at index 0, so its curve starts at index 1. "
        "These indices are not aligned with COCO-FreeView indices.\n"
    )


def run(root: str | Path, config_path: str | Path = "configs/osie_v1.yaml") -> dict:
    root = Path(root)
    cfg = load_config(root / config_path)
    output_dir = root / cfg["output_dir"]
    if output_dir.resolve() == (root / "outputs" / "temporal_v1").resolve():
        raise RuntimeError("refusing to write OSIE results into the COCO temporal directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = Grid(cfg["canvas"]["width_px"], cfg["canvas"]["height_px"], cfg["grid"]["nx"], cfg["grid"]["ny"])
    role_kwargs = {
        "train_name": "train",
        "modulus": int(cfg["splits"]["role_modulus"]),
        "bandwidth_bucket": int(cfg["splits"]["bandwidth_bucket"]),
        "mixture_bucket": int(cfg["splits"]["mixture_bucket"]),
    }
    scanpaths = load_osie_scanpaths(root / cfg["fixation_path"])
    outside = [
        scan.image_name
        for scan in scanpaths
        if np.any(scan.x < 0) or np.any(scan.y < 0) or np.any(scan.x > grid.width) or np.any(scan.y > grid.height)
    ]
    if outside:
        raise ValueError(f"OSIE coordinates fall outside the 800 by 600 canvas, for example {outside[0]}")
    records = pack_scanpaths(scanpaths, grid, role_kwargs)
    if any(record.role != "eval" and record.split == "valid" for record in records):
        raise AssertionError("an OSIE held-out image was given a fitting role")
    alpha = float(cfg["models"]["alpha_primary"])
    candidates = [float(v) for v in cfg["models"]["sigma_candidates_px"]]
    lambdas = [float(v) for v in cfg["models"]["lambda_candidates"]]
    grouped = _group_records(records)
    sigma_center, rows_c = select_center_sigma(records, grid, candidates, alpha, 0)
    sigma_history, rows_h = select_history_sigma(records, grid, candidates, alpha)
    sigma_loo, rows_l = select_loo_sigma(grouped, grid, candidates, alpha, 0)
    fit_center = density_from_counts(
        accumulate_counts(records, lambda r: r.role == "fit", 0, None, grid),
        sigma_in_bins(sigma_center, grid),
        alpha,
    )
    fit_kernel = blur_displacement(
        displacement_histogram(records, lambda r: r.role == "fit", grid),
        sigma_in_bins(sigma_history, grid),
    )
    lambda_b5, lambda_b5c, rows_lambda = select_lambdas(
        records,
        grouped,
        grid,
        fit_center,
        fit_kernel,
        canvas_kernel_mass(fit_kernel, grid),
        sigma_in_bins(sigma_loo, grid),
        alpha,
        lambdas,
        index_min=0,
    )
    selection = pd.DataFrame(rows_c + rows_h + rows_l + rows_lambda)
    by_split: dict[str, list[str]] = {}
    for record in records:
        by_split.setdefault(record.split, []).append(record.image)
    shuffle_map: dict[str, str] = {}
    for split, names in by_split.items():
        shuffle_map.update(derangement(sorted(set(names)), int(cfg["uncertainty"]["shuffle_seed"]) + (0 if split == "train" else 1)))
    adapted = _fit_and_score(records, grid, alpha, sigma_center, sigma_history, sigma_loo, lambda_b5, lambda_b5c, shuffle_map)
    frozen = _fit_and_score(
        records,
        grid,
        alpha,
        float(cfg["models"]["coco_frozen_sigma_center_px"]),
        float(cfg["models"]["coco_frozen_sigma_history_px"]),
        float(cfg["models"]["coco_frozen_sigma_loo_px"]),
        float(cfg["models"]["coco_frozen_lambda_center_history"]),
        float(cfg["models"]["coco_frozen_lambda_loo_history"]),
        shuffle_map,
    )
    summary_a, contrast_a = _summarize(adapted, cfg, "adapted")
    summary_f, contrast_f = _summarize(frozen, cfg, "frozen_coco_bandwidth")
    summary = pd.concat([summary_a, summary_f], ignore_index=True)
    contrasts = pd.concat([contrast_a, contrast_f], ignore_index=True)
    selection.to_csv(output_dir / "hyperparameter_selection.csv", index=False)
    summary.to_csv(output_dir / "information_gain_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_contrasts.csv", index=False)
    sample_counts(records, int(cfg["temporal"]["index_curve_max"])).to_csv(output_dir / "sample_counts_by_index.csv", index=False)
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "fixation_sha256": file_sha256(root / cfg["fixation_path"]),
        "n_scanpaths": len(records),
        "n_subjects": len({record.subject for record in records}),
        "n_images": len({record.image for record in records}),
        "n_valid_images": len({record.image for record in records if record.split == "valid"}),
        "selected_sigma_center_px": sigma_center,
        "selected_sigma_history_px": sigma_history,
        "selected_sigma_loo_px": sigma_loo,
        "selected_lambda_center_history": lambda_b5,
        "selected_lambda_loo_history": lambda_b5c,
        "primary_includes_index_0": True,
        "history_defined_at_index_0": False,
        "pooled_with_coco_freeview": False,
        "stimuli_downloaded": False,
        "duration_column_used": False,
        "hypothesis_adopted": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _plot(summary, output_dir)
    return manifest
