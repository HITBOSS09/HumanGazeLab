"""Run the temporal_v1 fixation-density experiment and write versioned tables.

The baseline directories outputs/figures and outputs/tables are not written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from humangazelab.adapter import load_scanpaths
from humangazelab.temporal import (
    MODEL_NAMES,
    Grid,
    accumulate_counts,
    aggregate_ig,
    assert_loo_excludes_target,
    blur_displacement,
    bootstrap_groups,
    calibration_table,
    canvas_kernel_mass,
    config_hash,
    density_from_counts,
    derangement,
    displacement_histogram,
    evaluate_records,
    file_sha256,
    image_role,
    information_gain,
    load_temporal_config,
    pack_scanpaths,
    sample_counts,
    scanpath_then_subject_means,
    select_center_sigma,
    select_history_sigma,
    select_lambdas,
    select_loo_sigma,
    sigma_in_bins,
    spatial_reliability,
    _group_records,
)
from humangazelab.failure_atlas import write_atlas
from humangazelab.temporal_plot import write_figures


BASELINE_DIRS = ("outputs/figures", "outputs/tables")
IG_COLUMNS = [f"ig_{name}" for name in MODEL_NAMES]
NSS_COLUMNS = ["nss_B1_center", "nss_B2_loo_image", "nss_B4_history"]


def _epoch_masks(frame: pd.DataFrame, base: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    index = frame["fixation_index"].to_numpy()
    temporal = cfg["temporal"]
    masks = {
        "all_primary_indices": base,
        "early_indices_1_to_3": base & (index >= temporal["early_start"]) & (index <= temporal["early_end"]),
        "middle_indices_4_to_8": base & (index >= temporal["middle_start"]) & (index <= temporal["middle_end"]),
        "late_indices_9_and_later": base & (index >= temporal["late_start"]),
    }
    curve_max = int(temporal["index_curve_max"])
    for fixation_index in range(1, curve_max + 1):
        masks[f"index_{fixation_index}"] = base & (index == fixation_index)
    masks[f"index_{curve_max + 1}_and_later"] = base & (index >= curve_max + 1)
    return masks


def _paired_epoch(frame, column, mask_a, mask_b, n_boot, seed, subset, label):
    part_a = frame.loc[mask_a]
    part_b = frame.loc[mask_b]
    mean_a = part_a.groupby(["subject", "scan_id"])[column].mean().groupby("subject").mean()
    mean_b = part_b.groupby(["subject", "scan_id"])[column].mean().groupby("subject").mean()
    shared = mean_a.index.intersection(mean_b.index)
    diff = (mean_a.loc[shared] - mean_b.loc[shared]).to_numpy(dtype=float)
    summary = bootstrap_groups(diff, label, subset, "subject_scanpath_equal", n_boot, seed)
    summary["contrast"] = label
    summary["n_fixations"] = int(mask_a.sum() + mask_b.sum())
    summary["n_fixations_early"] = int(mask_a.sum())
    summary["n_fixations_late"] = int(mask_b.sum())
    summary["n_scanpaths"] = int(frame.loc[mask_a | mask_b, "scan_id"].nunique())
    return summary


def _remix_alpha(probability: np.ndarray, alpha_old: float, alpha_new: float, n_bins: int) -> np.ndarray:
    raw = (probability - alpha_old / n_bins) / (1.0 - alpha_old)
    raw = np.clip(raw, 0.0, None)
    mixed = (1.0 - alpha_new) * raw + alpha_new / n_bins
    return mixed


def _assert_output_dir(root: Path, output_dir: Path) -> None:
    resolved = output_dir.resolve()
    for relative in BASELINE_DIRS:
        if resolved == (root / relative).resolve():
            raise RuntimeError(f"refusing to write temporal results into baseline directory {relative}")
    if resolved == (root / "outputs").resolve():
        raise RuntimeError("temporal results must go in a versioned subdirectory of outputs/")


def run(root: str | Path, config_path: str | Path = "configs/temporal_v1.yaml") -> dict:
    root = Path(root)
    cfg = load_temporal_config(root / config_path)
    output_dir = root / cfg["output_dir"]
    _assert_output_dir(root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid(cfg["canvas"]["width_px"], cfg["canvas"]["height_px"], cfg["grid"]["nx"], cfg["grid"]["ny"])
    role_kwargs = {
        "train_name": cfg["splits"]["train_split_name"],
        "modulus": int(cfg["splits"]["role_modulus"]),
        "bandwidth_bucket": int(cfg["splits"]["bandwidth_bucket"]),
        "mixture_bucket": int(cfg["splits"]["mixture_bucket"]),
    }
    scanpaths = load_scanpaths(root / cfg["fixation_path"])
    records = pack_scanpaths(scanpaths, grid, role_kwargs)
    grouped = _group_records(records)
    checked = 0
    for name, group in grouped.items():
        assert_loo_excludes_target(group, grid, index_min=1)
        checked += 1
        if checked >= 25:
            break

    alpha = float(cfg["models"]["alpha_primary"])
    candidates = [float(v) for v in cfg["models"]["sigma_candidates_px"]]
    lambdas = [float(v) for v in cfg["models"]["lambda_candidates"]]
    index_min = int(cfg["temporal"]["primary_index_min"])
    if index_min != 1:
        raise ValueError("temporal_v1 primary_index_min is fixed at 1")

    sigma_center, rows_center = select_center_sigma(records, grid, candidates, alpha, index_min)
    sigma_history, rows_history = select_history_sigma(records, grid, candidates, alpha)
    sigma_loo, rows_loo = select_loo_sigma(grouped, grid, candidates, alpha, index_min)

    fit_center_counts = accumulate_counts(records, lambda r: r.role == "fit", index_min, None, grid)
    fit_center = density_from_counts(fit_center_counts, sigma_in_bins(sigma_center, grid), alpha)
    fit_hist = displacement_histogram(records, lambda r: r.role == "fit", grid)
    fit_kernel = blur_displacement(fit_hist, sigma_in_bins(sigma_history, grid))
    fit_mass = canvas_kernel_mass(fit_kernel, grid)
    lambda_b5, lambda_b5c, rows_lambda = select_lambdas(
        records, grouped, grid, fit_center, fit_kernel, fit_mass, sigma_in_bins(sigma_loo, grid), alpha, lambdas
    )
    selection = pd.DataFrame(rows_center + rows_history + rows_loo + rows_lambda)
    selection["selected"] = False
    selection.loc[(selection["parameter"] == "sigma_center_px") & (selection["candidate"] == sigma_center), "selected"] = True
    selection.loc[(selection["parameter"] == "sigma_history_px") & (selection["candidate"] == sigma_history), "selected"] = True
    selection.loc[(selection["parameter"] == "sigma_loo_px") & (selection["candidate"] == sigma_loo), "selected"] = True
    selection.loc[(selection["parameter"] == "lambda_center_history") & (selection["candidate"] == lambda_b5), "selected"] = True
    selection.loc[(selection["parameter"] == "lambda_loo_history") & (selection["candidate"] == lambda_b5c), "selected"] = True

    train_counts = accumulate_counts(records, lambda r: r.split == "train", index_min, None, grid)
    n_train_targets = sum(max(0, record.length - index_min) for record in records if record.split == "train")
    if int(train_counts.sum()) != n_train_targets:
        raise AssertionError("train center-bias counts include fixations outside the train split or the primary indices")
    center = density_from_counts(train_counts, sigma_in_bins(sigma_center, grid), alpha)
    subjects = sorted({record.subject for record in records})
    center_by_subject = {}
    for subject in subjects:
        counts = accumulate_counts(
            records,
            lambda r, held=subject: r.split == "train" and r.subject != held,
            index_min,
            None,
            grid,
        )
        center_by_subject[subject] = density_from_counts(counts, sigma_in_bins(sigma_center, grid), alpha)
    kernel = blur_displacement(
        displacement_histogram(records, lambda r: r.split == "train", grid),
        sigma_in_bins(sigma_history, grid),
    )
    kernel_mass = canvas_kernel_mass(kernel, grid)

    by_split: dict[str, list[str]] = {}
    for record in records:
        by_split.setdefault(record.split, []).append(record.image)
    shuffle_map: dict[str, str] = {}
    pair_rows = []
    for split, names in by_split.items():
        mapping = derangement(sorted(set(names)), int(cfg["uncertainty"]["shuffle_seed"]) + (0 if split == "train" else 1))
        if any(left == right for left, right in mapping.items()):
            raise AssertionError("image shuffle paired an image with itself")
        if set(mapping) != set(mapping.values()):
            raise AssertionError("image shuffle is not a permutation")
        shuffle_map.update(mapping)
        for left, right in mapping.items():
            pair_rows.append({"split": split, "image": left, "reference_image": right})

    frame = evaluate_records(
        records,
        grid,
        center,
        center_by_subject,
        sigma_in_bins(sigma_loo, grid),
        kernel,
        kernel_mass,
        alpha,
        lambda_b5,
        lambda_b5c,
        shuffle_map,
        index_min=index_min,
    )
    if frame["fixation_index"].min() < index_min:
        raise AssertionError("primary table contains a fixation index below the primary minimum")
    if set(frame.loc[frame["split"] != "train", "role"].unique()) != {"eval"}:
        raise AssertionError("a non-train image was assigned a fitting role")

    n_boot = int(cfg["uncertainty"]["n_bootstrap"])
    seed = int(cfg["uncertainty"]["bootstrap_seed"])
    summaries = []
    valid = frame["split"].eq(cfg["splits"]["eval_split_name"]).to_numpy()
    train = frame["split"].eq("train").to_numpy()
    balanced_length = int(cfg["temporal"]["balanced_min_length"])
    valid_balanced = valid & frame["length"].ge(balanced_length).to_numpy()
    nondegenerate = valid & ~frame["order_control_degenerate"].to_numpy()
    subset_masks = {
        "valid": valid,
        "valid_balanced_length": valid_balanced,
        "train_descriptive": train,
    }
    for subset, base in subset_masks.items():
        for epoch, mask in _epoch_masks(frame, base, cfg).items():
            for column in IG_COLUMNS + NSS_COLUMNS:
                # NSS is reported for the overall validation comparison and the index curve, not every train cell.
                if column.startswith("nss_") and not (subset == "valid" and epoch in {"all_primary_indices", "early_indices_1_to_3", "middle_indices_4_to_8", "late_indices_9_and_later"} or epoch.startswith("index_")):
                    continue
                if subset == "train_descriptive" and epoch not in {"all_primary_indices", "early_indices_1_to_3", "late_indices_9_and_later"} and not epoch.startswith("index_"):
                    continue
                estimand = "subject_scanpath_equal"
                summaries.append(aggregate_ig(frame, column, mask, n_boot, seed, f"{subset}|{epoch}", estimand))
            if subset == "valid" and epoch == "all_primary_indices":
                for column in ("ig_B1_center", "ig_B2_loo_image", "ig_B4_history", "ig_B5_center_history"):
                    summaries.append(aggregate_ig(frame, column, mask, n_boot, seed, f"{subset}|{epoch}", "subject_fixation_weighted"))
                    summaries.append(aggregate_ig(frame, column, mask, n_boot, seed, f"{subset}|{epoch}", "image"))

    contrasts = []
    contrast_specs = [
        ("ig_B1_center", "ig_B0_uniform", valid, "valid|all_primary_indices"),
        ("ig_B4_history", "ig_B1_center", valid, "valid|all_primary_indices"),
        ("ig_B5_center_history", "ig_B1_center", valid, "valid|all_primary_indices"),
        ("ig_B2_loo_image", "ig_B1_center", valid, "valid|all_primary_indices"),
        ("ig_B3_loo_index", "ig_B2_loo_image", valid, "valid|all_primary_indices"),
        ("ig_B5c_loo_history", "ig_B2_loo_image", valid, "valid|all_primary_indices"),
        ("ig_B2_loo_image", "ig_B6_image_shuffle", valid, "valid|all_primary_indices"),
        ("ig_B4_history", "ig_B6_order_shuffle", nondegenerate, "valid|order_control_nondegenerate"),
        ("ig_B1_subject_held_out", "ig_B0_uniform", valid, "valid|all_primary_indices"),
    ]
    for left, right, mask, subset in contrast_specs:
        part = frame.loc[mask].copy()
        part["diff"] = part[left] - part[right]
        per_subject = part.groupby(["subject", "scan_id"])["diff"].mean().groupby("subject").mean()
        summary = bootstrap_groups(per_subject.to_numpy(dtype=float), f"{left}_minus_{right}", subset, "subject_scanpath_equal", n_boot, seed)
        summary["contrast"] = f"{left} - {right}"
        summary["n_fixations"] = int(mask.sum())
        contrasts.append(summary)
        per_image = part.groupby("image")["diff"].mean()
        image_summary = bootstrap_groups(per_image.to_numpy(dtype=float), f"{left}_minus_{right}", subset, "image", n_boot, seed)
        image_summary["contrast"] = f"{left} - {right}"
        image_summary["n_fixations"] = int(mask.sum())
        contrasts.append(image_summary)

    temporal = cfg["temporal"]
    index = frame["fixation_index"].to_numpy()
    early = valid_balanced & (index >= temporal["early_start"]) & (index <= temporal["early_end"])
    late = valid_balanced & (index >= temporal["late_start"])
    for column in ("ig_B1_center", "ig_B2_loo_image", "ig_B4_history", "ig_B5_center_history"):
        contrasts.append(_paired_epoch(frame, column, early, late, n_boot, seed, "valid_balanced_length|early_minus_late", f"{column}_early_minus_late"))

    # Floor sensitivity from stored probabilities. This does not refit bandwidths.
    floor_rows = []
    n_bins = grid.n_bins
    valid_frame = frame.loc[valid]
    for alpha_new in [alpha, *[float(v) for v in cfg["models"]["alpha_sensitivity"]]]:
        for source, label in (("p_B1_center", "B1_center"), ("p_B2_loo_image", "B2_loo_image"), ("p_B4_history", "B4_history")):
            remixed = _remix_alpha(valid_frame[source].to_numpy(), alpha, alpha_new, n_bins)
            ig = information_gain(np.maximum(remixed, np.finfo(float).tiny), n_bins)
            values = scanpath_then_subject_means(valid_frame["scan_id"].to_numpy(), valid_frame["subject"].to_numpy(), ig)
            summary = bootstrap_groups(values, f"ig_{label}", f"valid|alpha_{alpha_new}", "subject_scanpath_equal", n_boot, seed)
            summary["alpha"] = alpha_new
            summary["model"] = label
            summary["primary_alpha"] = alpha_new == alpha
            floor_rows.append(summary)

    reliability = spatial_reliability(
        valid_frame["fixation_iy"].to_numpy(),
        valid_frame["fixation_ix"].to_numpy(),
        center,
    )
    probability_quantiles = pd.concat(
        [
            calibration_table(valid_frame, "p_B1_center"),
            calibration_table(valid_frame, "p_B2_loo_image"),
            calibration_table(valid_frame, "p_B4_history"),
        ],
        ignore_index=True,
    )

    subject_rows = []
    for (subject, model), values in valid_frame.melt(
        id_vars=["subject", "scan_id"], value_vars=IG_COLUMNS, var_name="model", value_name="ig"
    ).groupby(["subject", "model"]):
        per_scan = values.groupby("scan_id")["ig"].mean()
        subject_rows.append({"subject": int(subject), "model": model, "mean_ig_bits": float(per_scan.mean()), "n_scanpaths": int(per_scan.size)})
    subject_table = pd.DataFrame(subject_rows)

    image_table = (
        valid_frame.groupby("image")[IG_COLUMNS]
        .mean()
        .reset_index()
        .rename(columns={column: column for column in IG_COLUMNS})
    )
    coverage = valid_frame.groupby("image").agg(n_subjects=("subject", "nunique"), n_fixations=("fixation_index", "size"), median_length=("length", "median")).reset_index()
    image_table = image_table.merge(coverage, on="image")

    counts = sample_counts(records, int(cfg["temporal"]["index_curve_max"]))
    index0_rows = _index0_sensitivity(records, grid, center, sigma_center, sigma_loo, alpha, n_boot, seed, cfg)

    selection.to_csv(output_dir / "hyperparameter_selection.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "information_gain_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output_dir / "paired_contrasts.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(output_dir / "floor_sensitivity.csv", index=False)
    reliability.to_csv(output_dir / "center_bias_spatial_reliability.csv", index=False)
    probability_quantiles.to_csv(output_dir / "predicted_probability_quantiles.csv", index=False)
    subject_table.to_csv(output_dir / "subject_information_gain_valid.csv", index=False)
    image_table.to_csv(output_dir / "image_information_gain_valid.csv", index=False)
    counts.to_csv(output_dir / "sample_counts_by_index.csv", index=False)
    pd.DataFrame(index0_rows).to_csv(output_dir / "index0_sensitivity.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "image_shuffle_pairs.csv", index=False)

    manifest = {
        "experiment_id": cfg["experiment_id"],
        "config_sha256": config_hash(cfg),
        "fixation_sha256": file_sha256(root / cfg["fixation_path"]),
        "n_scanpaths": len(records),
        "n_primary_fixations": int(len(frame)),
        "n_valid_primary_fixations": int(valid.sum()),
        "n_train_primary_fixations": int(train.sum()),
        "selected_sigma_center_px": sigma_center,
        "selected_sigma_history_px": sigma_history,
        "selected_sigma_loo_px": sigma_loo,
        "selected_lambda_center_history": lambda_b5,
        "selected_lambda_loo_history": lambda_b5c,
        "alpha_primary": alpha,
        "grid": [grid.nx, grid.ny],
        "loo_count_identity_checked_on_n_images": checked,
        "train_center_count_mass": int(train_counts.sum()),
        "same_image_models": ["B2_loo_image", "B3_loo_index", "B5c_loo_history", "B6_image_shuffle"],
        "confirmatory_split": "valid",
        "train_scores_are_descriptive": True,
        "duration_column_used": False,
        "hypothesis_adopted": False,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    grid_table = attach_grid_robustness(root, cfg, scanpaths, n_boot, seed)
    grid_table.to_csv(output_dir / "grid_sensitivity.csv", index=False)
    write_atlas(output_dir, scanpaths, records, center, grid, root / "data/raw/images_subset", cfg["atlas"])
    write_figures(output_dir)
    return manifest


def _index0_sensitivity(records, grid, center_primary, sigma_center, sigma_loo, alpha, n_boot, seed, cfg) -> list[dict]:
    """Score released index 0 under maps that did and did not see index 0. Index 0 is not a primary target."""
    from humangazelab.temporal import histogram_counts, normalize_and_floor, blur_counts

    rows = []
    train_counts_with0 = accumulate_counts(records, lambda r: r.split == "train", 0, None, grid)
    center_with0 = density_from_counts(train_counts_with0, sigma_in_bins(sigma_center, grid), alpha)
    grouped = _group_records(records)
    subjects, scans, ig_b0, ig_b1, ig_b1_with0, ig_b2_index0, splits = [], [], [], [], [], [], []
    sigma_bins = sigma_in_bins(sigma_loo, grid)
    for name, group in grouped.items():
        hist0 = {}
        for record in group:
            if record.length < 1:
                continue
            hist0[record.subject] = histogram_counts(record.iy[:1], record.ix[:1], grid.ny, grid.nx)
        total0 = np.sum(list(hist0.values()), axis=0) if hist0 else np.zeros((grid.ny, grid.nx))
        blurred_total = blur_counts(total0, sigma_bins)
        for record in group:
            if record.length < 1 or record.subject not in hist0:
                continue
            iy = int(record.iy[0])
            ix = int(record.ix[0])
            reference = normalize_and_floor(blurred_total - blur_counts(hist0[record.subject], sigma_bins), alpha)
            subjects.append(record.subject)
            scans.append(record.scan_id)
            splits.append(record.split)
            ig_b0.append(0.0)
            ig_b1.append(float(information_gain(np.array([center_primary[iy, ix]]), grid.n_bins)[0]))
            ig_b1_with0.append(float(information_gain(np.array([center_with0[iy, ix]]), grid.n_bins)[0]))
            ig_b2_index0.append(float(information_gain(np.array([reference[iy, ix]]), grid.n_bins)[0]))
    payload = {
        "subject": np.array(subjects),
        "scan": np.array(scans),
        "split": np.array(splits),
        "ig_B0_uniform": np.array(ig_b0),
        "ig_B1_center_map_excludes_index0": np.array(ig_b1),
        "ig_B1_center_map_includes_index0": np.array(ig_b1_with0),
        "ig_B2_other_observers_index0": np.array(ig_b2_index0),
    }
    valid = payload["split"] == cfg["splits"]["eval_split_name"]
    for column in ("ig_B0_uniform", "ig_B1_center_map_excludes_index0", "ig_B1_center_map_includes_index0", "ig_B2_other_observers_index0"):
        values = scanpath_then_subject_means(payload["scan"][valid], payload["subject"][valid], payload[column][valid])
        summary = bootstrap_groups(values, column, "valid|index_0_only", "subject_scanpath_equal", n_boot, seed)
        summary["note"] = "Index 0 is not in the primary target set. This row is a sensitivity."
        rows.append(summary)
    return rows


def attach_grid_robustness(root: Path, cfg: dict, scanpaths, n_boot: int, seed: int) -> pd.DataFrame:
    """Score B1 and B4 on alternate grids using frozen pixel bandwidths and validation images."""
    rows = []
    alpha = float(cfg["models"]["alpha_primary"])
    # Selected bandwidths are read back from the manifest written by run().
    manifest = json.loads((root / cfg["output_dir"] / "run_manifest.json").read_text())
    for nx, ny in cfg["grid"]["robustness_grids"]:
        grid = Grid(cfg["canvas"]["width_px"], cfg["canvas"]["height_px"], int(nx), int(ny))
        role_kwargs = {
            "train_name": cfg["splits"]["train_split_name"],
            "modulus": int(cfg["splits"]["role_modulus"]),
            "bandwidth_bucket": int(cfg["splits"]["bandwidth_bucket"]),
            "mixture_bucket": int(cfg["splits"]["mixture_bucket"]),
        }
        records = pack_scanpaths(scanpaths, grid, role_kwargs)
        counts = accumulate_counts(records, lambda r: r.split == "train", 1, None, grid)
        center = density_from_counts(counts, sigma_in_bins(manifest["selected_sigma_center_px"], grid), alpha)
        kernel = blur_displacement(
            displacement_histogram(records, lambda r: r.split == "train", grid),
            sigma_in_bins(manifest["selected_sigma_history_px"], grid),
        )
        mass = canvas_kernel_mass(kernel, grid)
        subjects, scans, ig_b1, ig_b4 = [], [], [], []
        for record in records:
            if record.split != "valid" or record.length < 2:
                continue
            targets = np.arange(1, record.length)
            p_b1 = center[record.iy[targets], record.ix[targets]]
            from humangazelab.temporal import history_probabilities

            p_b4 = history_probabilities(
                kernel, mass, record.iy[targets - 1], record.ix[targets - 1], record.iy[targets], record.ix[targets], grid, alpha
            )
            ig1 = information_gain(p_b1, grid.n_bins)
            ig4 = information_gain(p_b4, grid.n_bins)
            subjects.append(np.full(targets.size, record.subject))
            scans.append(np.full(targets.size, record.scan_id))
            ig_b1.append(ig1)
            ig_b4.append(ig4)
        subject_arr = np.concatenate(subjects)
        scan_arr = np.concatenate(scans)
        for label, parts in (("ig_B1_center", ig_b1), ("ig_B4_history", ig_b4)):
            values = scanpath_then_subject_means(scan_arr, subject_arr, np.concatenate(parts))
            summary = bootstrap_groups(values, label, f"valid|grid_{nx}x{ny}", "subject_scanpath_equal", n_boot, seed)
            summary["nx"] = int(nx)
            summary["ny"] = int(ny)
            summary["sigma_source"] = "frozen_from_primary_grid_selection"
            rows.append(summary)
    return pd.DataFrame(rows)
