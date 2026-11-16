"""Stream frozen DeepGaze III maps into OSIE semantic-region probabilities.

The checkpoint, scale, history, and center-bias template are the ones used for
outputs/osie_deepgaze3_v2. Full maps are not stored. Each map is reduced to
region masses on the original 800 by 600 canvas and then discarded.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion

from humangazelab.audit_checkpoint import two_way_subject_bootstrap
from humangazelab.osie import load_osie_scanpaths
from humangazelab.osie_deepgaze import paired_targets, scaled_coordinate
from humangazelab.osie_semantics import (
    COVERAGE_MIN_IMAGES,
    PRIMARY_LABELS,
    OsieAnnotations,
    manifest_frame,
    select_alignment_cases,
    stage_a_stems,
)
from humangazelab.protocolshift import FROZEN_OUTPUTS, assert_new_output
from humangazelab.semantic_calibration import (
    LOG_LOSS_CLIP,
    brier_score,
    crossfit_linear_residual,
    dilate,
    fixation_pixel,
    inverse_index,
    log_loss,
    matched_shifts,
    observed_expected,
    project_log_density,
    region_covariates,
    reliability_rows,
    shift_mask,
)
from humangazelab.summarize import metric_seed, percentile_bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "semantic_calibration_v1.yaml"
EXTRA_FROZEN = (
    "outputs/baseline_v1",
    "outputs/osie_deepgaze3_v2",
    "outputs/protocolshift_b3_v1",
    "outputs/protocolshift_prep_v1",
)


def _load_deepgaze():
    path = ROOT / "scripts" / "run_osie_deepgaze3_v2.py"
    spec = importlib.util.spec_from_file_location("osie_deepgaze3_v2_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse(output: Path) -> None:
    assert_new_output(ROOT, output)
    for relative in (*FROZEN_OUTPUTS, *EXTRA_FROZEN):
        if output.resolve() == (ROOT / relative).resolve():
            raise RuntimeError(f"refusing to write into {relative}")


def _contour(mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) < 3:
        return mask
    eroded = binary_erosion(mask)
    edge = mask & ~eroded
    return edge if edge.any() else mask


def _overlays(annotations: OsieAnnotations, cases: pd.DataFrame, scanpaths, image_dir: Path, figure_dir: Path) -> None:
    by_image: dict[str, list] = {}
    for scan in scanpaths:
        by_image.setdefault(Path(str(scan.image_name)).stem, []).append(scan)
    for row in cases.itertuples(index=False):
        stem = str(row.image)
        objects = annotations.objects(stem)
        chosen = objects[int(row.object_index)]
        image = np.asarray(Image.open(image_dir / f"{stem}.jpg").convert("RGB"))
        figure, axis = plt.subplots(figsize=(6.4, 4.8))
        axis.imshow(image)
        edge = _contour(chosen.mask)
        display = image.copy()
        display[edge] = (220, 60, 40)
        axis.imshow(display)
        for scan in by_image.get(stem, []):
            axis.plot(scan.x, scan.y, ".", color="white", markersize=3, alpha=0.8)
        axis.set_axis_off()
        axis.set_title(f"{stem} {row.reason} object {int(row.object_index)}")
        figure.tight_layout()
        figure.savefig(figure_dir / f"alignment_{row.reason}_{stem}.png", dpi=120)
        plt.close(figure)


def _hit_rate(mask: np.ndarray, scans) -> tuple[int, int]:
    hits = 0
    total = 0
    for scan in scans:
        for index in range(4, len(scan.x)):
            row, column = fixation_pixel(float(scan.x[index]), float(scan.y[index]))
            total += 1
            hits += int(mask[row, column])
    return hits, total


def _alignment_audit(annotations: OsieAnnotations, scanpaths, cases: pd.DataFrame) -> pd.DataFrame:
    by_image: dict[str, list] = {}
    for scan in scanpaths:
        by_image.setdefault(Path(str(scan.image_name)).stem, []).append(scan)
    rows = []
    for stem in annotations.stems:
        labels = annotations.label_masks(stem)
        hits, total = _hit_rate(labels["object_union"], by_image.get(stem, []))
        face_hits, _ = _hit_rate(labels["face"], by_image.get(stem, []))
        rows.append(
            {
                "image": stem,
                "n_eligible_index_ge_4": total,
                "object_hits": hits,
                "object_hit_rate": hits / total if total else math.nan,
                "object_area_fraction": float(labels["object_union"].mean()),
                "face_hits": face_hits,
                "face_area_fraction": float(labels["face"].mean()),
                "text_area_fraction": float(labels["text"].mean()),
                "n_objects": int(sum(1 for _ in annotations.objects(stem))),
            }
        )
    frame = pd.DataFrame(rows)
    present = frame[frame["face_area_fraction"] > 0]
    lift = (present["face_hits"] / present["n_eligible_index_ge_4"]) / present["face_area_fraction"]
    frame["face_lift_among_images_with_a_face"] = float(lift.mean()) if len(present) else math.nan
    frame["alignment_case"] = frame["image"].isin(set(cases["image"].astype(str)))
    return frame


def _score_image(dg3, model, stem, rows, image_path, annotations, center_mass, row_index, column_index, src_x, src_y, width, height, center_log, rng):
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    original = np.asarray(image)
    image_tensor = torch.tensor(array.transpose(2, 0, 1)[None, ...])
    readout, _orig, readout_shape = dg3._backbone(model, image_tensor)
    saliency_cache = [network(readout) for network in model.saliency_networks]
    labels = annotations.label_masks(stem)
    covariates = {name: region_covariates(mask, original) for name, mask in labels.items()}
    dilated = {name: {1: dilate(mask, 1), 12: dilate(mask, 12)} for name, mask in labels.items()}
    shifts = {}
    for name, mask in labels.items():
        found, matched = matched_shifts(
            mask, rng, n_shifts=8, radius=80, min_fraction=0.8, max_candidates=40
        )
        shifts[name] = (found, matched, [shift_mask(mask, dy, dx) for dy, dx in found])
    order = np.array([-1, -2, -3, -4])
    scored = []
    example = None
    with torch.no_grad():
        for start in range(0, len(rows), dg3.BATCH):
            chunk = rows[start : start + dg3.BATCH]
            hist_x, hist_y = [], []
            for row in chunk:
                hist_x.append(scaled_coordinate(row["history_x"][order], 800, width))
                hist_y.append(scaled_coordinate(row["history_y"][order], 600, height))
            x_hist = torch.tensor(np.stack(hist_x), dtype=torch.float32)
            y_hist = torch.tensor(np.stack(hist_y), dtype=torch.float32)
            center_tensor = torch.tensor(center_log[None, ...], dtype=torch.float32)
            density = dg3._density_batch(
                model, readout_shape, center_tensor, x_hist, y_hist, saliency_cache, src_x, src_y
            )
            density_np = density[:, 0].numpy()
            mass = project_log_density(density_np, row_index, column_index)
            for row_index_batch, row in enumerate(chunk):
                pixel_row, pixel_column = fixation_pixel(row["x"], row["y"])
                fine_x = int(np.clip(np.rint(row["x"] * width / 800.0), 0, width - 1))
                fine_y = int(np.clip(np.rint(row["y"] * height / 600.0), 0, height - 1))
                log_model = float(density_np[row_index_batch, fine_y, fine_x])
                log_center = float(center_log[fine_y, fine_x])
                if example is None:
                    example = mass[row_index_batch]
                fixation_distance = math.hypot(row["x"] - 399.5, row["y"] - 299.5)
                for name, mask in labels.items():
                    present = bool(mask.any())
                    outcome = int(mask[pixel_row, pixel_column]) if present else 0
                    p_model = float(mass[row_index_batch][mask].sum()) if present else 0.0
                    p_center = float(center_mass[mask].sum()) if present else 0.0
                    p_uniform = float(mask.mean()) if present else 0.0
                    found, matched, moved = shifts[name]
                    if matched and found:
                        p_shift = float(np.mean([item_mass.sum() for item_mass in (mass[row_index_batch][item] for item in moved)]))
                        y_shift = float(np.mean([int(item[pixel_row, pixel_column]) for item in moved]))
                    else:
                        p_shift = math.nan
                        y_shift = math.nan
                    cov = covariates[name]
                    scored.append(
                        {
                            "image": stem,
                            "subject": row["subject"],
                            "target_index": row["target_index"],
                            "x": row["x"],
                            "y": row["y"],
                            "prev_x": float(row["history_x"][-1]),
                            "prev_y": float(row["history_y"][-1]),
                            "fixation_center_distance_px": fixation_distance,
                            "label": name,
                            "mask_present": int(present),
                            "y": outcome,
                            "y_border_1": int(dilated[name][1][pixel_row, pixel_column]) if present else 0,
                            "y_border_12": int(dilated[name][12][pixel_row, pixel_column]) if present else 0,
                            "p_model": p_model,
                            "p_center": p_center,
                            "p_uniform": p_uniform,
                            "p_shift_mean": p_shift,
                            "y_shift_mean": y_shift,
                            "shift_matched": int(matched),
                            "log_density_nats": log_model,
                            "centerbias_log_density_nats": log_center,
                            **cov,
                        }
                    )
    return scored, example


def _attach(scanpaths, targets):
    lookup = {(str(scan.image_name), int(scan.subject_id)): scan for scan in scanpaths}
    attached = []
    for row in targets:
        scan = lookup[(row["image_name"], row["subject"])]
        index = row["target_index"]
        attached.append(
            {
                **row,
                "history_x": np.asarray(scan.x[:index], dtype=np.float64),
                "history_y": np.asarray(scan.y[:index], dtype=np.float64),
            }
        )
    return attached


def _write_summaries(events: pd.DataFrame, output: Path, config: dict, complete: bool) -> None:
    present = events[events["mask_present"] == 1].copy()
    present["gap"] = present["y"] - present["p_model"]
    image_rows = []
    observer_rows = []
    summary_rows = []
    curve_rows = []
    uncertainty_rows = []
    control_rows = []
    n_boot = int(config["bootstrap_draws"])
    seed = int(config["bootstrap_seed"])
    for label, part in present.groupby("label", sort=True):
        image_gap = part.groupby("image", sort=True)["gap"].mean()
        subject_gap = part.groupby("subject", sort=True)["gap"].mean()
        stats = observed_expected(part["p_model"], part["y"])
        center_stats = observed_expected(part["p_center"], part["y"])
        uniform_stats = observed_expected(part["p_uniform"], part["y"])
        role = "primary" if label in PRIMARY_LABELS else "exploratory"
        n_images = int(part["image"].nunique())
        coverage = "sufficient" if n_images >= COVERAGE_MIN_IMAGES else "below_predeclared_minimum"
        summary_rows.append(
            {
                "label": label,
                "role": role,
                "coverage": coverage,
                "n_images": n_images,
                "n_subjects": int(part["subject"].nunique()),
                "n_fixations": int(len(part)),
                "prevalence_of_images": n_images / int(events["image"].nunique()),
                **{f"model_{key}": value for key, value in stats.items()},
                "model_brier": brier_score(part["p_model"], part["y"]),
                "model_log_loss": log_loss(part["p_model"], part["y"], LOG_LOSS_CLIP),
                "center_difference": center_stats["difference"],
                "center_brier": brier_score(part["p_center"], part["y"]),
                "uniform_difference": uniform_stats["difference"],
                "uniform_brier": brier_score(part["p_uniform"], part["y"]),
                "shift_difference": float((part["y_shift_mean"] - part["p_shift_mean"]).mean()),
                "border1_difference": float((part["y_border_1"] - part["p_model"]).mean()),
                "border12_difference": float((part["y_border_12"] - part["p_model"]).mean()),
            }
        )
        image_boot = percentile_bootstrap_mean(image_gap.to_numpy(), n_boot, metric_seed(seed, label, "present", "image"))
        subject_boot = percentile_bootstrap_mean(subject_gap.to_numpy(), n_boot, metric_seed(seed, label, "present", "subject"))
        crossed = two_way_subject_bootstrap(
            part["subject"].to_numpy(),
            part["image"].to_numpy(),
            part["gap"].to_numpy(dtype=float),
            n_boot,
            metric_seed(seed, label, "present", "crossed"),
        )
        for estimand, boot in (
            ("image_equal_gap", image_boot),
            ("subject_equal_gap", subject_boot),
        ):
            uncertainty_rows.append({"label": label, "estimand": estimand, "role": role, **boot})
        uncertainty_rows.append(
            {
                "label": label,
                "estimand": "subject_equal_gap_crossed_resample",
                "role": role,
                "n_groups": crossed["n_subjects"],
                "estimate": crossed["estimate"],
                "sd": crossed["sd"],
                "sem": crossed["sem"],
                "ci_low": crossed["ci_low_crossed"],
                "ci_high": crossed["ci_high_crossed"],
            }
        )
        for image, image_part in part.groupby("image", sort=True):
            image_rows.append(
                {
                    "image": image,
                    "label": label,
                    "n": int(len(image_part)),
                    "mask_present": 1,
                    **observed_expected(image_part["p_model"], image_part["y"]),
                    "p_center": float(image_part["p_center"].mean()),
                    "p_uniform": float(image_part["p_uniform"].mean()),
                    "area_fraction": float(image_part["area_fraction"].iloc[0]),
                    "centroid_distance_px": float(image_part["centroid_distance_px"].iloc[0]),
                    "luminance_contrast": float(image_part["luminance_contrast"].iloc[0]),
                    "gradient_energy": float(image_part["gradient_energy"].iloc[0]),
                    "orientation_energy": float(image_part["orientation_energy"].iloc[0]),
                    "eccentricity": float(image_part["eccentricity"].iloc[0]),
                }
            )
        for subject, subject_part in part.groupby("subject", sort=True):
            observer_rows.append(
                {"subject": int(subject), "label": label, "n": int(len(subject_part)), **observed_expected(subject_part["p_model"], subject_part["y"])}
            )
        for curve in reliability_rows(part["p_model"].to_numpy(), part["y"].to_numpy(), int(config["reliability_bins"])):
            curve_rows.append({"label": label, "scope": "pooled_fixations_descriptive", **curve})
        early = part[part["target_index"] <= 6]
        late = part[part["target_index"] >= 7]
        near_px = float(config["controls"]["previous_fixation_near_px"])
        prev_distance = np.hypot(part["x"] - part["prev_x"], part["y"] - part["prev_y"])
        near = part[prev_distance <= near_px]
        far = part[prev_distance > near_px]
        control_rows.append(
            {
                "label": label,
                "model_difference": stats["difference"],
                "center_difference": center_stats["difference"],
                "uniform_difference": uniform_stats["difference"],
                "shift_difference": float((part["y_shift_mean"] - part["p_shift_mean"]).mean()),
                "n_shift_unmatched_fixations": int(part["shift_matched"].eq(0).sum()),
                "early_index_difference": float((early["y"] - early["p_model"]).mean()) if len(early) else math.nan,
                "later_index_difference": float((late["y"] - late["p_model"]).mean()) if len(late) else math.nan,
                "previous_near_px": near_px,
                "previous_near_difference": float((near["y"] - near["p_model"]).mean()) if len(near) else math.nan,
                "previous_far_difference": float((far["y"] - far["p_model"]).mean()) if len(far) else math.nan,
                "n_previous_near": int(len(near)),
                "n_previous_far": int(len(far)),
            }
        )
    semantic = present[present["label"] != "object_union"]
    hit_counts = semantic.groupby(["image", "subject", "target_index"], sort=False)["y"].sum().rename("n_semantic_hits")
    for row in control_rows:
        label = row["label"]
        part = present[present["label"] == label].merge(hit_counts, on=["image", "subject", "target_index"], how="left")
        others = part["n_semantic_hits"] - (part["y"] if label != "object_union" else 0)
        multi = part[part["n_semantic_hits"] >= 2]
        single = part[part["n_semantic_hits"] == 1]
        row["multi_label_difference"] = float((multi["y"] - multi["p_model"]).mean()) if len(multi) else math.nan
        row["single_label_difference"] = float((single["y"] - single["p_model"]).mean()) if len(single) else math.nan
        row["n_multi_label_fixations"] = int(len(multi))
        row["n_fixations_with_another_semantic_label"] = int((others >= 1).sum())
    image_frame = pd.DataFrame(image_rows)
    if not image_frame.empty:
        pieces = []
        for label in PRIMARY_LABELS:
            subset = image_frame[image_frame["label"] == label].rename(columns={"difference": "gap"})
            if len(subset) < 10:
                continue
            residual = crossfit_linear_residual(
                subset, "gap", ["area_fraction", "centroid_distance_px"], int(config["crossfit_folds"]), seed
            )
            residual["label"] = label
            pieces.append(residual)
        if pieces:
            pd.concat(pieces, ignore_index=True).to_csv(output / "covariate_residuals.csv", index=False)
    image_frame.to_csv(output / "per_image_summary.csv", index=False)
    pd.DataFrame(observer_rows).to_csv(output / "per_observer_summary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "semantic_calibration_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "calibration_curves.csv", index=False)
    pd.DataFrame(uncertainty_rows).to_csv(output / "uncertainty_summary.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output / "control_comparisons.csv", index=False)
    _failure_atlas(image_frame, events, output, config, complete)
    _curve_figure(pd.DataFrame(curve_rows), output / "figures" / "reliability_primary.png")


def _failure_atlas(image_frame: pd.DataFrame, events: pd.DataFrame, output: Path, config: dict, complete: bool) -> None:
    rule = config["failure_rule"]
    rows = []
    for label in rule["labels"]:
        subset = image_frame[(image_frame["label"] == label) & (image_frame["n"] >= int(rule["min_fixations"]))].copy()
        if subset.empty:
            continue
        ordered = subset.sort_values(["difference", "image"], ascending=[True, True])
        chosen = pd.concat([ordered.head(int(rule["n_each_tail"])), ordered.tail(int(rule["n_each_tail"]))], ignore_index=True)
        chosen = chosen.drop_duplicates(["image", "label"])
        for row in chosen.itertuples(index=False):
            rows.append(
                {
                    "label": label,
                    "image": row.image,
                    "n": int(row.n),
                    "difference": float(row.difference),
                    "mean_predicted": float(row.mean_predicted),
                    "observed_frequency": float(row.observed_frequency),
                    "area_fraction": float(row.area_fraction),
                    "centroid_distance_px": float(row.centroid_distance_px),
                    "tail": "observed_below_predicted" if row.difference < 0 else "observed_above_predicted",
                    "atlas_complete": complete,
                }
            )
    pd.DataFrame(rows).to_csv(output / "failure_cases.csv", index=False)


def _curve_figure(curves: pd.DataFrame, path: Path) -> None:
    if curves.empty:
        return
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    for label in PRIMARY_LABELS:
        part = curves[curves["label"] == label]
        if part.empty:
            continue
        axis.plot(part["mean_predicted"], part["observed_frequency"], marker="o", label=label)
    axis.plot([0, 1], [0, 1], color="#666666", linewidth=0.8)
    axis.set_xlabel("Mean predicted region probability")
    axis.set_ylabel("Observed fixation frequency")
    axis.set_title("Pooled reliability, descriptive")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _failure_overlays(annotations, cases: pd.DataFrame, image_dir: Path, figure_dir: Path, scanpaths) -> None:
    if cases.empty:
        return
    by_image = {}
    for scan in scanpaths:
        by_image.setdefault(Path(str(scan.image_name)).stem, []).append(scan)
    for row in cases.itertuples(index=False):
        stem = str(row.image)
        labels = annotations.label_masks(stem)
        mask = labels[str(row.label)]
        image = np.asarray(Image.open(image_dir / f"{stem}.jpg").convert("RGB"))
        display = image.copy()
        display[_contour(mask)] = (220, 60, 40)
        figure, axis = plt.subplots(figsize=(6.4, 4.8))
        axis.imshow(display)
        for scan in by_image.get(stem, []):
            axis.plot(np.asarray(scan.x)[4:], np.asarray(scan.y)[4:], ".", color="white", markersize=3)
        axis.set_axis_off()
        axis.set_title(f"{stem} {row.label} gap {row.difference:.2f}")
        figure.tight_layout()
        figure.savefig(figure_dir / f"failure_{row.label}_{stem}.png", dpi=120)
        plt.close(figure)


def _failure_densities(dg3, model, failure, by_image, image_dir, annotations, center_log, row_index, column_index, src_x, src_y, width, height, figure_dir, example_dir) -> None:
    """Recompute mean conditional maps for the predeclared failure images only."""
    if failure.empty:
        return
    stems = sorted(set(failure["image"].astype(str)))
    for stem in stems:
        rows = by_image.get(stem)
        if not rows:
            continue
        image = Image.open(image_dir / f"{stem}.jpg").convert("RGB")
        array = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        image_tensor = torch.tensor(array.transpose(2, 0, 1)[None, ...])
        readout, _orig, readout_shape = dg3._backbone(model, image_tensor)
        saliency_cache = [network(readout) for network in model.saliency_networks]
        labels = annotations.label_masks(stem)
        order = np.array([-1, -2, -3, -4])
        total = None
        count = 0
        with torch.no_grad():
            for start in range(0, len(rows), dg3.BATCH):
                chunk = rows[start : start + dg3.BATCH]
                hist_x, hist_y = [], []
                for row in chunk:
                    hist_x.append(scaled_coordinate(row["history_x"][order], 800, width))
                    hist_y.append(scaled_coordinate(row["history_y"][order], 600, height))
                x_hist = torch.tensor(np.stack(hist_x), dtype=torch.float32)
                y_hist = torch.tensor(np.stack(hist_y), dtype=torch.float32)
                center_tensor = torch.tensor(center_log[None, ...], dtype=torch.float32)
                density = dg3._density_batch(
                    model, readout_shape, center_tensor, x_hist, y_hist, saliency_cache, src_x, src_y
                )
                mass = project_log_density(density[:, 0].numpy(), row_index, column_index)
                total = mass.sum(axis=0) if total is None else total + mass.sum(axis=0)
                count += mass.shape[0]
        if total is None or count == 0:
            continue
        mean_mass = total / count
        np.savez_compressed(example_dir / f"{stem}_mean_mass.npz", mass=mean_mass.astype(np.float32))
        picture = np.asarray(image)
        for row in failure[failure["image"].astype(str) == stem].itertuples(index=False):
            display = picture.copy()
            display[_contour(labels[str(row.label)])] = (220, 60, 40)
            figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
            axes[0].imshow(display)
            axes[0].plot([item["x"] for item in rows], [item["y"] for item in rows], ".", color="white", markersize=2, alpha=0.7)
            axes[1].imshow(mean_mass, cmap="magma")
            axes[1].contour(labels[str(row.label)], levels=[0.5], colors=["#7CFFCB"], linewidths=0.6)
            for axis in axes:
                axis.set_axis_off()
            axes[0].set_title(f"{stem} {row.label}")
            axes[1].set_title("Mean conditional mass")
            figure.tight_layout()
            figure.savefig(figure_dir / f"failure_density_{row.label}_{stem}.png", dpi=120)
            plt.close(figure)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    _refuse(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "events").mkdir(exist_ok=True)
    (output / "examples").mkdir(exist_ok=True)
    image_dir = ROOT / config["stimuli"]
    scanpaths = load_osie_scanpaths(ROOT / config["fixations"])
    with OsieAnnotations(ROOT / config["attributes_file"]) as annotations:
        manifest = manifest_frame(annotations)
        manifest.to_csv(output / "annotation_manifest.csv", index=False)
        cases = select_alignment_cases(manifest)
        cases.to_csv(output / "alignment_cases.csv", index=False)
        audit = _alignment_audit(annotations, scanpaths, cases)
        audit.to_csv(output / "alignment_audit.csv", index=False)
        face_lift = float(audit["face_lift_among_images_with_a_face"].iloc[0])
        print(f"face lift {face_lift:.3f}", flush=True)
        if not math.isfinite(face_lift) or face_lift < 2.0:
            raise RuntimeError("face masks do not align with fixations; stopping")
        _overlays(annotations, cases, scanpaths, image_dir, output / "figures")
        stage = os.environ.get("SEMANTIC_STAGE", "all")
        requested = stage_a_stems(manifest) if stage == "A" else annotations.stems
        targets, counts = paired_targets(scanpaths)
        targets = [row for row in _attach(scanpaths, targets) if row["image"] in set(requested)]
        by_image = {}
        for row in targets:
            by_image.setdefault(row["image"], []).append(row)
        (output / "eligibility.json").write_text(json.dumps({"paired": counts, "requested_images": len(requested)}, indent=2) + "\n", encoding="utf-8")
        dg3 = _load_deepgaze()
        model, missing_keys, unexpected_keys = dg3._load_model()
        template = np.load(dg3.CENTERBIAS)
        width = int(round(800 * 35 / 24))
        height = int(round(600 * 35 / 24))
        center_log = dg3._centerbias(template, height, width)
        row_index, column_index = inverse_index(height, width)
        center_mass = project_log_density(center_log, row_index, column_index)
        readout_shape = [
            math.ceil(height / model.downsample / model.readout_factor),
            math.ceil(width / model.downsample / model.readout_factor),
        ]
        src_x, src_y = dg3._nearest_source(height, width, readout_shape)
        done_path = output / "completed_images.txt"
        done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
        prior = pd.read_csv(ROOT / config["previously_scored_output"])
        prior["image"] = prior["image"].astype(str)
        pending = [stem for stem in requested if stem not in done]
        print(f"pending {len(pending)} of {len(requested)}", flush=True)
        started = time.time()
        for image_index, stem in enumerate(pending, start=1):
            image_started = time.time()
            image_rng = np.random.default_rng(metric_seed(int(config["controls"]["seed"]), stem, "shift", "mask"))
            scored, example = _score_image(
                dg3, model, stem, by_image[stem], image_dir / f"{stem}.jpg", annotations,
                center_mass, row_index, column_index, src_x, src_y, width, height, center_log, image_rng,
            )
            block = pd.DataFrame(scored)
            block.to_csv(output / "events" / f"{stem}.csv", index=False)
            one = block.drop_duplicates(["image", "subject", "target_index"])
            merged = prior[prior["image"] == stem].merge(one, on=["image", "subject", "target_index"], suffixes=("_frozen", "_new"))
            if len(merged) != int((prior["image"] == stem).sum()):
                raise RuntimeError(f"{stem} matched {len(merged)} frozen rows")
            gap = float((merged["log_density_nats_frozen"] - merged["log_density_nats_new"]).abs().max())
            if gap > 1e-3:
                raise RuntimeError(f"{stem} log density moved by {gap}")
            if stem == "1001" and example is not None:
                np.savez_compressed(output / "examples" / "1001_first_target_mass.npz", mass=example)
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(stem + "\n")
            print(f"{stem} rows {len(block)} gap {gap:.3e} {time.time() - image_started:.1f}s {image_index}/{len(pending)}", flush=True)
        frames = [pd.read_csv(output / "events" / f"{stem}.csv") for stem in requested if (output / "events" / f"{stem}.csv").exists()]
        events = pd.concat(frames, ignore_index=True)
        events.to_csv(output / "region_events.csv", index=False)
        complete = stage != "A" and set(requested) <= set(done_path.read_text(encoding="utf-8").split())
        _write_summaries(events, output, config, complete)
        if complete:
            failure = pd.read_csv(output / "failure_cases.csv")
            _failure_overlays(annotations, failure, image_dir, output / "figures", scanpaths)
            _failure_densities(
                dg3, model, failure, by_image, image_dir, annotations, center_log,
                row_index, column_index, src_x, src_y, width, height,
                output / "figures", output / "examples",
            )
        report = {
            "exploratory": True,
            "complete": complete,
            "stage": stage,
            "deepgaze_commit": config["checkpoint"]["commit"],
            "state_dict_missing": missing_keys,
            "state_dict_unexpected": unexpected_keys,
            "n_requested_images": len(requested),
            "n_event_rows": int(len(events)),
            "face_lift": face_lift,
            "seconds_this_process": time.time() - started,
            "cuda": bool(torch.cuda.is_available()),
            "gpu_memory_bytes": None,
        }
        (output / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
