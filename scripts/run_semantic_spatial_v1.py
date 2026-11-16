"""Within-mask equal-mass bins for frozen DeepGaze III face and text regions.

Full conditional maps are not on disk. Each map is projected onto the original
canvas, reduced to four bins inside the mask, and discarded. The checkpoint,
scale, history, and eligibility are the frozen 700-image run.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from humangazelab.audit_checkpoint import two_way_subject_bootstrap
from humangazelab.osie import load_osie_scanpaths
from humangazelab.osie_deepgaze import paired_targets, scaled_coordinate
from humangazelab.osie_semantics import OsieAnnotations
from humangazelab.protocolshift import FROZEN_OUTPUTS, assert_new_output
from humangazelab.semantic_calibration import (
    fixation_pixel,
    inverse_index,
    matched_shifts,
    project_log_density,
    shift_mask,
)
from humangazelab.summarize import metric_seed, percentile_bootstrap_mean
from humangazelab.within_mask import (
    N_BINS,
    assign_equal_mass_bins,
    eroded_mask,
    image_bin_rates,
    pilot_stems,
    reproduce_primary_summary,
    weights_at,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "semantic_spatial_v1.yaml"
EXTRA_FROZEN = (
    "outputs/baseline_v1",
    "outputs/osie_deepgaze3_v2",
    "outputs/semantic_calibration_v1",
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


def _attach(scanpaths, targets):
    lookup = {(str(scan.image_name), int(scan.subject_id)): scan for scan in scanpaths}
    attached = []
    for row in targets:
        scan = lookup[(row["image_name"], row["subject"])]
        index = row["target_index"]
        attached.append({**row, "history_x": np.asarray(scan.x[:index], dtype=np.float64), "history_y": np.asarray(scan.y[:index], dtype=np.float64)})
    return attached


def _centroid(mask: np.ndarray) -> tuple[int, float]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return 0, math.nan
    distance = math.hypot(float(columns.mean()) - 399.5, float(rows.mean()) - 299.5)
    return int(len(rows)), distance


def _load_region_priors(directory: Path) -> dict[str, pd.DataFrame]:
    priors = {}
    for label in ("face", "text"):
        frame = pd.read_csv(directory / f"{label}.csv", usecols=["image", "subject", "target_index", "mask_present", "y", "p_model"])
        frame["image"] = frame["image"].astype(str)
        priors[label] = frame.set_index(["image", "subject", "target_index"], drop=False)
    return priors


def _reproduce(config: dict, output: Path) -> pd.DataFrame:
    frames = [pd.read_csv(ROOT / config["region_events"] / name) for name in ("object_union.csv", "face.csv", "text.csv")]
    events = pd.concat(frames, ignore_index=True)
    summary = pd.read_csv(ROOT / config["region_summary"])
    checked = reproduce_primary_summary(events, summary)
    checked.to_csv(output / "region_summary_reproduction.csv", index=False)
    if not bool(checked["n_matches"].all()) or float(checked["difference_gap"].max()) > 1e-8:
        raise RuntimeError("stored region-mass summary does not reproduce")
    return checked


def _face_lift(annotations: OsieAnnotations, scanpaths, stems: tuple[str, ...]) -> float:
    by_image: dict[str, list] = {}
    for scan in scanpaths:
        by_image.setdefault(Path(str(scan.image_name)).stem, []).append(scan)
    lifts = []
    for stem in stems:
        mask = annotations.label_masks(stem)["face"]
        area = float(mask.mean())
        if area <= 0:
            continue
        hits = 0
        total = 0
        for scan in by_image.get(stem, []):
            for index in range(4, len(scan.x)):
                row, column = fixation_pixel(float(scan.x[index]), float(scan.y[index]))
                total += 1
                hits += int(mask[row, column])
        if total:
            lifts.append((hits / total) / area)
    return float(np.mean(lifts)) if lifts else math.nan


def _score_image(dg3, model, stem, rows, image_path, masks, priors, row_index, column_index, src_x, src_y, width, height, center_log, rng, controls):
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    image_tensor = torch.tensor(array.transpose(2, 0, 1)[None, ...])
    readout, _orig, readout_shape = dg3._backbone(model, image_tensor)
    saliency_cache = [network(readout) for network in model.saliency_networks]
    geometry = {}
    eroded = {}
    shifts = {}
    for name, mask in masks.items():
        area, distance = _centroid(mask)
        geometry[name] = (area, distance)
        eroded[name] = eroded_mask(mask, 1)
        found, matched = matched_shifts(
            mask, rng, n_shifts=int(controls["n_shifts"]), radius=int(controls["radius_px"]),
            min_fraction=float(controls["min_fraction_inside"]), max_candidates=int(controls["max_candidates"]),
        )
        shifts[name] = (bool(matched), [shift_mask(mask, dy, dx) for dy, dx in found] if matched else [])
    order = np.array([-1, -2, -3, -4])
    scored = []
    example = None
    mass_gap = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), dg3.BATCH):
            chunk = rows[start : start + dg3.BATCH]
            hist_x, hist_y = [], []
            for row in chunk:
                hist_x.append(scaled_coordinate(row["history_x"][order], 800, width))
                hist_y.append(scaled_coordinate(row["history_y"][order], 600, height))
            density = dg3._density_batch(
                model, readout_shape, torch.tensor(center_log[None, ...], dtype=torch.float32),
                torch.tensor(np.stack(hist_x), dtype=torch.float32), torch.tensor(np.stack(hist_y), dtype=torch.float32),
                saliency_cache, src_x, src_y,
            )
            density_np = density[:, 0].numpy()
            projected = project_log_density(density_np, row_index, column_index)
            for batch_index, row in enumerate(chunk):
                pixel_row, pixel_column = fixation_pixel(row["x"], row["y"])
                fine_x = int(np.clip(np.rint(row["x"] * width / 800.0), 0, width - 1))
                fine_y = int(np.clip(np.rint(row["y"] * height / 600.0), 0, height - 1))
                log_model = float(density_np[batch_index, fine_y, fine_x])
                mass = projected[batch_index]
                if example is None and stem == "1001":
                    example = mass
                other = {"face": int(masks["text"][pixel_row, pixel_column]), "text": int(masks["face"][pixel_row, pixel_column])}
                for name, mask in masks.items():
                    key = (stem, int(row["subject"]), int(row["target_index"]))
                    stored = priors[name].loc[key]
                    region = float(mass[mask].sum()) if mask.any() else 0.0
                    if int(stored["mask_present"]) == 1:
                        mass_gap = max(mass_gap, abs(region - float(stored["p_model"])))
                    assignment = assign_equal_mass_bins(mass, mask) if mask.any() else None
                    in_mask = int(mask[pixel_row, pixel_column]) if mask.any() else 0
                    record = {
                        "image": stem,
                        "subject": int(row["subject"]),
                        "target_index": int(row["target_index"]),
                        "label": name,
                        "in_mask": in_mask,
                        "region_mass": region,
                        "stored_region_mass": float(stored["p_model"]),
                        "area_px": geometry[name][0],
                        "centroid_distance_px": geometry[name][1],
                        "concentration": math.nan,
                        "n_pixels": 0,
                        "n_boundary_pixels": 0,
                        "n_tied_pixels": 0,
                        "tied_mass": 0.0,
                        "on_rim": int(in_mask == 1 and not bool(eroded[name][pixel_row, pixel_column])),
                        "also_other": other[name],
                        "log_density_nats": log_model,
                        "shift_matched": int(shifts[name][0]),
                        "shift_hits": 0,
                        "shift_obs0_sum": 0.0,
                        "shift_pred0_sum": 0.0,
                        "skipped": int(assignment is None),
                    }
                    for bin_index in range(N_BINS):
                        record[f"obs{bin_index}"] = math.nan
                        record[f"pred{bin_index}"] = math.nan
                    if assignment is not None and in_mask:
                        weights = weights_at(assignment, pixel_row, pixel_column)
                        record["concentration"] = assignment["concentration"]
                        record["n_pixels"] = assignment["n_pixels"]
                        record["n_boundary_pixels"] = assignment["n_boundary_pixels"]
                        record["n_tied_pixels"] = assignment["n_tied_pixels"]
                        record["tied_mass"] = assignment["tied_mass"]
                        for bin_index, value in enumerate(weights):
                            record[f"obs{bin_index}"] = float(value)
                        for bin_index, value in enumerate(assignment["predicted"]):
                            record[f"pred{bin_index}"] = float(value)
                    if shifts[name][0]:
                        for moved in shifts[name][1]:
                            if not moved[pixel_row, pixel_column]:
                                continue
                            moved_assignment = assign_equal_mass_bins(mass, moved)
                            if moved_assignment is None:
                                continue
                            moved_weights = weights_at(moved_assignment, pixel_row, pixel_column)
                            if moved_weights is None:
                                continue
                            record["shift_hits"] += 1
                            record["shift_pred0_sum"] += float(moved_assignment["predicted"][0])
                            record["shift_obs0_sum"] += float(moved_weights[0])
                    scored.append(record)
    return scored, example, mass_gap


def _uncertainty(rates: pd.DataFrame, events: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    draws = int(config["bootstrap_draws"])
    seed = int(config["bootstrap_seed"])
    inside = events[events["in_mask"] == 1]
    for label, part in rates.groupby("label", sort=True):
        for column, estimand in (("highest_gap", "image_equal_highest_bin_minus_predicted"), ("highest_vs_quarter", "image_equal_highest_bin_minus_quarter")):
            boot = percentile_bootstrap_mean(part[column].to_numpy(), draws, metric_seed(seed, label, "inside", estimand))
            rows.append({"label": label, "estimand": estimand, "scope": "all_inside", **boot})
        fixation = inside[(inside["label"] == label) & inside["obs0"].notna()]
        if fixation.empty:
            continue
        gap = fixation["obs0"].to_numpy(dtype=float) - fixation["pred0"].to_numpy(dtype=float)
        subject_gap = fixation.assign(gap=fixation["obs0"] - fixation["pred0"]).groupby("subject")["gap"].mean()
        subject = percentile_bootstrap_mean(subject_gap.to_numpy(), draws, metric_seed(seed, label, "inside", "subject_gap"))
        rows.append({"label": label, "estimand": "subject_equal_highest_bin_minus_predicted", "scope": "all_inside", **subject})
        crossed = two_way_subject_bootstrap(
            fixation["subject"].to_numpy(), fixation["image"].to_numpy(), gap, draws, metric_seed(seed, label, "inside", "crossed"),
        )
        rows.append(
            {
                "label": label,
                "estimand": "crossed_highest_bin_minus_predicted",
                "scope": "all_inside",
                "n_groups": crossed["n_subjects"],
                "estimate": crossed["estimate"],
                "sd": crossed["sd"],
                "sem": crossed["sem"],
                "ci_low": crossed["ci_low_crossed"],
                "ci_high": crossed["ci_high_crossed"],
            }
        )
    return pd.DataFrame(rows)


def _controls(events: pd.DataFrame, rates: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    inside = events[(events["in_mask"] == 1) & events["obs0"].notna()].copy()
    for label, part in inside.groupby("label", sort=True):
        early = part[part["target_index"] <= int(config["early_index_max"])]
        later = part[part["target_index"] > int(config["early_index_max"])]
        interior = part[part["on_rim"] == 0]
        both = part[part["also_other"] == 1]
        only = part[part["also_other"] == 0]
        label_rows = events[events["label"] == label]
        shift = label_rows[label_rows["shift_hits"] > 0]
        per_image = []
        for _, group in shift.groupby("image", sort=True):
            hits = float(group["shift_hits"].sum())
            if hits <= 0:
                continue
            per_image.append(float(group["shift_obs0_sum"].sum() / hits - group["shift_pred0_sum"].sum() / hits))
        shift_boot = percentile_bootstrap_mean(
            np.asarray(per_image, dtype=float),
            int(config["bootstrap_draws"]),
            metric_seed(int(config["bootstrap_seed"]), label, "shift", "bin0_gap"),
        ) if per_image else {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n_groups": 0}
        def _gap(frame: pd.DataFrame) -> float:
            if frame.empty:
                return math.nan
            return float((frame["obs0"] - frame["pred0"]).mean())
        row = {
            "label": label,
            "n_inside": int(len(part)),
            "n_images": int(part["image"].nunique()),
            "n_subjects": int(part["subject"].nunique()),
            "early_gap": _gap(early),
            "later_gap": _gap(later),
            "interior_gap": _gap(interior),
            "overlap_gap": _gap(both),
            "single_label_gap": _gap(only),
            "n_early": int(len(early)),
            "n_later": int(len(later)),
            "n_interior": int(len(interior)),
            "n_overlap": int(len(both)),
            "shift_observed_bin0": float(shift["shift_obs0_sum"].sum() / shift["shift_hits"].sum()) if len(shift) and shift["shift_hits"].sum() else math.nan,
            "shift_predicted_bin0": float(shift["shift_pred0_sum"].sum() / shift["shift_hits"].sum()) if len(shift) and shift["shift_hits"].sum() else math.nan,
            "shift_gap_image_equal": shift_boot["estimate"],
            "shift_gap_ci_low": shift_boot["ci_low"],
            "shift_gap_ci_high": shift_boot["ci_high"],
            "n_shift_images": int(shift_boot["n_groups"]),
            "n_shift_hits": int(shift["shift_hits"].sum()) if len(shift) else 0,
            "mean_tied_fraction": float((part["n_tied_pixels"] / part["n_pixels"].clip(lower=1)).mean()),
            "mean_tied_mass": float(part["tied_mass"].mean()),
            "mean_boundary_pixels": float(part["n_boundary_pixels"].mean()),
        }
        image_part = rates[rates["label"] == label]
        if len(image_part) >= 3:
            row["corr_area"] = float(image_part["highest_gap"].corr(image_part["area_px"]))
            row["corr_centroid"] = float(image_part["highest_gap"].corr(image_part["centroid_distance_px"]))
            row["corr_concentration"] = float(image_part["highest_gap"].corr(image_part["concentration"]))
        for threshold in config["area_sensitivity_px"]:
            kept = part[part["area_px"] >= int(threshold)]
            row[f"gap_area_at_least_{int(threshold)}"] = _gap(kept)
            row[f"n_area_at_least_{int(threshold)}"] = int(len(kept))
        rows.append(row)
    return pd.DataFrame(rows)


def _histogram(rates: pd.DataFrame, path: Path) -> None:
    if rates.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for axis, label in zip(axes, ("face", "text")):
        part = rates[rates["label"] == label]
        if part.empty:
            axis.set_visible(False)
            continue
        observed = [float(part[f"observed_{index}"].mean()) for index in range(N_BINS)]
        predicted = [float(part[f"predicted_{index}"].mean()) for index in range(N_BINS)]
        positions = np.arange(N_BINS)
        axis.bar(positions - 0.15, observed, width=0.3, label="Observed", color="#1f4e79")
        axis.bar(positions + 0.15, predicted, width=0.3, label="Predicted mass", color="#c4a35a")
        axis.axhline(0.25, color="#666666", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(["Highest", "2", "3", "Lowest"])
        axis.set_title(label)
        axis.set_xlabel("Conditional density bin")
    axes[0].set_ylabel("Image-equal proportion")
    axes[0].legend(frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _decision(uncertainty: pd.DataFrame, config: dict) -> pd.DataFrame:
    stored = pd.read_csv(ROOT / config["decision_rule"]["total_mass_source"])
    rows = []
    letters = {}
    for label in config["labels"]:
        mass = stored[(stored["label"] == label) & (stored["estimand"] == config["decision_rule"]["total_mass_estimand"])].iloc[0]
        within = uncertainty[(uncertainty["label"] == label) & (uncertainty["estimand"] == "image_equal_highest_bin_minus_predicted") & (uncertainty["scope"] == "all_inside")]
        mass_excludes = float(mass["ci_low"]) > 0 or float(mass["ci_high"]) < 0
        if within.empty or not np.isfinite(within.iloc[0]["ci_low"]):
            within_excludes = False
            within_low = math.nan
            within_high = math.nan
            within_estimate = math.nan
        else:
            within_low = float(within.iloc[0]["ci_low"])
            within_high = float(within.iloc[0]["ci_high"])
            within_estimate = float(within.iloc[0]["estimate"])
            within_excludes = within_low > 0 or within_high < 0
        if mass_excludes and within_excludes:
            letter = "C"
        elif mass_excludes:
            letter = "A"
        elif within_excludes:
            letter = "B"
        else:
            letter = "D"
        letters[label] = letter
        rows.append(
            {
                "label": label,
                "total_mass_estimate": float(mass["estimate"]),
                "total_mass_ci_low": float(mass["ci_low"]),
                "total_mass_ci_high": float(mass["ci_high"]),
                "within_estimate": within_estimate,
                "within_ci_low": within_low,
                "within_ci_high": within_high,
                "letter": letter,
            }
        )
    overall = next(iter(letters.values())) if len(set(letters.values())) == 1 else "split:" + ",".join(f"{label}={letter}" for label, letter in letters.items())
    for row in rows:
        row["overall"] = overall
    return pd.DataFrame(rows)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    _refuse(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "events").mkdir(exist_ok=True)
    checked = _reproduce(config, output)
    print(checked.to_string(index=False), flush=True)
    scanpaths = load_osie_scanpaths(ROOT / config["fixations"])
    with OsieAnnotations(ROOT / config["attributes_file"]) as annotations:
        stage = os.environ.get("SEMANTIC_SPATIAL_STAGE", "pilot")
        requested = pilot_stems(list(annotations.stems), int(config["pilot_stride"])) if stage == "pilot" else annotations.stems
        lift = _face_lift(annotations, scanpaths, requested)
        print(f"face lift on requested images {lift:.3f}", flush=True)
        if not math.isfinite(lift) or lift < 2.0:
            raise RuntimeError("face masks do not align on the requested images")
        targets, counts = paired_targets(scanpaths)
        targets = [row for row in _attach(scanpaths, targets) if row["image"] in set(requested)]
        by_image: dict[str, list] = {}
        for row in targets:
            by_image.setdefault(row["image"], []).append(row)
        priors = _load_region_priors(ROOT / config["region_events"])
        dg3 = _load_deepgaze()
        model, missing_keys, unexpected_keys = dg3._load_model()
        width = int(round(800 * 35 / 24))
        height = int(round(600 * 35 / 24))
        center_log = dg3._centerbias(np.load(dg3.CENTERBIAS), height, width)
        row_index, column_index = inverse_index(height, width)
        readout_shape = [math.ceil(height / model.downsample / model.readout_factor), math.ceil(width / model.downsample / model.readout_factor)]
        src_x, src_y = dg3._nearest_source(height, width, readout_shape)
        prior = pd.read_csv(ROOT / config["previously_scored_output"])
        prior["image"] = prior["image"].astype(str)
        done_path = output / "completed_images.txt"
        done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
        pending = [stem for stem in requested if stem not in done]
        print(f"pending {len(pending)} of {len(requested)} stage {stage}", flush=True)
        started = time.time()
        for image_index, stem in enumerate(pending, start=1):
            image_started = time.time()
            masks = {name: annotations.label_masks(stem)[name] for name in config["labels"]}
            image_rng = np.random.default_rng(metric_seed(int(config["controls"]["seed"]), stem, "shift", "mask"))
            scored, example, mass_gap = _score_image(
                dg3, model, stem, by_image[stem], ROOT / config["stimuli"] / f"{stem}.jpg", masks, priors,
                row_index, column_index, src_x, src_y, width, height, center_log, image_rng, config["controls"],
            )
            block = pd.DataFrame(scored)
            one = block.drop_duplicates(["image", "subject", "target_index"])
            merged = prior[prior["image"] == stem].merge(one, on=["image", "subject", "target_index"], suffixes=("_frozen", "_new"))
            if len(merged) != int((prior["image"] == stem).sum()):
                raise RuntimeError(f"{stem} matched {len(merged)} frozen rows")
            density_gap = float((merged["log_density_nats_frozen"] - merged["log_density_nats_new"]).abs().max())
            if density_gap > float(config["log_density_stop"]) or mass_gap > float(config["region_mass_stop"]):
                raise RuntimeError(f"{stem} density gap {density_gap} region-mass gap {mass_gap}")
            block.to_csv(output / "events" / f"{stem}.csv", index=False)
            if example is not None:
                np.savez_compressed(output / "examples_1001_first_mass.npz", mass=example)
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(stem + "\n")
            print(f"{stem} rows {len(block)} density {density_gap:.3e} mass {mass_gap:.3e} {time.time() - image_started:.1f}s {image_index}/{len(pending)}", flush=True)
        frames = [pd.read_csv(output / "events" / f"{stem}.csv") for stem in requested if (output / "events" / f"{stem}.csv").exists()]
        events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        events.to_csv(output / "fixation_bins.csv", index=False)
        rates = image_bin_rates(events) if not events.empty else pd.DataFrame()
        rates.to_csv(output / "per_image_bins.csv", index=False)
        observer_rows = []
        inside = events[(events["in_mask"] == 1) & events["obs0"].notna()] if not events.empty else events
        if not inside.empty:
            for (subject, label), part in inside.groupby(["subject", "label"], sort=True):
                observer_rows.append(
                    {
                        "subject": int(subject),
                        "label": label,
                        "n": int(len(part)),
                        "observed_0": float(part["obs0"].mean()),
                        "predicted_0": float(part["pred0"].mean()),
                        "highest_gap": float(part["obs0"].mean() - part["pred0"].mean()),
                    }
                )
        pd.DataFrame(observer_rows).to_csv(output / "per_observer_bins.csv", index=False)
        bin_rows = []
        for label, part in rates.groupby("label") if not rates.empty else []:
            for bin_index in range(N_BINS):
                bin_rows.append(
                    {
                        "label": label,
                        "bin": bin_index,
                        "image_equal_observed": float(part[f"observed_{bin_index}"].mean()),
                        "image_equal_predicted": float(part[f"predicted_{bin_index}"].mean()),
                        "n_images": int(len(part)),
                    }
                )
        pd.DataFrame(bin_rows).to_csv(output / "bin_proportions.csv", index=False)
        uncertainty = _uncertainty(rates, events, config) if not rates.empty else pd.DataFrame()
        uncertainty.to_csv(output / "uncertainty.csv", index=False)
        controls = _controls(events, rates, config) if not events.empty else pd.DataFrame()
        controls.to_csv(output / "controls.csv", index=False)
        _histogram(rates, output / "figures" / "within_mask_bins.png")
        complete = stage == "all" and set(requested) <= set(done_path.read_text(encoding="utf-8").split())
        decision = _decision(uncertainty, config) if complete and not uncertainty.empty else pd.DataFrame()
        if not decision.empty:
            decision.to_csv(output / "decision.csv", index=False)
        manifest = {
            "exploratory": True,
            "complete": complete,
            "stage": stage,
            "deepgaze_commit": config["checkpoint"]["commit"],
            "state_dict_missing": missing_keys,
            "state_dict_unexpected": unexpected_keys,
            "n_requested_images": len(requested),
            "n_fixation_rows": int(len(events)),
            "face_lift_requested": lift,
            "paired_counts": counts,
            "seconds_this_process": time.time() - started,
            "cuda": bool(torch.cuda.is_available()),
            "maps_cached_before_this_run": False,
            "reproduction_max_gap": float(checked["difference_gap"].max()),
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
