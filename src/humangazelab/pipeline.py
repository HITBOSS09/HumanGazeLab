"""Run the descriptive COCO-FreeView analysis and write tables and figures."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import PIL
import yaml

from humangazelab import __version__
from humangazelab.adapter import Scanpath, load_scanpaths
from humangazelab.config import AnalysisConfig, load_config
from humangazelab.features import extract_feature_table, number_token
from humangazelab.plotting import (
    plot_all_subjects,
    plot_density_pair,
    plot_early_late,
    plot_radius_hist,
    plot_saccade_hist,
    plot_scanpath_grid,
    plot_subject_radius,
)
from humangazelab.quality import (
    quality_report,
    quality_table,
    schema_comparison,
    split_counts_table,
    subject_coverage_table,
)
from humangazelab.spatial import (
    density_table,
    disk_fraction,
    grid_mean_radius,
    ordinal_epoch,
    pooled_marginals,
    post_initial,
    subject_mean_density,
    uniform_mean_radius,
    uniform_radius_sample,
)
from humangazelab.summarize import metric_seed, summarize_column

WEIGHTING = "Each scanpath has equal weight inside a group. Each group then has equal weight."


def run(root: str | Path, config_path: str | Path) -> dict:
    root = Path(root)
    config = load_config(config_path)
    fixation_path = root / config.fixation_path
    scanpaths = load_scanpaths(fixation_path)
    report = quality_report(scanpaths, config.canvas_width_px, config.canvas_height_px)
    features = extract_feature_table(scanpaths, config)
    comparisons, null_record = _reference_distribution(config)
    summary = build_summary(features, config, comparisons)
    output = root / config.output_dir
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    quality_table(report).to_csv(table_dir / "data_quality.csv", index=False)
    schema_comparison(report).to_csv(table_dir / "schema_comparison.csv", index=False)
    subject_coverage_table(scanpaths).to_csv(table_dir / "subject_coverage.csv", index=False)
    split_counts_table(scanpaths).to_csv(table_dir / "split_counts.csv", index=False)
    pd.DataFrame(
        [{"n_subjects": n, "n_images": count} for n, count in report["images_by_n_subjects"].items()]
    ).to_csv(table_dir / "images_by_subject_count.csv", index=False)
    summary.to_csv(table_dir / "metrics_summary.csv", index=False)
    pooled_marginals(scanpaths, config.canvas_width_px, config.canvas_height_px).to_csv(
        table_dir / "descriptive_marginals.csv", index=False
    )
    _group_table(features, config, "subject_id").to_csv(table_dir / "subject_metrics.csv", index=False)
    _group_table(features, config, "image_name").to_csv(table_dir / "image_metrics.csv", index=False)
    features.to_csv(table_dir / "scanpath_level.csv", index=False)
    (output / "uniform_null.json").write_text(json.dumps(null_record, indent=2) + "\n")

    _write_density(scanpaths, config, table_dir, figure_dir)
    _write_marginal_figures(scanpaths, config, figure_dir)
    _write_subject_figure(features, null_record["comparison_mean_radius_px"], figure_dir)
    image_note = _write_scanpath_figures(root, config, scanpaths, figure_dir)
    _write_captions(figure_dir, config)

    verification = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
        "pyyaml": yaml.__version__,
        "fixation_path": config.fixation_path,
        "fixation_sha256": _sha256(fixation_path),
        "fixation_bytes": fixation_path.stat().st_size,
        "n_scanpaths": report["n_scanpaths"],
        "n_fixations": report["n_fixations"],
        "n_images": report["n_images"],
        "n_subjects": report["n_subjects"],
        "subject_ids": report["subject_ids"],
        "uniform_crosscheck_abs_diff_px": null_record["abs_diff_mc_minus_grid_px"],
        "scanpath_figures": image_note,
        "primary_fixation_set": "post_initial" if config.drop_initial_fixation else "all_released",
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    report_text = _text_report(verification, report, summary, null_record)
    (output / "verification_report.txt").write_text(report_text)
    resolved = {
        "config": _config_dict(config),
        "versions": {key: verification[key] for key in ("package_version", "python", "numpy", "pandas", "scipy", "matplotlib", "pillow", "pyyaml")},
        "generated_at_utc": verification["generated_at_utc"],
        "fixation_sha256": verification["fixation_sha256"],
    }
    (output / "analysis_config_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(report_text)
    return {"report": report, "summary": summary, "verification": verification}


def build_summary(features: pd.DataFrame, config: AnalysisConfig, comparisons: dict[str, float]) -> pd.DataFrame:
    rows = []
    for spec in _metric_specs(config):
        for subset in spec["subsets"]:
            for estimand in spec["estimands"]:
                mask = _subset_mask(features, subset)
                group_col = "subject_id" if estimand == "subject" else "image_name"
                seed = metric_seed(config.bootstrap_seed, spec["metric"], subset, estimand)
                stats = summarize_column(features, spec["column"], group_col, mask, config.n_bootstrap, seed)
                comparison_name = spec["comparison_key"] or ""
                comparison_value = comparisons.get(comparison_name, np.nan) if comparison_name else np.nan
                estimate = stats["estimate"]
                rows.append(
                    {
                        "role": spec["role"],
                        "metric": spec["metric"],
                        "column": spec["column"],
                        "subset": subset,
                        "estimand": estimand,
                        "weighting": WEIGHTING,
                        "n_groups": stats["n_groups"],
                        "n_scanpaths": stats["n_scanpaths"],
                        "estimate": estimate,
                        "sd": stats["sd"],
                        "sem": stats["sem"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "n_bootstrap": config.n_bootstrap,
                        "bootstrap_seed": seed,
                        "comparison_name": comparison_name,
                        "comparison_value": comparison_value,
                        "estimate_minus_comparison": estimate - comparison_value if comparison_name else np.nan,
                        "details": spec["details"],
                    }
                )
    return pd.DataFrame(rows)


def _metric_specs(config: AnalysisConfig) -> list[dict]:
    family = "post" if config.drop_initial_fixation else "all"
    other = "all" if config.drop_initial_fixation else "post"
    radius = number_token(config.center_radius_primary_px)
    k = config.early_k_primary
    t = number_token(config.early_time_primary_ms)
    included = (
        "Fixation index 0 is included."
        if family == "all"
        else "Fixation index 0 is excluded. The readme calls that point the initial center fixation."
    )
    specs = [
        _spec("primary", f"mean_radial_distance_{family}_px", f"mean_radius_{family}", "uniform_mean_radius_px",
              f"{included} Mean Euclidean distance to the canvas center. Comparison is the mean distance from the center to every 1 px cell center on the canvas.",
              ("all", "train", "valid"), ("subject", "image")),
        _spec("sensitivity", f"mean_radial_distance_{other}_px", f"mean_radius_{other}", "uniform_mean_radius_px",
              "The other inclusion rule for fixation index 0, same radius definition.",
              ("all",), ("subject", "image")),
        _spec("primary", f"fraction_within_{radius}px_{family}", f"frac_within_{radius}_{family}", f"disk_fraction_{radius}",
              f"{included} Share of fixations within {radius} px of the canvas center. Comparison is the disk area divided by the canvas area.",
              ("all", "train", "valid"), ("subject", "image")),
        _spec("primary", f"mean_saccade_{family}_px", f"mean_saccade_{family}", None,
              f"{included} Mean successive Euclidean distance.",
              ("all", "train", "valid"), ("subject", "image")),
        _spec("sensitivity", f"mean_saccade_{other}_px", f"mean_saccade_{other}", None,
              "The other inclusion rule for fixation index 0, same successive-distance definition.",
              ("all",), ("subject", "image")),
        _spec("primary", f"shuffled_minus_real_saccade_{family}_px", f"shuffle_minus_real_{family}", None,
              f"{included} Mean successive distance after within-scanpath shuffles, minus the released-order mean. "
              f"{config.n_shuffles} permutations, seed {config.shuffle_seed}.",
              ("all", "train", "valid"), ("subject", "image")),
        _spec("sensitivity", f"shuffled_minus_real_saccade_{other}_px", f"shuffle_minus_real_{other}", None,
              "Shuffle contrast under the other inclusion rule for fixation index 0.",
              ("all",), ("subject", "image")),
        _spec("primary", f"late_minus_early_radius_{family}_k{k}_px", f"late_minus_early_radius_{family}_k{k}", None,
              f"{included} Ordinal split at the first {k} analyzed fixations. Late mean radius minus early mean radius. "
              "The boundary saccade is excluded from the saccade contrast and does not enter this radius contrast.",
              ("all", "train", "valid"), ("subject", "image")),
        _spec("primary", f"late_minus_early_saccade_{family}_k{k}_px", f"late_minus_early_saccade_{family}_k{k}", None,
              f"{included} Ordinal split at the first {k} analyzed fixations. Late mean successive distance minus early mean successive distance. "
              "The saccade that crosses the split is excluded.",
              ("all", "train", "valid"), ("subject", "image")),
        _spec("primary", f"late_minus_early_radius_{family}_t{t}_px", f"late_minus_early_radius_{family}_t{t}", None,
              f"{included} A fixation is early when its trial-relative onset is below {t}. "
              "Onset is the sum of released durations of earlier fixations. Saccade time is not in the file. "
              "The working unit is milliseconds; see docs/ASSUMPTIONS.md.",
              ("all", "train", "valid"), ("subject", "image")),
        _spec("primary", f"late_minus_early_saccade_{family}_t{t}_px", f"late_minus_early_saccade_{family}_t{t}", None,
              f"{included} Temporal saccade contrast at onset threshold {t}, with the crossing saccade excluded.",
              ("all", "train", "valid"), ("subject", "image")),
    ]
    for extra_k in config.early_k_sensitivity:
        specs.append(_spec(
            "sensitivity", f"late_minus_early_radius_{family}_k{extra_k}_px",
            f"late_minus_early_radius_{family}_k{extra_k}", None,
            f"{included} Sensitivity of the ordinal radius contrast at k={extra_k}.",
            ("all",), ("subject",),
        ))
        specs.append(_spec(
            "sensitivity", f"late_minus_early_saccade_{family}_k{extra_k}_px",
            f"late_minus_early_saccade_{family}_k{extra_k}", None,
            f"{included} Sensitivity of the ordinal saccade contrast at k={extra_k}.",
            ("all",), ("subject",),
        ))
    for extra_t in config.early_time_sensitivity_ms:
        token = number_token(extra_t)
        specs.append(_spec(
            "sensitivity", f"late_minus_early_radius_{family}_t{token}_px",
            f"late_minus_early_radius_{family}_t{token}", None,
            f"{included} Sensitivity of the temporal radius contrast at threshold {token}.",
            ("all",), ("subject",),
        ))
    for extra_r in config.center_radii_px:
        token = number_token(extra_r)
        if token == radius:
            continue
        specs.append(_spec(
            "sensitivity", f"fraction_within_{token}px_{family}", f"frac_within_{token}_{family}",
            f"disk_fraction_{token}",
            f"{included} Sensitivity of the central-disk share at radius {token} px.",
            ("all",), ("subject",),
        ))
    return specs


def _spec(role, metric, column, comparison, details, subsets, estimands) -> dict:
    return {
        "role": role,
        "metric": metric,
        "column": column,
        "comparison_key": comparison,
        "details": details,
        "subsets": subsets,
        "estimands": estimands,
    }


def _subset_mask(features: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "all":
        return pd.Series(True, index=features.index)
    if subset in {"train", "valid"}:
        return features["split"] == subset
    raise ValueError(f"unknown subset {subset}")


def _reference_distribution(config: AnalysisConfig) -> tuple[dict[str, float], dict]:
    mc = uniform_mean_radius(config.canvas_width_px, config.canvas_height_px, config.uniform_mc_n, config.uniform_mc_seed)
    grid_step = 1.0
    grid = grid_mean_radius(config.canvas_width_px, config.canvas_height_px, step=grid_step)
    # The reported comparison is the deterministic grid. A single Monte Carlo
    # sample of a few hundred thousand draws can sit about 1 px away from it.
    comparisons = {"uniform_mean_radius_px": grid}
    fractions = {}
    for radius in config.center_radii_px:
        token = number_token(radius)
        value = disk_fraction(config.canvas_width_px, config.canvas_height_px, radius)
        comparisons[f"disk_fraction_{token}"] = value
        fractions[token] = value
    record = {
        "canvas_width_px": config.canvas_width_px,
        "canvas_height_px": config.canvas_height_px,
        "canvas_source": config.canvas_source,
        "support": "[0, width] x [0, height]",
        "mc_n": config.uniform_mc_n,
        "mc_seed": config.uniform_mc_seed,
        "mc_mean_radius_px": mc["mean"],
        "mc_se_radius_px": mc["se"],
        "grid_step_px": grid_step,
        "grid_mean_radius_px": grid,
        "comparison_mean_radius_px": grid,
        "comparison_definition": "mean distance from the canvas center to the center of every 1 px cell",
        "abs_diff_mc_minus_grid_px": abs(mc["mean"] - grid),
        "disk_fraction_by_radius_px": fractions,
    }
    return comparisons, record


def _group_table(features: pd.DataFrame, config: AnalysisConfig, group_col: str) -> pd.DataFrame:
    family = "post" if config.drop_initial_fixation else "all"
    radius = number_token(config.center_radius_primary_px)
    k = config.early_k_primary
    t = number_token(config.early_time_primary_ms)
    columns = [
        f"mean_radius_{family}",
        f"mean_saccade_{family}",
        f"shuffle_minus_real_{family}",
        f"late_minus_early_radius_{family}_k{k}",
        f"late_minus_early_saccade_{family}_k{k}",
        f"late_minus_early_radius_{family}_t{t}",
        f"late_minus_early_saccade_{family}_t{t}",
        f"frac_within_{radius}_{family}",
        "duration_sum_all",
        "n_fixations",
    ]
    grouped = features.groupby(group_col, sort=True)
    means = grouped[columns].mean()
    means["n_scanpaths"] = grouped.size()
    if group_col == "image_name":
        means["split"] = grouped["split"].first()
    means = means.reset_index()
    return means


def _write_density(scanpaths, config: AnalysisConfig, table_dir: Path, figure_dir: Path) -> None:
    k = config.early_k_primary
    edges_x, edges_y, mean, sd, _subjects = subject_mean_density(
        scanpaths, config.canvas_width_px, config.canvas_height_px,
        config.density_bins_x, config.density_bins_y, post_initial,
    )
    density_table(edges_x, edges_y, mean, sd).to_csv(table_dir / "density_post_initial_subject_mean.csv", index=False)
    plot_density_pair(
        mean, sd, config.canvas_width_px, config.canvas_height_px,
        figure_dir / "fixation_density_post_initial.png",
        "Post-initial fixations. Within-subject histogram, then unweighted mean and SD across subjects.",
    )
    _ex, _ey, early, _early_sd, _ = subject_mean_density(
        scanpaths, config.canvas_width_px, config.canvas_height_px,
        config.density_bins_x, config.density_bins_y,
        lambda sp: ordinal_epoch(sp, k, "early"),
    )
    _lx, _ly, late, _late_sd, _ = subject_mean_density(
        scanpaths, config.canvas_width_px, config.canvas_height_px,
        config.density_bins_x, config.density_bins_y,
        lambda sp: ordinal_epoch(sp, k, "late"),
    )
    plot_early_late(
        early, late, config.canvas_width_px, config.canvas_height_px,
        figure_dir / "fixation_density_early_vs_late.png",
        f"Ordinal epochs on post-initial fixations, k={k}. Each map is a subject-mean share.",
    )


def _write_marginal_figures(scanpaths, config: AnalysisConfig, figure_dir: Path) -> None:
    radii = _pooled_radii(scanpaths, config.canvas_width_px, config.canvas_height_px)
    uniform = uniform_radius_sample(
        config.canvas_width_px, config.canvas_height_px, config.uniform_mc_n, config.uniform_mc_seed,
    )
    plot_radius_hist(radii, uniform, figure_dir / "radial_distance_vs_uniform.png")
    real = _pooled_saccades(scanpaths)
    shuffled = _one_shuffle_saccades(scanpaths, config.shuffle_seed + 1)
    plot_saccade_hist(real, shuffled, figure_dir / "saccade_distance_real_vs_shuffled.png")


def _write_subject_figure(features: pd.DataFrame, uniform_mean: float, figure_dir: Path) -> None:
    means = features.groupby("subject_id")["mean_radius_post"].mean()
    plot_subject_radius(
        means.index.to_numpy(),
        means.to_numpy(),
        float(means.mean()),
        uniform_mean,
        figure_dir / "subject_mean_radial_distance.png",
    )


def _write_scanpath_figures(root: Path, config: AnalysisConfig, scanpaths: list[Scanpath], figure_dir: Path) -> str:
    manifest_path = root / config.output_dir / "tables" / "image_subset_manifest.csv"
    image_dir = root / config.image_dir
    if not manifest_path.exists():
        return "skipped: image_subset_manifest.csv is absent"
    manifest = pd.read_csv(manifest_path)
    by_image: dict[str, list[Scanpath]] = {}
    wanted = set(manifest["image_name"])
    for scanpath in scanpaths:
        if scanpath.image_name in wanted:
            by_image.setdefault(scanpath.image_name, []).append(scanpath)
    panels = []
    for record in manifest.head(6).to_dict(orient="records"):
        name = record["image_name"]
        path = image_dir / name
        if not path.exists() or name not in by_image:
            continue
        chosen = sorted(by_image[name], key=lambda sp: sp.subject_id)[0]
        panels.append(
            {
                "image_path": path,
                "x": chosen.x,
                "y": chosen.y,
                "duration": chosen.duration,
                "title": f"{name}  subject {chosen.subject_id}  {chosen.split}",
            }
        )
    if panels:
        plot_scanpath_grid(panels, figure_dir / "scanpath_examples.png")
    first = manifest.iloc[0]
    first_path = image_dir / first["image_name"]
    if first_path.exists() and first["image_name"] in by_image:
        series = []
        for scanpath in sorted(by_image[first["image_name"]], key=lambda sp: sp.subject_id):
            series.append({"subject_id": scanpath.subject_id, "x": scanpath.x, "y": scanpath.y})
        plot_all_subjects(
            {
                "image_path": first_path,
                "title": f"{first['image_name']}  all released subjects",
                "series": series,
            },
            figure_dir / "scanpath_all_subjects_example.png",
        )
    return f"wrote scanpath figures for {len(panels)} subset images"


def _write_captions(figure_dir: Path, config: AnalysisConfig) -> None:
    text = f"""fixation_density_post_initial.png
Subject-mean spatial histogram of fixations after index 0. Bins: {config.density_bins_x} by {config.density_bins_y} on the {config.canvas_width_px:g} by {config.canvas_height_px:g} canvas. The SD panel is the sample standard deviation across subject maps.

fixation_density_early_vs_late.png
Same histogram construction, split at the first {config.early_k_primary} post-initial fixations. Color scales are shared.

radial_distance_vs_uniform.png
Fixation-weighted distances to the canvas center, compared with one Monte Carlo uniform sample (seed and n are in configs/analysis.yaml). That sample mean can differ from the 1 px grid mean by about its standard error. metrics_summary.csv uses the grid mean. This histogram is not the subject-equal estimand.

saccade_distance_real_vs_shuffled.png
One permutation of each post-initial sequence, seed {config.shuffle_seed + 1}. The subject-equal shuffle contrast uses the mean of {config.n_shuffles} permutations and is in metrics_summary.csv.

subject_mean_radial_distance.png
One point per subject id. The blue line is the unweighted mean of those points. The dashed line is the mean distance from the canvas center to 1 px cell centers, the same comparison value as metrics_summary.csv.

scanpath_examples.png
Six seeded subset images. The plotted subject is the lowest subject id present for that image. Coordinates are placed on a continuous canvas with extent [0, width] by [0, height]. Marker area increases with the released duration value.

scanpath_all_subjects_example.png
Every subject id in the release for the first image in the seeded subset manifest.
"""
    (figure_dir / "captions.txt").write_text(text)


def _pooled_radii(scanpaths: list[Scanpath], width: float, height: float) -> np.ndarray:
    cx = width / 2.0
    cy = height / 2.0
    chunks = [np.hypot(sp.x[1:] - cx, sp.y[1:] - cy) for sp in scanpaths if len(sp.x) > 1]
    return np.concatenate(chunks)


def _pooled_saccades(scanpaths: list[Scanpath]) -> np.ndarray:
    chunks = [np.hypot(np.diff(sp.x[1:]), np.diff(sp.y[1:])) for sp in scanpaths if len(sp.x) > 2]
    return np.concatenate(chunks)


def _one_shuffle_saccades(scanpaths: list[Scanpath], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for scanpath in scanpaths:
        x = scanpath.x[1:]
        y = scanpath.y[1:]
        if len(x) < 2:
            continue
        perm = rng.permutation(len(x))
        chunks.append(np.hypot(np.diff(x[perm]), np.diff(y[perm])))
    return np.concatenate(chunks)


def _text_report(verification: dict, report: dict, summary: pd.DataFrame, null_record: dict) -> str:
    primary = summary[(summary["role"] == "primary") & (summary["subset"] == "all") & (summary["estimand"] == "subject")]
    lines = [
        "HumanGazeLab verification",
        f"generated_at_utc: {verification['generated_at_utc']}",
        f"fixation_sha256: {verification['fixation_sha256']}",
        f"fixation_bytes: {verification['fixation_bytes']}",
        f"n_scanpaths: {report['n_scanpaths']}",
        f"n_fixations: {report['n_fixations']}",
        f"n_images: {report['n_images']}",
        f"n_subjects: {report['n_subjects']}",
        f"subject_ids: {report['subject_ids']}",
        f"split_scanpath_counts: {report['split_scanpath_counts']}",
        f"split_image_counts: {report['split_image_counts']}",
        f"condition_counts: {report['condition_counts']}",
        f"n_fixations_outside_canvas: {report['n_fixations_outside_canvas']}",
        f"duration_range: {report['duration_min']} {report['duration_max']}",
        f"duration_sum_median: {report['duration_sum_median']}",
        f"n_scanpaths_duration_sum_gt_5000: {report['n_scanpaths_duration_sum_gt_5000']}",
        f"n_durations_gt_5000: {report['n_durations_gt_5000']}",
        f"fraction_first_fixation_within_50px: {report['fraction_first_fixation_within_50px']}",
        f"uniform_mc_mean_radius_px: {null_record['mc_mean_radius_px']}",
        f"uniform_mc_se_radius_px: {null_record['mc_se_radius_px']}",
        f"uniform_grid_mean_radius_px: {null_record['grid_mean_radius_px']}",
        f"uniform_crosscheck_abs_diff_px: {null_record['abs_diff_mc_minus_grid_px']}",
        f"scanpath_figures: {verification['scanpath_figures']}",
        "",
        "Primary subject-equal estimates on subset=all",
        "metric\testimate\tci_low\tci_high\tcomparison\tn_groups\tn_scanpaths",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"{row.metric}\t{row.estimate:.6g}\t{row.ci_low:.6g}\t{row.ci_high:.6g}\t"
            f"{row.comparison_value}\t{row.n_groups}\t{row.n_scanpaths}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _config_dict(config: AnalysisConfig) -> dict:
    return {key: (list(value) if isinstance(value, tuple) else value) for key, value in config.__dict__.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
