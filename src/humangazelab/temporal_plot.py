"""Figures for temporal_v1. Axes state the metric and the split. They do not state a conclusion."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "ig_B0_uniform": "#222222",
    "ig_B1_center": "#d95f02",
    "ig_B1_subject_held_out": "#e6ab02",
    "ig_B2_loo_image": "#1b9e77",
    "ig_B3_loo_index": "#66a61e",
    "ig_B4_history": "#7570b3",
    "ig_B5_center_history": "#e7298a",
    "ig_B5c_loo_history": "#a6761d",
    "ig_B6_image_shuffle": "#1f78b4",
    "ig_B6_order_shuffle": "#80b1d3",
}

CURVE_MODELS = [
    "ig_B0_uniform",
    "ig_B1_center",
    "ig_B2_loo_image",
    "ig_B3_loo_index",
    "ig_B4_history",
    "ig_B5_center_history",
    "ig_B6_image_shuffle",
    "ig_B6_order_shuffle",
]


def _label(metric: str) -> str:
    return metric.removeprefix("ig_").replace("_", " ")


def write_figures(output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(output_dir / "information_gain_summary.csv")
    contrasts = pd.read_csv(output_dir / "paired_contrasts.csv")
    subjects = pd.read_csv(output_dir / "subject_information_gain_valid.csv")
    images = pd.read_csv(output_dir / "image_information_gain_valid.csv")
    counts = pd.read_csv(output_dir / "sample_counts_by_index.csv")
    reliability = pd.read_csv(output_dir / "center_bias_spatial_reliability.csv")
    _curve(summary, figure_dir / "temporal_information_gain_valid.png")
    _bars(summary, figure_dir / "model_comparison_valid.png")
    _subjects(subjects, figure_dir / "observer_information_gain_valid.png")
    _images(images, figure_dir / "image_heterogeneity_valid.png")
    _contrasts(contrasts, figure_dir / "paired_contrasts_valid.png")
    _counts(counts, figure_dir / "sample_size_by_index.png")
    _reliability(reliability, figure_dir / "center_bias_reliability_valid.png")
    _epochs(summary, figure_dir / "epoch_information_gain_valid.png")
    index0_path = output_dir / "index0_sensitivity.csv"
    if index0_path.exists():
        _index0(pd.read_csv(index0_path), figure_dir / "index0_sensitivity_valid.png")
    caption = """\
temporal_information_gain_valid.png
Validation images. Subject-equal mean information gain in bits per fixation, after an equal-weight mean inside each scanpath. The band is the percentile bootstrap of the 10 subject means. Index 0 is not a target. B2 and B3 use other observers on the same validation image. B6 image uses a within-split image derangement. B6 order uses a non-predecessor fixation from the same scanpath.

model_comparison_valid.png
Same aggregation, all primary indices together. Whiskers are the subject-bootstrap interval.

observer_information_gain_valid.png
One point is one subject's scanpath-equal mean on validation images.

image_heterogeneity_valid.png
One observation is one validation image's unweighted mean information gain. This is not a subject-level interval.

paired_contrasts_valid.png
Paired subject-level differences. Early-minus-late contrasts use only scanpaths whose released length is at least 12. An interval that excludes zero is a statement about this resample, not a causal claim.

sample_size_by_index.png
Released scanpaths that still have a fixation at that index. Later indices are a selected subset of longer scanpaths.

center_bias_reliability_valid.png
Validation fixations against the train-fitted B1 map. Each point pools an equal number of bins ordered by predicted probability. The vertical axis is the share of validation fixations in those bins. The horizontal axis is the share of probability the map assigns to those bins.

epoch_information_gain_valid.png
Pre-specified index groups on validation images: 1-3, 4-8, and 9 and later. Later groups contain fewer scanpaths.

index0_sensitivity_valid.png
Released index 0 scored as a target. The primary analysis does not include this index as a target. The two B1 bars use the primary map and a map refit with index 0 included. B2 here uses only other observers' index-0 locations.
"""
    (figure_dir / "captions.txt").write_text(caption)


def _valid_subject(summary: pd.DataFrame) -> pd.DataFrame:
    part = summary[summary["estimand"] == "subject_scanpath_equal"].copy()
    return part[part["subset"].str.startswith("valid|")]


def _curve(summary: pd.DataFrame, path: Path) -> None:
    part = _valid_subject(summary)
    part = part[part["subset"].str.contains(r"\|index_\d+$", regex=True)]
    fig, ax = plt.subplots(figsize=(10.6, 4.6), constrained_layout=True)
    for metric in CURVE_MODELS:
        rows = part[part["metric"] == metric].copy()
        if rows.empty:
            continue
        rows["index"] = rows["subset"].str.extract(r"index_(\d+)$").astype(float)
        rows = rows.sort_values("index")
        ax.plot(rows["index"], rows["estimate"], color=COLORS[metric], label=_label(metric), linewidth=1.6)
        ax.fill_between(rows["index"], rows["ci_low"], rows["ci_high"], color=COLORS[metric], alpha=0.15, linewidth=0)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Released fixation index")
    ax.set_ylabel("Information gain (bits per fixation)")
    ax.set_title("Validation images, subject-equal means")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _bars(summary: pd.DataFrame, path: Path) -> None:
    order = CURVE_MODELS + ["ig_B1_subject_held_out", "ig_B5c_loo_history"]
    part = _valid_subject(summary)
    part = part[part["subset"] == "valid|all_primary_indices"]
    part = part[part["metric"].isin(order)].copy()
    part["metric"] = pd.Categorical(part["metric"], order, ordered=True)
    part = part.sort_values("metric")
    fig, ax = plt.subplots(figsize=(10.6, 4.4), constrained_layout=True)
    x = np.arange(len(part))
    ax.bar(x, part["estimate"], color=[COLORS.get(m, "#333333") for m in part["metric"]], width=0.72)
    ax.errorbar(x, part["estimate"], yerr=[part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]], fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x, [_label(m) for m in part["metric"]], rotation=35, ha="right")
    ax.set_ylabel("Information gain (bits per fixation)")
    ax.set_title("Validation images, all primary indices")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _subjects(subjects: pd.DataFrame, path: Path) -> None:
    models = [m for m in CURVE_MODELS if m in set(subjects["model"])]
    fig, ax = plt.subplots(figsize=(10.6, 4.4), constrained_layout=True)
    for position, metric in enumerate(models):
        values = subjects.loc[subjects["model"] == metric, "mean_ig_bits"].to_numpy()
        ax.scatter(np.full(values.shape, position), values, s=28, color=COLORS[metric], zorder=3)
        ax.hlines(np.mean(values), position - 0.2, position + 0.2, color="black", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(models)), [_label(m) for m in models], rotation=35, ha="right")
    ax.set_ylabel("Subject mean information gain (bits)")
    ax.set_title("Ten observers, validation images")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _images(images: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.2), constrained_layout=True)
    bins = 30
    ax.hist(images["ig_B1_center"], bins=bins, histtype="step", linewidth=1.6, label="B1 center", color=COLORS["ig_B1_center"])
    ax.hist(images["ig_B2_loo_image"], bins=bins, histtype="step", linewidth=1.6, label="B2 same-image LOO", color=COLORS["ig_B2_loo_image"])
    ax.hist(images["ig_B4_history"], bins=bins, histtype="step", linewidth=1.6, label="B4 history", color=COLORS["ig_B4_history"])
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Image-mean information gain (bits per fixation)")
    ax.set_ylabel("Validation images")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _contrasts(contrasts: pd.DataFrame, path: Path) -> None:
    part = contrasts[contrasts["estimand"] == "subject_scanpath_equal"].copy()
    part = part[~part["subset"].str.startswith("train")]
    fig, ax = plt.subplots(figsize=(10.6, 5.2), constrained_layout=True)
    y = np.arange(len(part))
    ax.errorbar(part["estimate"], y, xerr=[part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]], fmt="o", color="#222222", ecolor="#222222", elinewidth=0.8, capsize=2, markersize=4)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_yticks(y, part["contrast"].str.replace("ig_", "", regex=False))
    ax.set_xlabel("Paired difference (bits per fixation)")
    ax.set_title("Subject-level paired differences")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _counts(counts: pd.DataFrame, path: Path) -> None:
    numeric = counts[pd.to_numeric(counts["fixation_index"], errors="coerce").notna()].copy()
    numeric["fixation_index"] = numeric["fixation_index"].astype(int)
    fig, ax = plt.subplots(figsize=(10.4, 4.0), constrained_layout=True)
    ax.plot(numeric["fixation_index"], numeric["n_scanpaths"], color="#1b4f72", label="Scanpaths")
    ax.plot(numeric["fixation_index"], numeric["n_images"], color="#b9770e", label="Images")
    ax.set_xlabel("Released fixation index")
    ax.set_ylabel("Count still observed")
    ax.set_title("How many scanpaths reach each index")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _reliability(reliability: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.8)
    ax.plot(reliability["predicted_bin_share"], reliability["observed_fixation_share"], marker="o", color=COLORS["ig_B1_center"])
    ax.set_xlabel("Predicted share of bin probability")
    ax.set_ylabel("Observed share of validation fixations")
    ax.set_title("B1 center map, validation")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _index0(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    x = np.arange(len(table))
    ax.bar(x, table["estimate"], color="#1b4f72", width=0.7)
    ax.errorbar(
        x,
        table["estimate"],
        yerr=[table["estimate"] - table["ci_low"], table["ci_high"] - table["estimate"]],
        fmt="none",
        ecolor="black",
        capsize=2,
    )
    ax.set_xticks(x, table["metric"].str.replace("ig_", "", regex=False), rotation=20, ha="right")
    ax.set_ylabel("Information gain (bits)")
    ax.set_title("Validation index 0 only, not a primary target")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _epochs(summary: pd.DataFrame, path: Path) -> None:
    wanted = {
        "valid|early_indices_1_to_3": "1–3",
        "valid|middle_indices_4_to_8": "4–8",
        "valid|late_indices_9_and_later": "9+",
    }
    models = ["ig_B1_center", "ig_B2_loo_image", "ig_B4_history", "ig_B5_center_history"]
    part = _valid_subject(summary)
    part = part[part["subset"].isin(wanted) & part["metric"].isin(models)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2), constrained_layout=True)
    epochs = list(wanted)
    width = 0.18
    for offset, metric in enumerate(models):
        rows = part[part["metric"] == metric].set_index("subset").loc[epochs]
        x = np.arange(len(epochs)) + (offset - 1.5) * width
        ax.bar(x, rows["estimate"], width=width, color=COLORS[metric], label=_label(metric))
        ax.errorbar(x, rows["estimate"], yerr=[rows["estimate"] - rows["ci_low"], rows["ci_high"] - rows["estimate"]], fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(epochs)), [wanted[item] for item in epochs])
    ax.set_xlabel("Fixation-index group")
    ax.set_ylabel("Information gain (bits per fixation)")
    ax.set_title("Validation images, pre-specified groups")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)
