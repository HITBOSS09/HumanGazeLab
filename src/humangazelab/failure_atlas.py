"""Predeclared failure-case selection for temporal_v1.

Cases are chosen from validation-image means by quartile rules fixed in
configs/temporal_v1.yaml. The figure records the numbers that caused the
selection. It is not a search for a visually preferred example.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from humangazelab.temporal import Grid, information_gain


def select_cases(images: pd.DataFrame, n_examples: int, high_q: float, low_q: float, min_observers: int) -> pd.DataFrame:
    eligible = images[images["n_subjects"] >= min_observers].copy()
    b1_low, b1_high = eligible["ig_B1_center"].quantile([low_q, high_q])
    b2_low, b2_high = eligible["ig_B2_loo_image"].quantile([low_q, high_q])
    rows = []

    def take(frame: pd.DataFrame, rule: str, rank_column: str, ascending: bool, definition: str) -> None:
        chosen = frame.sort_values(rank_column, ascending=ascending).head(n_examples)
        for rank, (_, record) in enumerate(chosen.iterrows(), start=1):
            rows.append(
                {
                    "rule": rule,
                    "definition": definition,
                    "rank": rank,
                    "n_eligible_images": int(len(eligible)),
                    "n_images_meeting_rule": int(len(frame)),
                    "image": record["image"],
                    "n_subjects": int(record["n_subjects"]),
                    "n_fixations": int(record["n_fixations"]),
                    "ig_B1_center": float(record["ig_B1_center"]),
                    "ig_B2_loo_image": float(record["ig_B2_loo_image"]),
                    "ig_B4_history": float(record["ig_B4_history"]),
                    "ig_B3_loo_index": float(record["ig_B3_loo_index"]),
                    "rank_value": float(record[rank_column]),
                    "b1_low_quartile": float(b1_low),
                    "b1_high_quartile": float(b1_high),
                    "b2_low_quartile": float(b2_low),
                    "b2_high_quartile": float(b2_high),
                }
            )

    high_agree = eligible[(eligible["ig_B2_loo_image"] >= b2_high) & (eligible["ig_B1_center"] <= b1_low)].copy()
    high_agree["rank_value"] = high_agree["ig_B2_loo_image"] - high_agree["ig_B1_center"]
    take(
        high_agree,
        "high_cross_observer_low_center",
        "rank_value",
        False,
        "B2 at or above the high quantile and B1 at or below the low quantile; rank by B2-B1",
    )
    low_agree = eligible[(eligible["ig_B2_loo_image"] <= b2_low) & (eligible["ig_B1_center"] >= b1_high)].copy()
    low_agree["rank_value"] = low_agree["ig_B1_center"] - low_agree["ig_B2_loo_image"]
    take(
        low_agree,
        "low_cross_observer_high_center",
        "rank_value",
        False,
        "B2 at or below the low quantile and B1 at or above the high quantile; rank by B1-B2",
    )
    diverge = eligible.copy()
    diverge["rank_value"] = (diverge["ig_B2_loo_image"] - diverge["ig_B4_history"]).abs()
    take(diverge, "largest_B2_B4_gap", "rank_value", False, "Largest absolute B2 minus B4, no quartile filter")
    return pd.DataFrame(rows)


def index0_inclusion_cases(records, center: np.ndarray, grid: Grid, images: pd.DataFrame, n_examples: int, min_observers: int) -> pd.DataFrame:
    """Largest change in the fixation-weighted image mean when index 0 is added as a target.

    The B1 map is the primary map, which was fit without index 0. Index 0 is scored
    under that frozen map. This does not refit the map.
    """
    eligible = set(images.loc[images["n_subjects"] >= min_observers, "image"])
    buckets: dict[str, list[tuple[str, float]]] = {}
    for record in records:
        if record.split != "valid" or record.image not in eligible or record.length < 1:
            continue
        buckets.setdefault(record.image, [])
        primary = information_gain(center[record.iy[1:], record.ix[1:]], grid.n_bins) if record.length > 1 else np.empty(0)
        index0 = information_gain(center[record.iy[:1], record.ix[:1]], grid.n_bins)
        for value in primary:
            buckets[record.image].append(("primary", float(value)))
        buckets[record.image].append(("index0", float(index0[0])))
    rows = []
    for image, pairs in buckets.items():
        primary_values = [value for kind, value in pairs if kind == "primary"]
        all_values = [value for _, value in pairs]
        if not primary_values:
            continue
        mean_primary = float(np.mean(primary_values))
        mean_with = float(np.mean(all_values))
        rows.append(
            {
                "image": image,
                "mean_ig_primary": mean_primary,
                "mean_ig_including_index0": mean_with,
                "delta_including_minus_primary": mean_with - mean_primary,
                "abs_delta": abs(mean_with - mean_primary),
            }
        )
    ranked = pd.DataFrame(rows).sort_values("abs_delta", ascending=False).head(n_examples)
    base = images.set_index("image")
    out = []
    for rank, (_, record) in enumerate(ranked.iterrows(), start=1):
        source = base.loc[record["image"]]
        out.append(
            {
                "rule": "largest_change_when_index0_included",
                "definition": "Largest absolute change in the fixation-weighted B1 image mean when index 0 is added as a target under the frozen primary B1 map",
                "rank": rank,
                "n_eligible_images": int((images["n_subjects"] >= min_observers).sum()),
                "n_images_meeting_rule": int(len(rows)),
                "image": record["image"],
                "n_subjects": int(source["n_subjects"]),
                "n_fixations": int(source["n_fixations"]),
                "ig_B1_center": float(source["ig_B1_center"]),
                "ig_B2_loo_image": float(source["ig_B2_loo_image"]),
                "ig_B4_history": float(source["ig_B4_history"]),
                "ig_B3_loo_index": float(source["ig_B3_loo_index"]),
                "rank_value": float(record["abs_delta"]),
                "mean_ig_primary_B1": float(record["mean_ig_primary"]),
                "mean_ig_including_index0_B1": float(record["mean_ig_including_index0"]),
                "delta_including_minus_primary": float(record["delta_including_minus_primary"]),
            }
        )
    return pd.DataFrame(out)


def _draw_case(ax, scanpaths_for_image, image_path: Path | None, title: str) -> None:
    if image_path is not None and image_path.exists():
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image, extent=[0, image.width, image.height, 0], origin="upper")
    else:
        ax.set_xlim(0, 1680)
        ax.set_ylim(1050, 0)
        ax.set_facecolor("#f4f4f4")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    subjects = sorted({scan.subject_id for scan in scanpaths_for_image})
    for position, subject in enumerate(subjects):
        scans = [scan for scan in scanpaths_for_image if scan.subject_id == subject]
        for scan in scans:
            x = np.asarray(scan.x, dtype=float)
            y = np.asarray(scan.y, dtype=float)
            if x.size > 1:
                ax.plot(x[1:], y[1:], color=colors[position % 10], linewidth=0.6, alpha=0.8)
            if x.size:
                ax.scatter(x[0], y[0], s=12, facecolors="none", edgecolors="red", linewidths=0.6)
                if x.size > 1:
                    ax.scatter(x[1:], y[1:], s=8, color=colors[position % 10], linewidths=0)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def write_atlas(output_dir: Path, scanpaths, records, center, grid: Grid, image_dir: Path, atlas_cfg: dict) -> pd.DataFrame:
    images = pd.read_csv(output_dir / "image_information_gain_valid.csv")
    n_examples = int(atlas_cfg["n_examples"])
    cases = select_cases(
        images,
        n_examples,
        float(atlas_cfg["high_quantile"]),
        float(atlas_cfg["low_quantile"]),
        int(atlas_cfg["min_observers"]),
    )
    index_cases = index0_inclusion_cases(records, center, grid, images, n_examples, int(atlas_cfg["min_observers"]))
    all_cases = pd.concat([cases, index_cases], ignore_index=True, sort=False)
    by_image = {}
    for scan in scanpaths:
        if scan.split == "valid":
            by_image.setdefault(scan.image_name, []).append(scan)
    rules = list(dict.fromkeys(all_cases["rule"]))
    fig, axes = plt.subplots(len(rules), n_examples, figsize=(3.4 * n_examples, 3.1 * len(rules)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, rule in enumerate(rules):
        subset = all_cases[all_cases["rule"] == rule].sort_values("rank")
        for column in range(n_examples):
            ax = axes[row, column]
            if column >= len(subset):
                ax.axis("off")
                continue
            case = subset.iloc[column]
            path = image_dir / case["image"]
            overlay = "photo" if path.exists() else "coordinates only"
            title = (
                f"{rule} #{int(case['rank'])} {overlay}\n{case['image']}\n"
                f"B1 {case['ig_B1_center']:.2f}  B2 {case['ig_B2_loo_image']:.2f}  B4 {case['ig_B4_history']:.2f}"
            )
            _draw_case(ax, by_image.get(case["image"], []), path if path.exists() else None, title)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "failure_atlas_valid.png", dpi=140)
    plt.close(fig)
    all_cases.to_csv(output_dir / "failure_atlas_cases.csv", index=False)
    note = (
        "Selection uses validation-image means and the quartile cutoffs in configs/temporal_v1.yaml.\n"
        "A red open circle is released fixation index 0. Other marks are later fixations.\n"
        "Panels without a photograph are coordinate plots on the 1680 by 1050 canvas.\n"
        "The title numbers are the image-mean information gains from image_information_gain_valid.csv.\n"
        "Same-image B2 is a human-consensus reference, not an image-only predictor.\n"
    )
    (figure_dir / "failure_atlas_readme.txt").write_text(note)
    return all_cases
