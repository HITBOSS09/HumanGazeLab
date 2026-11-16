"""Out-of-fold event calibration. Run only after the descriptive audit.

This does not modify DeepGaze III and does not emit a spatial density.
A lower region-event loss is not a higher scanpath likelihood.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from humangazelab.semantic_calibration import brier_score, log_loss, stage_d_probabilities
from humangazelab.summarize import metric_seed, percentile_bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "semantic_calibration_v1.yaml"


def _score_block(frame: pd.DataFrame, config: dict, label: str) -> list[dict]:
    stage = config["stage_d"]
    predicted = stage_d_probabilities(
        frame,
        n_folds=int(stage["folds"]),
        seed=int(stage["seed"]),
        l2=float(stage["l2"]),
        steps=int(stage["newton_steps"]),
        covariate_columns=tuple(stage["covariate_columns"]),
    )
    rows = []
    for column, name in (
        ("p_unmodified", "unmodified"),
        ("p_intercept", "intercept_only"),
        ("p_platt", "platt"),
        ("p_covariates", "covariates"),
    ):
        usable = predicted.dropna(subset=[column])
        image_brier = usable.groupby("image").apply(
            lambda part, col=column: brier_score(part[col], part["y"]),
            include_groups=False,
        )
        base = usable.groupby("image").apply(
            lambda part: brier_score(part["p_unmodified"], part["y"]),
            include_groups=False,
        )
        aligned = image_brier.align(base, join="inner")
        difference = aligned[0] - aligned[1]
        boot = percentile_bootstrap_mean(
            difference.to_numpy(),
            int(config["bootstrap_draws"]),
            metric_seed(int(stage["seed"]), label, name, "brier_minus_unmodified"),
        )
        rows.append(
            {
                "label": label,
                "model": name,
                "n_fixations": int(len(usable)),
                "n_images": int(usable["image"].nunique()),
                "brier": brier_score(usable[column], usable["y"]),
                "log_loss": log_loss(usable[column], usable["y"]),
                "image_brier_minus_unmodified": boot["estimate"],
                "image_brier_difference_ci_low": boot["ci_low"],
                "image_brier_difference_ci_high": boot["ci_high"],
            }
        )
    return rows


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("descriptive run_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise SystemExit("refusing Stage D before the descriptive run is complete")
    events = pd.read_csv(output / "region_events.csv")
    present = events[events["mask_present"] == 1].copy()
    present["prev_distance_px"] = np.hypot(present["x"] - present["prev_x"], present["y"] - present["prev_y"])
    rows = []
    for label in config["stage_d"]["labels"]:
        part = present[present["label"] == label]
        rows.extend(_score_block(part, config, label))
    pooled = present[present["label"].isin(config["stage_d"]["labels"])].copy()
    rows.extend(_score_block(pooled, config, "pooled_primary_labels"))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "stage_d_comparisons.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
