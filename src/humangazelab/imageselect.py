"""Deterministic image-subset selection. Selection does not look at gaze values."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_image_subset(
    image_table: pd.DataFrame,
    n_train: int,
    n_valid: int,
    seed: int,
    min_subjects: int,
) -> pd.DataFrame:
    """Sample images with a seeded Generator.

    ``image_table`` needs columns ``image_name``, ``split``, and ``n_subjects``.
    Train is drawn first, then validation, from names sorted within each split.
    The returned names are sorted within split for stable filenames.
    """
    required = {"image_name", "split", "n_subjects"}
    missing = required - set(image_table.columns)
    if missing:
        raise ValueError(f"image table is missing {sorted(missing)}")
    rng = np.random.default_rng(seed)
    frames = [
        _pick(image_table, "train", n_train, rng, min_subjects),
        _pick(image_table, "valid", n_valid, rng, min_subjects),
    ]
    return pd.concat(frames, ignore_index=True)


def _pick(image_table: pd.DataFrame, split: str, n: int, rng: np.random.Generator, min_subjects: int) -> pd.DataFrame:
    pool = image_table[(image_table["split"] == split) & (image_table["n_subjects"] >= min_subjects)]
    pool = pool.sort_values("image_name")
    if len(pool) < n:
        raise ValueError(f"{split} has {len(pool)} images with at least {min_subjects} subjects; asked for {n}")
    chosen = rng.choice(len(pool), size=n, replace=False)
    out = pool.iloc[np.sort(chosen)].copy()
    out["selection"] = f"seeded sample of {n} {split} images with n_subjects>={min_subjects}"
    return out
