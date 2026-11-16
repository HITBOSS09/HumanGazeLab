"""Per-scanpath spatial, ordinal, temporal, and shuffle features.

Definitions used by the primary table live in docs/METHODOLOGY.md.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from humangazelab.adapter import Scanpath
from humangazelab.config import AnalysisConfig


def mean_successive_distance(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    return float(np.hypot(np.diff(x), np.diff(y)).mean())


def shuffled_mean_distance(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    """Mean successive distance after permuting fixation order, averaged over permutations.

    The multiset of positions is unchanged. Only order changes.
    """
    n = len(x)
    if n < 2 or n_perm < 1:
        return math.nan
    total = 0.0
    order = np.arange(n)
    for _ in range(n_perm):
        perm = rng.permutation(order)
        total += float(np.hypot(np.diff(x[perm]), np.diff(y[perm])).mean())
    return total / n_perm


def epoch_distance_means(x: np.ndarray, y: np.ndarray, early_count: int) -> dict[str, float]:
    """Split a sequence into a prefix and a suffix.

    Saccades with both endpoints inside the prefix are early. Saccades with both
    endpoints inside the suffix are late. The single saccade that crosses the
    boundary is reported separately and is not included in either mean.
    """
    n = len(x)
    early_count = int(max(0, min(int(early_count), n)))

    def segment(start: int, stop: int) -> float:
        if stop - start < 2:
            return math.nan
        return float(np.hypot(np.diff(x[start:stop]), np.diff(y[start:stop])).mean())

    boundary = math.nan
    if 0 < early_count < n:
        boundary = float(math.hypot(float(x[early_count] - x[early_count - 1]), float(y[early_count] - y[early_count - 1])))
    return {
        "early": segment(0, early_count),
        "late": segment(early_count, n),
        "boundary": boundary,
        "n_early": float(early_count),
        "n_late": float(n - early_count),
    }


def trial_onset(duration: np.ndarray) -> np.ndarray:
    """Onset of each fixation as the sum of released durations before it.

    Saccade time is not in the release, so this is fixation-time only.
    """
    if np.any(duration < 0):
        raise ValueError("trial onset is defined for non-negative released durations")
    cumulative = np.cumsum(duration)
    return cumulative - duration


def temporal_early_count(onset: np.ndarray, threshold: float) -> int:
    """Number of leading points whose onset is strictly below ``threshold``."""
    if len(onset) == 0:
        return 0
    if np.any(np.diff(onset) < 0):
        raise ValueError("temporal prefix split requires nondecreasing onsets")
    return int(np.searchsorted(onset, threshold, side="left"))


def _mean_radius(x: np.ndarray, y: np.ndarray, cx: float, cy: float) -> float:
    if len(x) == 0:
        return math.nan
    return float(np.hypot(x - cx, y - cy).mean())


def _fraction_within(x: np.ndarray, y: np.ndarray, cx: float, cy: float, radius: float) -> float:
    if len(x) == 0:
        return math.nan
    return float(np.mean(np.hypot(x - cx, y - cy) <= radius))


def _difference(late: float, early: float) -> float:
    if math.isnan(late) or math.isnan(early):
        return math.nan
    return float(late - early)


def extract_feature_table(scanpaths: list[Scanpath], config: AnalysisConfig) -> pd.DataFrame:
    """One row per released scanpath.

    Shuffle permutations advance one Generator in ``source_index`` order.
    A sequence is permuted only when it contains at least two fixations:
    ``n_shuffles`` permutations of the post-initial sequence, then
    ``n_shuffles`` permutations of the full sequence.
    """
    cx = config.canvas_width_px / 2.0
    cy = config.canvas_height_px / 2.0
    rng = np.random.default_rng(config.shuffle_seed)
    rows: list[dict] = []
    ordinal_ks = [config.early_k_primary, *config.early_k_sensitivity]
    time_thresholds = [config.early_time_primary_ms, *config.early_time_sensitivity_ms]

    for scanpath in scanpaths:
        x_all = scanpath.x
        y_all = scanpath.y
        d_all = scanpath.duration
        x_post = x_all[1:]
        y_post = y_all[1:]
        d_post = d_all[1:]
        onset = trial_onset(d_all)
        row: dict = {
            "source_index": scanpath.source_index,
            "image_name": scanpath.image_name,
            "subject_id": scanpath.subject_id,
            "split": scanpath.split,
            "n_fixations": int(len(x_all)),
            "n_post": int(len(x_post)),
            "duration_sum_all": float(d_all.sum()) if len(d_all) else math.nan,
            "duration_sum_post": float(d_post.sum()) if len(d_post) else math.nan,
            "first_radius": float(math.hypot(float(x_all[0] - cx), float(y_all[0] - cy))),
            "mean_radius_all": _mean_radius(x_all, y_all, cx, cy),
            "mean_radius_post": _mean_radius(x_post, y_post, cx, cy),
            "mean_x_post": float(x_post.mean()) if len(x_post) else math.nan,
            "mean_y_post": float(y_post.mean()) if len(y_post) else math.nan,
            "mean_duration_post": float(d_post.mean()) if len(d_post) else math.nan,
            "mean_saccade_all": mean_successive_distance(x_all, y_all),
            "mean_saccade_post": mean_successive_distance(x_post, y_post),
        }
        for radius in config.center_radii_px:
            token = _token(radius)
            row[f"frac_within_{token}_post"] = _fraction_within(x_post, y_post, cx, cy, radius)
            row[f"frac_within_{token}_all"] = _fraction_within(x_all, y_all, cx, cy, radius)

        shuffled_post = shuffled_mean_distance(x_post, y_post, rng, config.n_shuffles)
        shuffled_all = shuffled_mean_distance(x_all, y_all, rng, config.n_shuffles)
        row["shuffled_mean_saccade_post"] = shuffled_post
        row["shuffled_mean_saccade_all"] = shuffled_all
        row["shuffle_minus_real_post"] = _difference(shuffled_post, row["mean_saccade_post"])
        row["shuffle_minus_real_all"] = _difference(shuffled_all, row["mean_saccade_all"])

        sequences = (
            ("post", x_post, y_post, onset[1:]),
            ("all", x_all, y_all, onset),
        )
        for label, x_seq, y_seq, onset_seq in sequences:
            for k in ordinal_ks:
                epochs = epoch_distance_means(x_seq, y_seq, k)
                radius_early = _mean_radius(x_seq[:k], y_seq[:k], cx, cy) if k > 0 else math.nan
                radius_late = _mean_radius(x_seq[k:], y_seq[k:], cx, cy)
                row[f"late_minus_early_radius_{label}_k{k}"] = _difference(radius_late, radius_early)
                row[f"late_minus_early_saccade_{label}_k{k}"] = _difference(epochs["late"], epochs["early"])
            for threshold in time_thresholds:
                early_count = temporal_early_count(onset_seq, threshold)
                epochs = epoch_distance_means(x_seq, y_seq, early_count)
                radius_early = _mean_radius(x_seq[:early_count], y_seq[:early_count], cx, cy)
                radius_late = _mean_radius(x_seq[early_count:], y_seq[early_count:], cx, cy)
                token = _token(threshold)
                row[f"late_minus_early_radius_{label}_t{token}"] = _difference(radius_late, radius_early)
                row[f"late_minus_early_saccade_{label}_t{token}"] = _difference(epochs["late"], epochs["early"])
        rows.append(row)
    return pd.DataFrame(rows)


def number_token(value: float) -> str:
    return _token(value)


def _token(value: float) -> str:
    as_int = int(value)
    if float(as_int) == float(value):
        return str(as_int)
    return str(value).replace(".", "p")
