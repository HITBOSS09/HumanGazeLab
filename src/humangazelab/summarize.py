"""Equal-group means and percentile bootstrap intervals.

The resampling unit is the group (subject or image), after each group has been
reduced to its unweighted mean of scanpath values.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd


def percentile_bootstrap_mean(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    probs: tuple[float, float] = (0.025, 0.975),
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n_groups": 0,
            "estimate": math.nan,
            "sd": math.nan,
            "sem": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }
    estimate = float(values.mean())
    if len(values) == 1:
        sd = math.nan
        sem = math.nan
    else:
        sd = float(values.std(ddof=1))
        sem = sd / math.sqrt(len(values))
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    ci_low, ci_high = (float(v) for v in np.quantile(draws, probs))
    return {
        "n_groups": int(len(values)),
        "estimate": estimate,
        "sd": sd,
        "sem": float(sem) if sem == sem else math.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def metric_seed(base_seed: int, metric: str, subset: str, estimand: str) -> int:
    token = f"{metric}|{subset}|{estimand}".encode()
    mixed = (base_seed + (zlib.crc32(token) % 1_000_003)) % (2**31 - 1)
    return int(mixed)


def summarize_column(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str,
    mask: pd.Series,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    part = frame.loc[mask].dropna(subset=[value_col])
    means = part.groupby(group_col, sort=True)[value_col].mean()
    stats = percentile_bootstrap_mean(means.to_numpy(), n_boot, seed)
    stats["n_scanpaths"] = int(len(part))
    return stats
