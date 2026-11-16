"""Canvas density maps and the uniform-on-canvas reference distribution."""

from __future__ import annotations

import math

import numpy as np

from humangazelab.adapter import Scanpath


def uniform_radius_sample(width: float, height: float, n: int, seed: int) -> np.ndarray:
    """Distances from the canvas center for continuous uniform points on [0, width] x [0, height]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, width, size=n)
    y = rng.uniform(0.0, height, size=n)
    return np.hypot(x - width / 2.0, y - height / 2.0)


def uniform_mean_radius(width: float, height: float, n: int, seed: int) -> dict[str, float]:
    """Monte Carlo mean distance from the canvas center under continuous uniform draws."""
    distance = uniform_radius_sample(width, height, n, seed)
    return {
        "n": float(n),
        "mean": float(distance.mean()),
        "se": float(distance.std(ddof=1) / math.sqrt(n)),
    }


def grid_mean_radius(width: float, height: float, step: float = 2.0) -> float:
    """Mean distance from center on a regular grid of cell centers. Deterministic cross-check."""
    xs = np.arange(step / 2.0, width, step)
    ys = np.arange(step / 2.0, height, step)
    cx = width / 2.0
    cy = height / 2.0
    total = 0.0
    count = 0
    for start in range(0, len(ys), 40):
        y = ys[start : start + 40]
        total += float(np.hypot(xs[None, :] - cx, y[:, None] - cy).sum())
        count += int(y.size * xs.size)
    return total / count


def disk_fraction(width: float, height: float, radius: float) -> float:
    """Area of a centered disk divided by canvas area.

    Caller must keep ``radius`` within the canvas so the disk is not clipped.
    """
    half_min = min(width, height) / 2.0
    if radius > half_min:
        raise ValueError("disk is clipped by the canvas; fraction is not pi r^2 / area")
    return float(math.pi * radius * radius / (width * height))


def subject_mean_density(
    scanpaths: list[Scanpath],
    width: float,
    height: float,
    bins_x: int,
    bins_y: int,
    selector,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Mean and SD across subjects of within-subject fixation histograms.

    ``selector(scanpath)`` returns the ``(x, y)`` arrays to count for that scanpath.
    Each subject histogram is divided by its own total, then subjects are averaged
    with equal weight. The SD is the sample SD across those subject maps (ddof=1).
    """
    edges_x = np.linspace(0.0, width, bins_x + 1)
    edges_y = np.linspace(0.0, height, bins_y + 1)
    by_subject: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for scanpath in scanpaths:
        x, y = selector(scanpath)
        if len(x) == 0:
            continue
        by_subject.setdefault(scanpath.subject_id, []).append((x, y))
    maps = []
    subjects = []
    for subject in sorted(by_subject):
        xs = np.concatenate([item[0] for item in by_subject[subject]])
        ys = np.concatenate([item[1] for item in by_subject[subject]])
        hist, _, _ = np.histogram2d(xs, ys, bins=[edges_x, edges_y])
        total = hist.sum()
        if total == 0:
            continue
        maps.append(hist / total)
        subjects.append(subject)
    if not maps:
        raise ValueError("density selector retained no fixations")
    stack = np.stack(maps, axis=0)
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1) if len(maps) > 1 else np.full(mean.shape, np.nan)
    return edges_x, edges_y, mean, sd, subjects


def post_initial(scanpath: Scanpath) -> tuple[np.ndarray, np.ndarray]:
    return scanpath.x[1:], scanpath.y[1:]


def ordinal_epoch(scanpath: Scanpath, early_k: int, which: str) -> tuple[np.ndarray, np.ndarray]:
    x = scanpath.x[1:]
    y = scanpath.y[1:]
    if which == "early":
        return x[:early_k], y[:early_k]
    if which == "late":
        return x[early_k:], y[early_k:]
    raise ValueError(which)


def density_table(edges_x, edges_y, mean, sd) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for i in range(len(edges_x) - 1):
        for j in range(len(edges_y) - 1):
            rows.append(
                {
                    "x_left": edges_x[i],
                    "x_right": edges_x[i + 1],
                    "y_bottom": edges_y[j],
                    "y_top": edges_y[j + 1],
                    "subject_mean_share": mean[i, j],
                    "subject_sd_share": sd[i, j],
                }
            )
    return pd.DataFrame(rows)


def pooled_marginals(scanpaths: list[Scanpath], width: float, height: float) -> "pd.DataFrame":
    """Fixation-weighted and saccade-weighted quantiles on post-initial points.

    These rows weight every fixation or saccade equally. They are a different
    weighting from the subject-mean table.
    """
    import pandas as pd

    cx = width / 2.0
    cy = height / 2.0
    xs = []
    ys = []
    durations = []
    saccades = []
    for scanpath in scanpaths:
        x = scanpath.x[1:]
        y = scanpath.y[1:]
        d = scanpath.duration[1:]
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
        durations.append(d)
        if len(x) >= 2:
            saccades.append(np.hypot(np.diff(x), np.diff(y)))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    duration = np.concatenate(durations)
    radius = np.hypot(x - cx, y - cy)
    saccade = np.concatenate(saccades) if saccades else np.asarray([])
    rows = [
        _quantile_row("x_px", "each post-initial fixation", x),
        _quantile_row("y_px", "each post-initial fixation", y),
        _quantile_row("radius_px", "each post-initial fixation", radius),
        _quantile_row("duration_released", "each post-initial fixation", duration),
        _quantile_row("saccade_px", "each post-initial successive pair", saccade),
    ]
    return pd.DataFrame(rows)


def _quantile_row(variable: str, weight: str, values: np.ndarray) -> dict:
    if len(values) == 0:
        stats = {name: math.nan for name in ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean")}
    else:
        q = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        stats = {
            "min": float(values.min()),
            "p10": float(q[0]),
            "p25": float(q[1]),
            "p50": float(q[2]),
            "p75": float(q[3]),
            "p90": float(q[4]),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return {"variable": variable, "weight": weight, "n": int(len(values)), **stats}
