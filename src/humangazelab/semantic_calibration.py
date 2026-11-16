"""Region-event summaries for a frozen spatial density.

The model stays a distribution on the 1167 by 875 grid. Each fine pixel is
assigned to the original 800 by 600 pixel that contains its center, and the
probability mass is added there. The projected map is renormalized. It is a
discrete distribution on the annotation canvas, not a new gaze model.

A semantic label is the union of object masks whose attribute value is positive.
Labels are separate binary events. They are not required to sum to one.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from humangazelab.osie_deepgaze import scaled_pixel

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
RESIZED_WIDTH = 1167
RESIZED_HEIGHT = 875
LOG_LOSS_CLIP = 1e-6


def inverse_index(resized_height: int = RESIZED_HEIGHT, resized_width: int = RESIZED_WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Original pixel that receives each fine-grid pixel, as (row_index, col_index)."""
    origin_x = (np.arange(resized_width) + 0.5) * CANVAS_WIDTH / resized_width - 0.5
    origin_y = (np.arange(resized_height) + 0.5) * CANVAS_HEIGHT / resized_height - 0.5
    column = np.clip(np.rint(origin_x), 0, CANVAS_WIDTH - 1).astype(np.int32)
    row = np.clip(np.rint(origin_y), 0, CANVAS_HEIGHT - 1).astype(np.int32)
    return row, column


def project_log_density(log_density: np.ndarray, row_index: np.ndarray, column_index: np.ndarray) -> np.ndarray:
    """Move a log-density on the fine grid onto the original canvas.

    log_density is (fine_height, fine_width) or (batch, fine_height, fine_width).
    The returned mass sums to one along the canvas axes.
    """
    single = log_density.ndim == 2
    stacked = log_density[None] if single else log_density
    shifted = stacked - stacked.max(axis=(1, 2), keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=(1, 2), keepdims=True)
    flat = probability.reshape(probability.shape[0], -1)
    if row_index.ndim == 1:
        rows = np.repeat(row_index, column_index.shape[0])
        columns = np.tile(column_index, row_index.shape[0])
    else:
        rows = row_index.reshape(-1)
        columns = column_index.reshape(-1)
    bins = (rows * CANVAS_WIDTH + columns).astype(np.int32)
    if flat.shape[1] != bins.shape[0]:
        raise ValueError("log-density shape does not match the fine grid")
    mass = np.zeros((flat.shape[0], CANVAS_HEIGHT * CANVAS_WIDTH), dtype=np.float64)
    for index in range(flat.shape[0]):
        mass[index] = np.bincount(bins, weights=flat[index], minlength=mass.shape[1])
    totals = mass.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("projected map has no probability mass")
    mass /= totals
    canvas = mass.reshape((-1, CANVAS_HEIGHT, CANVAS_WIDTH))
    return canvas[0] if single else canvas


def region_mass(mass: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum probability mass on a boolean mask. mass is (H, W) or (batch, H, W)."""
    if mass.ndim == 2:
        value = float((mass * mask).sum())
        if not 0.0 <= value <= 1.0 + 1e-8:
            raise ValueError(f"region mass {value} is outside [0, 1]")
        return np.asarray(min(1.0, max(0.0, value)))
    value = (mass * mask).sum(axis=(1, 2))
    if np.any(value < -1e-8) or np.any(value > 1.0 + 1e-8):
        raise ValueError("a region mass is outside [0, 1]")
    return np.clip(value, 0.0, 1.0)


def fixation_pixel(x: float, y: float) -> tuple[int, int]:
    """Nearest original pixel. This is the annotation coordinate, not the fine grid."""
    column = int(np.clip(np.rint(x), 0, CANVAS_WIDTH - 1))
    row = int(np.clip(np.rint(y), 0, CANVAS_HEIGHT - 1))
    return row, column


def scoring_pixel_agrees(x: float, y: float, row_index: np.ndarray, column_index: np.ndarray) -> bool:
    """True when the frozen fine-grid lookup falls in the same original pixel as rint."""
    row, column = fixation_pixel(x, y)
    fine_x = scaled_pixel(x, CANVAS_WIDTH, RESIZED_WIDTH)
    fine_y = scaled_pixel(y, CANVAS_HEIGHT, RESIZED_HEIGHT)
    return int(column_index[fine_x]) == column and int(row_index[fine_y]) == row


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return binary_dilation(mask, iterations=radius)


def log_loss(probability: np.ndarray, outcome: np.ndarray, clip: float = LOG_LOSS_CLIP) -> float:
    probability = np.clip(np.asarray(probability, dtype=float), clip, 1.0 - clip)
    outcome = np.asarray(outcome, dtype=float)
    if len(outcome) == 0:
        return math.nan
    loss = -(outcome * np.log(probability) + (1.0 - outcome) * np.log(1.0 - probability))
    return float(np.mean(loss))


def brier_score(probability: np.ndarray, outcome: np.ndarray) -> float:
    if len(outcome) == 0:
        return math.nan
    return float(np.mean((np.asarray(probability, dtype=float) - np.asarray(outcome, dtype=float)) ** 2))


def observed_expected(probability: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(outcome) == 0:
        return {"mean_predicted": math.nan, "observed_frequency": math.nan, "difference": math.nan, "ratio": math.nan}
    mean_p = float(np.mean(probability))
    mean_y = float(np.mean(outcome))
    ratio = mean_y / mean_p if mean_p > 1e-8 else math.nan
    return {
        "mean_predicted": mean_p,
        "observed_frequency": mean_y,
        "difference": mean_y - mean_p,
        "ratio": ratio,
    }


def reliability_rows(probability: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> list[dict]:
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        if index == n_bins - 1:
            chosen = (probability >= low) & (probability <= high)
        else:
            chosen = (probability >= low) & (probability < high)
        count = int(chosen.sum())
        rows.append(
            {
                "bin": index,
                "low": float(low),
                "high": float(high),
                "n": count,
                "mean_predicted": float(probability[chosen].mean()) if count else math.nan,
                "observed_frequency": float(outcome[chosen].mean()) if count else math.nan,
            }
        )
    return rows


def shift_mask(mask: np.ndarray, delta_row: int, delta_column: int) -> np.ndarray:
    """Translate a mask and drop the part that leaves the canvas."""
    moved = np.zeros_like(mask)
    height, width = mask.shape
    source_row = slice(max(0, -delta_row), min(height, height - delta_row))
    source_column = slice(max(0, -delta_column), min(width, width - delta_column))
    dest_row = slice(max(0, delta_row), min(height, height + delta_row))
    dest_column = slice(max(0, delta_column), min(width, width + delta_column))
    moved[dest_row, dest_column] = mask[source_row, source_column]
    return moved


def matched_shifts(
    mask: np.ndarray,
    rng: np.random.Generator,
    n_shifts: int,
    radius: int,
    min_fraction: float,
    max_candidates: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Translations that keep most of the mask inside the canvas.

    The zero shift is excluded. If fewer than n_shifts translations qualify,
    the match is reported as impossible and the qualifying shifts are returned.
    """
    area = int(mask.sum())
    if area == 0:
        return [], False
    found: list[tuple[int, int]] = []
    seen = {(0, 0)}
    for _ in range(max_candidates):
        delta_row = int(rng.integers(-radius, radius + 1))
        delta_column = int(rng.integers(-radius, radius + 1))
        if (delta_row, delta_column) in seen:
            continue
        seen.add((delta_row, delta_column))
        moved = shift_mask(mask, delta_row, delta_column)
        if moved.sum() >= min_fraction * area:
            found.append((delta_row, delta_column))
        if len(found) == n_shifts:
            return found, True
    return found, False


def image_folds(stems: list[str], n_folds: int, seed: int) -> dict[str, int]:
    """Assign whole images to folds. A stem never shares its fold with a copy of itself."""
    if n_folds < 2:
        raise ValueError("cross-fitting needs at least two folds")
    order = np.array(sorted(set(stems)))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return {str(stem): int(index % n_folds) for index, stem in enumerate(order)}


def crossfit_linear_residual(frame: pd.DataFrame, target: str, covariates: list[str], n_folds: int, seed: int) -> pd.DataFrame:
    """Out-of-fold residual of an image-level gap after a linear covariate fit.

    Training rows are other images. This is a diagnostic residual, not a gaze model.
    """
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[target, *covariates, "image"]).copy()
    folds = image_folds(work["image"].astype(str).tolist(), n_folds, seed)
    residuals = []
    for image, fold in folds.items():
        row = work.loc[work["image"].astype(str) == image]
        if row.empty:
            continue
        train = work.loc[work["image"].astype(str).map(folds) != fold]
        if len(train) < len(covariates) + 2:
            residual = math.nan
        else:
            design = np.column_stack([np.ones(len(train)), train[covariates].to_numpy(dtype=float)])
            coef, *_ = np.linalg.lstsq(design, train[target].to_numpy(dtype=float), rcond=None)
            point = np.array([1.0, *[float(row.iloc[0][name]) for name in covariates]])
            residual = float(row.iloc[0][target] - point @ coef)
        residuals.append({"image": image, "fold": fold, "residual": residual, "target": target})
    return pd.DataFrame(residuals)


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def _logit(probability: np.ndarray, clip: float = LOG_LOSS_CLIP) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), clip, 1.0 - clip)
    return np.log(probability) - np.log(1.0 - probability)


def _fit_logistic(design: np.ndarray, outcome: np.ndarray, l2: float, steps: int = 25) -> np.ndarray:
    """Newton steps for an L2-penalized logistic. The intercept is not penalized."""
    beta = np.zeros(design.shape[1], dtype=float)
    eye = np.eye(design.shape[1])
    for _ in range(steps):
        probability = _sigmoid(design @ beta)
        weight = probability * (1.0 - probability)
        gradient = design.T @ (probability - outcome)
        gradient[1:] += l2 * beta[1:]
        hessian = design.T @ (design * weight[:, None])
        hessian[1:, 1:] += l2 * eye[1:, 1:]
        hessian = hessian + 1e-6 * eye
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return beta


def stage_d_probabilities(
    frame: pd.DataFrame,
    n_folds: int,
    seed: int,
    l2: float = 1.0,
    steps: int = 25,
    covariate_columns: tuple[str, ...] = (
        "area_fraction",
        "centroid_distance_px",
        "luminance_contrast",
        "target_index",
        "prev_distance_px",
    ),
) -> pd.DataFrame:
    """Out-of-fold event probabilities. Training rows come from other images.

    This compares an unmodified region probability, an intercept-only base rate,
    a one-dimensional Platt scaling of that probability, and the same scaling
    plus the predeclared non-semantic covariates. It does not change the spatial density.
    """
    required = ["image", "y", "p_model", *covariate_columns]
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    work["image"] = work["image"].astype(str)
    folds = image_folds(work["image"].tolist(), n_folds, seed)
    work["_fold"] = work["image"].map(folds)
    work["_logit"] = _logit(work["p_model"].to_numpy())
    columns = {
        "p_unmodified": work["p_model"].to_numpy(dtype=float),
        "p_intercept": np.full(len(work), np.nan),
        "p_platt": np.full(len(work), np.nan),
        "p_covariates": np.full(len(work), np.nan),
    }
    feature_names = ["_logit", *covariate_columns]
    for fold in sorted(set(folds.values())):
        train = work["_fold"] != fold
        test = work["_fold"] == fold
        if int(train.sum()) < len(feature_names) + 2 or int(test.sum()) == 0:
            continue
        outcome = work.loc[train, "y"].to_numpy(dtype=float)
        columns["p_intercept"][test.to_numpy()] = float(outcome.mean())
        platt_design = np.column_stack([np.ones(int(train.sum())), work.loc[train, "_logit"].to_numpy()])
        platt = _fit_logistic(platt_design, outcome, l2, steps)
        test_platt = np.column_stack([np.ones(int(test.sum())), work.loc[test, "_logit"].to_numpy()])
        columns["p_platt"][test.to_numpy()] = _sigmoid(test_platt @ platt)
        train_features = work.loc[train, feature_names].to_numpy(dtype=float)
        center = train_features.mean(axis=0)
        scale = train_features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        train_design = np.column_stack([np.ones(len(train_features)), (train_features - center) / scale])
        fitted = _fit_logistic(train_design, outcome, l2, steps)
        test_features = work.loc[test, feature_names].to_numpy(dtype=float)
        test_design = np.column_stack([np.ones(len(test_features)), (test_features - center) / scale])
        columns["p_covariates"][test.to_numpy()] = _sigmoid(test_design @ fitted)
    result = work[["image", "y", "p_model"]].copy()
    for name, values in columns.items():
        result[name] = values
    return result.reset_index(drop=True)


def region_covariates(mask: np.ndarray, image: np.ndarray) -> dict[str, float]:
    """Area, center distance, eccentricity, luminance contrast, and gradient energy.

    image is uint8 or float RGB in original resolution. Empty masks return NaN
    covariates and area 0. Eccentricity is the ellipse eccentricity of the mask
    pixels. Orientation energy is the mean absolute difference of squared
    horizontal and vertical gradients inside the mask.
    """
    area = int(mask.sum())
    if area == 0:
        return {
            "area_px": 0,
            "area_fraction": 0.0,
            "centroid_distance_px": math.nan,
            "eccentricity": math.nan,
            "luminance_contrast": math.nan,
            "gradient_energy": math.nan,
            "orientation_energy": math.nan,
        }
    rows, columns = np.nonzero(mask)
    center_row = float(rows.mean())
    center_column = float(columns.mean())
    distance = math.hypot(center_column - (CANVAS_WIDTH - 1) / 2, center_row - (CANVAS_HEIGHT - 1) / 2)
    if area < 2:
        eccentricity = 0.0
    else:
        centered_row = rows - center_row
        centered_column = columns - center_column
        covariance = np.cov(centered_column, centered_row, ddof=0)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
        largest = float(np.max(eigenvalues))
        smallest = float(np.min(eigenvalues))
        eccentricity = math.sqrt(max(0.0, 1.0 - smallest / largest)) if largest > 0 else 0.0
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    gradient_row, gradient_column = np.gradient(luminance)
    inside = mask
    return {
        "area_px": area,
        "area_fraction": area / float(CANVAS_HEIGHT * CANVAS_WIDTH),
        "centroid_distance_px": distance,
        "eccentricity": eccentricity,
        "luminance_contrast": float(luminance[inside].std()),
        "gradient_energy": float((gradient_column[inside] ** 2 + gradient_row[inside] ** 2).mean()),
        "orientation_energy": float(np.abs(gradient_column[inside] ** 2 - gradient_row[inside] ** 2).mean()),
    }
