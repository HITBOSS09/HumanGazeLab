"""Equal-mass bins inside a semantic mask.

The conditional density is the frozen map restricted to the mask and divided
by the mask's original probability mass. Four bins each receive one quarter
of that conditional mass. Pixels that share an exact conditional density are
exchangeable: the model does not rank them. Their mass is split across bin
boundaries, and every pixel with that density receives the same fractional
weights. A fixation on such a pixel counts fractionally in each bin it
touches. Raster order is not used.

The nearest upsampling in the frozen finalizer makes large exact plateaus.
Slicing a plateau by row and column would assign different bins to pixels
the density does not distinguish.

This is a check of where mass sits inside the mask. It is not the check of
how much mass the mask received in total.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

N_BINS = 4
MIN_PIXELS = 4
MIN_REGION_MASS = 1e-8


def assign_equal_mass_bins(mass: np.ndarray, mask: np.ndarray, n_bins: int = N_BINS) -> dict | None:
    """Bin mask pixels by conditional density.

    mass is a probability map on the original canvas and must sum to one.
    Returns None when the mask has fewer than n_bins pixels or almost no mass.
    """
    if mass.shape != mask.shape:
        raise ValueError("mass and mask shapes differ")
    total_mass = float(mass.sum())
    if abs(total_mass - 1.0) > 1e-5:
        raise ValueError(f"probability map sums to {total_mass}, not 1")
    rows, columns = np.nonzero(mask)
    if len(rows) < n_bins:
        return None
    values = mass[rows, columns].astype(np.float64)
    region_mass = float(values.sum())
    if region_mass <= MIN_REGION_MASS:
        return None
    conditional = values / region_mass
    conditional_sum = float(conditional.sum())
    if abs(conditional_sum - 1.0) > 1e-8:
        raise ValueError(f"conditional mass sums to {conditional_sum}")
    unique_values, inverse, counts = np.unique(conditional, return_inverse=True, return_counts=True)
    value_mass = unique_values * counts.astype(np.float64)
    order = np.argsort(-unique_values, kind="mergesort")
    ordered_mass = value_mass[order]
    cumulative = np.cumsum(ordered_mass)
    start = cumulative - ordered_mass
    shares = np.zeros((len(order), n_bins), dtype=np.float64)
    for bin_index in range(n_bins):
        low = bin_index / n_bins
        high = (bin_index + 1) / n_bins
        shares[:, bin_index] = np.maximum(0.0, np.minimum(cumulative, high) - np.maximum(start, low))
    predicted = shares.sum(axis=0)
    if abs(float(predicted.sum()) - 1.0) > 1e-8:
        raise ValueError("bin masses do not sum to one")
    positive = ordered_mass > 0
    ordered_weights = np.zeros_like(shares)
    ordered_weights[positive] = shares[positive] / ordered_mass[positive, None]
    # Exact zeros sit past the last unit of mass. They are the lowest bin.
    ordered_weights[~positive, -1] = 1.0
    value_weights = np.empty_like(ordered_weights)
    value_weights[order] = ordered_weights
    weights = value_weights[inverse]
    boundary = (shares > 1e-12).sum(axis=1) > 1
    boundary_pixels = int(counts[order][boundary].sum()) if boundary.any() else 0
    tied_mask = counts > 1
    tied = int(counts[tied_mask].sum()) if tied_mask.any() else 0
    tied_mass = float(value_mass[tied_mask].sum()) if tied_mask.any() else 0.0
    top = max(1, int(math.ceil(0.1 * len(conditional))))
    concentration = float(np.sort(conditional)[-top:].sum())
    index_map = np.full(mask.shape, -1, dtype=np.int32)
    index_map[rows, columns] = np.arange(len(rows), dtype=np.int32)
    return {
        "index_map": index_map,
        "weights": weights,
        "predicted": predicted,
        "region_mass": region_mass,
        "n_pixels": int(len(rows)),
        "n_boundary_pixels": boundary_pixels,
        "n_tied_pixels": tied,
        "tied_mass": tied_mass,
        "concentration": concentration,
        "conditional_sum": conditional_sum,
    }


def weights_at(assignment: dict, row: int, column: int) -> np.ndarray | None:
    """Fractional bin weights of one pixel, or None when it is outside the mask."""
    index = int(assignment["index_map"][row, column])
    if index < 0:
        return None
    return assignment["weights"][index]


def eroded_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    eroded = binary_erosion(mask, iterations=radius)
    return eroded if int(eroded.sum()) >= N_BINS else mask


def image_bin_rates(frame: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """One row per image and label, using only fixations that landed inside the mask."""
    inside = frame[(frame["in_mask"] == 1) & frame["obs0"].notna()]
    rows = []
    if inside.empty:
        return pd.DataFrame(rows)
    for (image, label), part in inside.groupby(["image", "label"], sort=True):
        row = {
            "image": str(image),
            "label": str(label),
            "n": int(len(part)),
            "area_px": float(part["area_px"].iloc[0]),
            "centroid_distance_px": float(part["centroid_distance_px"].iloc[0]),
            "concentration": float(part["concentration"].mean()),
        }
        for bin_index in range(n_bins):
            row[f"observed_{bin_index}"] = float(part[f"obs{bin_index}"].mean())
            row[f"predicted_{bin_index}"] = float(part[f"pred{bin_index}"].mean())
        row["highest_gap"] = row["observed_0"] - row["predicted_0"]
        row["highest_vs_quarter"] = row["observed_0"] - 0.25
        rows.append(row)
    return pd.DataFrame(rows)


def pilot_stems(stems: list[str], stride: int = 20) -> tuple[str, ...]:
    """Every stride-th sorted name, plus image 1001. Fixed before any bin count."""
    ordered = sorted({str(stem) for stem in stems})
    chosen = set(ordered[::stride])
    chosen.add("1001")
    return tuple(sorted(chosen))


def reproduce_primary_summary(events: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Recompute the stored face, text, and object-union rows from the event table."""
    present = events[events["mask_present"] == 1]
    rows = []
    for label in ("object_union", "face", "text"):
        part = present[present["label"] == label]
        stored = summary[summary["label"] == label].iloc[0]
        difference = float(part["y"].mean() - part["p_model"].mean())
        rows.append(
            {
                "label": label,
                "n_fixations": int(len(part)),
                "n_images": int(part["image"].nunique()),
                "difference": difference,
                "stored_n_fixations": int(stored["n_fixations"]),
                "stored_difference": float(stored["model_difference"]),
                "n_matches": int(len(part)) == int(stored["n_fixations"]),
                "difference_gap": abs(difference - float(stored["model_difference"])),
            }
        )
    return pd.DataFrame(rows)
