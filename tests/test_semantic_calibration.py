"""Probability mass, region events, and image-disjoint folds. No network."""

import math

import numpy as np
import pandas as pd

from humangazelab.semantic_calibration import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    RESIZED_HEIGHT,
    RESIZED_WIDTH,
    brier_score,
    crossfit_linear_residual,
    dilate,
    image_folds,
    inverse_index,
    log_loss,
    matched_shifts,
    observed_expected,
    project_log_density,
    region_mass,
    scoring_pixel_agrees,
    shift_mask,
    stage_d_probabilities,
)
from humangazelab.osie_deepgaze import scaled_pixel


def test_projected_mass_sums_to_one_and_keeps_a_single_fine_pixel():
    row_index, column_index = inverse_index()
    log_density = np.full((RESIZED_HEIGHT, RESIZED_WIDTH), -100.0)
    log_density[10, 20] = 0.0
    mass = project_log_density(log_density, row_index, column_index)
    assert math.isclose(float(mass.sum()), 1.0, abs_tol=1e-10)
    assert mass[row_index[10], column_index[20]] == 1.0


def test_region_mass_is_the_share_of_the_mask():
    mass = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH))
    mass[0, 0] = 0.25
    mass[0, 1] = 0.75
    mask = np.zeros_like(mass, dtype=bool)
    mask[0, 0] = True
    assert float(region_mass(mass, mask)) == 0.25
    empty = np.zeros_like(mask)
    assert float(region_mass(mass, empty)) == 0.0


def test_integer_fixation_pixel_matches_the_frozen_lookup():
    row_index, column_index = inverse_index()
    for x, y in ((0, 0), (400, 300), (799, 599), (338, 494)):
        assert scoring_pixel_agrees(x, y, row_index, column_index)
        assert scaled_pixel(x, CANVAS_WIDTH, RESIZED_WIDTH) >= 0


def test_log_loss_clips_and_brier_matches_a_known_value():
    probability = np.array([0.0, 1.0, 0.5])
    outcome = np.array([0.0, 1.0, 1.0])
    assert brier_score(probability, outcome) == (0.0 + 0.0 + 0.25) / 3
    assert math.isfinite(log_loss(probability, outcome))
    assert math.isnan(observed_expected(np.array([0.0, 0.0]), np.array([1.0, 0.0]))["ratio"])


def test_shift_preserves_area_when_the_mask_stays_inside():
    mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
    mask[20:30, 20:40] = True
    moved = shift_mask(mask, 5, -3)
    assert int(moved.sum()) == int(mask.sum())
    assert not moved[20, 20]
    rng = np.random.default_rng(20261115)
    first, ok_first = matched_shifts(mask, rng, n_shifts=4, radius=30, min_fraction=0.8, max_candidates=40)
    rng = np.random.default_rng(20261115)
    second, ok_second = matched_shifts(mask, np.random.default_rng(20261115), n_shifts=4, radius=30, min_fraction=0.8, max_candidates=40)
    assert ok_first and ok_second
    assert first == second


def test_image_folds_do_not_put_an_image_in_its_own_training_set():
    folds = image_folds(["b", "a", "a", "c", "d"], n_folds=2, seed=7)
    assert set(folds) == {"a", "b", "c", "d"}
    for stem, fold in folds.items():
        training = [other for other, other_fold in folds.items() if other_fold != fold]
        assert stem not in training


def test_crossfit_residual_uses_other_images_only():
    frame = pd.DataFrame(
        {
            "image": list("abcdefgh"),
            "gap": [1, 1, 1, 1, 3, 3, 3, 3],
            "area": [1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    residual = crossfit_linear_residual(frame, "gap", ["area"], n_folds=2, seed=1)
    assert set(residual["image"]) == set("abcdefgh")
    assert residual["residual"].notna().all()


def test_stage_d_held_out_image_does_not_change_its_own_prediction():
    rows = []
    for image_index, image in enumerate("abcdef"):
        for fixation in range(12):
            area = 0.2 if image_index < 3 else 0.8
            rows.append(
                {
                    "image": image,
                    "y": int(area > 0.5),
                    "p_model": 0.4,
                    "area_fraction": area,
                    "centroid_distance_px": 10.0 + fixation,
                    "luminance_contrast": 0.1,
                    "target_index": 4 + fixation,
                    "prev_distance_px": 50.0,
                }
            )
    frame = pd.DataFrame(rows)
    first = stage_d_probabilities(frame, n_folds=3, seed=5, l2=1.0)
    flipped = frame.copy()
    flipped.loc[flipped["image"] == "a", "y"] = 1 - flipped.loc[flipped["image"] == "a", "y"]
    second = stage_d_probabilities(flipped, n_folds=3, seed=5, l2=1.0)
    left = first[first["image"] == "a"]["p_covariates"].to_numpy()
    right = second[second["image"] == "a"]["p_covariates"].to_numpy()
    assert np.allclose(left, right)
    assert np.all((first["p_unmodified"] > 0) & (first["p_unmodified"] < 1))
    assert first["p_covariates"].notna().all()


def test_dilation_grows_a_mask_without_changing_the_source():
    mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
    mask[100, 100] = True
    grown = dilate(mask, 1)
    assert grown.sum() > 1
    assert mask.sum() == 1
