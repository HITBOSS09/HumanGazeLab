import numpy as np
import pandas as pd

from humangazelab.within_mask import (
    N_BINS,
    assign_equal_mass_bins,
    image_bin_rates,
    pilot_stems,
    reproduce_primary_summary,
    weights_at,
)


def _canvas(mask_mass: dict[tuple[int, int], float]) -> tuple[np.ndarray, np.ndarray]:
    mass = np.zeros((6, 8), dtype=np.float64)
    mask = np.zeros((6, 8), dtype=bool)
    for (row, column), value in mask_mass.items():
        mass[row, column] = value
        mask[row, column] = True
    outside = 1.0 - mass.sum()
    mass[0, 0] += outside
    return mass, mask


def test_four_equal_pixels_share_one_quarter_in_every_bin():
    mass, mask = _canvas({(2, 2): 0.1, (2, 3): 0.1, (2, 4): 0.1, (2, 5): 0.1})
    assignment = assign_equal_mass_bins(mass, mask)
    assert assignment is not None
    assert abs(assignment["conditional_sum"] - 1.0) < 1e-12
    assert abs(assignment["region_mass"] - 0.4) < 1e-12
    assert np.allclose(assignment["predicted"], 0.25)
    for column in (2, 3, 4, 5):
        assert np.allclose(weights_at(assignment, 2, column), 0.25)


def test_a_heavy_pixel_is_split_across_the_quarters_it_covers():
    mass, mask = _canvas({(3, 3): 0.4, (3, 4): 0.01, (4, 3): 0.01, (4, 4): 0.01})
    assignment = assign_equal_mass_bins(mass, mask)
    weights = weights_at(assignment, 3, 3)
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert weights[0] > weights[3]
    assert np.allclose(assignment["predicted"], 0.25)
    assert assignment["n_boundary_pixels"] >= 1


def test_tied_pixels_receive_the_same_weights_regardless_of_position():
    mass, mask = _canvas({(1, 1): 0.30, (1, 2): 0.20, (1, 3): 0.30, (1, 4): 0.20})
    assignment = assign_equal_mass_bins(mass, mask)
    high = weights_at(assignment, 1, 1)
    assert np.allclose(high, weights_at(assignment, 1, 3))
    assert np.allclose(weights_at(assignment, 1, 2), weights_at(assignment, 1, 4))
    # The 0.30 pair holds 0.60 of the conditional mass and covers the first
    # two quarters plus 0.10 of the third.
    assert np.allclose(high, [0.25 / 0.60, 0.25 / 0.60, 0.10 / 0.60, 0.0])
    assert assignment["n_tied_pixels"] == 4
    assert abs(float(assignment["predicted"].sum()) - 1.0) < 1e-12


def test_exact_density_ties_do_not_depend_on_raster_order():
    mass, mask = _canvas({(1, 1): 0.05, (1, 2): 0.05, (2, 1): 0.05, (2, 2): 0.05})
    first = assign_equal_mass_bins(mass, mask)
    assert first["n_tied_pixels"] == 4
    assert np.allclose(weights_at(first, 1, 1), weights_at(first, 2, 2))
    assert np.allclose(weights_at(first, 1, 1), 0.25)


def test_pixel_outside_the_mask_has_no_bin():
    mass, mask = _canvas({(2, 2): 0.05, (2, 3): 0.05, (2, 4): 0.05, (2, 5): 0.05})
    assignment = assign_equal_mass_bins(mass, mask)
    assert weights_at(assignment, 5, 7) is None


def test_too_small_or_empty_masks_are_not_binned():
    mass = np.full((6, 8), 1.0 / 48)
    tiny = np.zeros((6, 8), dtype=bool)
    tiny[1, 1] = True
    assert assign_equal_mass_bins(mass, tiny) is None
    empty = np.zeros((6, 8), dtype=bool)
    assert assign_equal_mass_bins(mass, empty) is None


def test_pilot_is_a_deterministic_stride_and_includes_the_reproduction_image():
    stems = [str(1000 + index) for index in range(40)]
    chosen = pilot_stems(stems, stride=10)
    assert "1001" in chosen
    assert chosen == tuple(sorted(chosen))
    assert pilot_stems(stems, stride=10) == chosen
    assert len(chosen) <= 50


def test_image_rates_weight_images_equally_and_ignore_outside_fixations():
    frame = pd.DataFrame(
        [
            {"image": "a", "label": "face", "in_mask": 1, "obs0": 1.0, "obs1": 0.0, "obs2": 0.0, "obs3": 0.0, "pred0": 0.25, "pred1": 0.25, "pred2": 0.25, "pred3": 0.25, "area_px": 10, "centroid_distance_px": 1, "concentration": 0.4},
            {"image": "a", "label": "face", "in_mask": 0, "obs0": float("nan"), "obs1": float("nan"), "obs2": float("nan"), "obs3": float("nan"), "pred0": 0.25, "pred1": 0.25, "pred2": 0.25, "pred3": 0.25, "area_px": 10, "centroid_distance_px": 1, "concentration": 0.4},
            {"image": "b", "label": "face", "in_mask": 1, "obs0": 0.0, "obs1": 0.0, "obs2": 0.0, "obs3": 1.0, "pred0": 0.25, "pred1": 0.25, "pred2": 0.25, "pred3": 0.25, "area_px": 20, "centroid_distance_px": 2, "concentration": 0.5},
            {"image": "b", "label": "face", "in_mask": 1, "obs0": 0.0, "obs1": 0.0, "obs2": 0.0, "obs3": 1.0, "pred0": 0.25, "pred1": 0.25, "pred2": 0.25, "pred3": 0.25, "area_px": 20, "centroid_distance_px": 2, "concentration": 0.5},
        ]
    )
    rates = image_bin_rates(frame)
    face = rates[rates["label"] == "face"].set_index("image")
    assert face.loc["a", "n"] == 1
    assert face.loc["a", "observed_0"] == 1.0
    assert face.loc["b", "observed_0"] == 0.0
    assert abs(face["highest_gap"].mean() - 0.25) < 1e-12


def test_reproduced_summary_catches_a_changed_count():
    events = pd.DataFrame(
        {
            "label": ["face", "face", "text", "text", "object_union", "object_union"],
            "mask_present": [1, 1, 1, 1, 1, 1],
            "image": ["a", "b", "a", "b", "a", "b"],
            "y": [1, 0, 1, 1, 1, 1],
            "p_model": [0.2, 0.2, 0.5, 0.5, 0.4, 0.6],
        }
    )
    summary = pd.DataFrame(
        {
            "label": ["face", "text", "object_union"],
            "n_fixations": [2, 2, 2],
            "model_difference": [0.3, 0.5, 0.5],
        }
    )
    # face observed 0.5 minus predicted 0.2 is 0.3. text is 1 - 0.5 = 0.5.
    # object union is 1 - 0.5 = 0.5.
    checked = reproduce_primary_summary(events, summary)
    assert checked["n_matches"].all()
    assert checked["difference_gap"].max() < 1e-12
    summary.loc[summary["label"] == "face", "n_fixations"] = 9
    assert not reproduce_primary_summary(events, summary)["n_matches"].all()
    assert N_BINS == 4
