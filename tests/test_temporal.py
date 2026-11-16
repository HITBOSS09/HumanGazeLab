"""Leakage, normalization, indexing, and control checks for temporal_v1."""

import math

import numpy as np
import pandas as pd
import pytest

from humangazelab.adapter import Scanpath
from humangazelab.summarize import percentile_bootstrap_mean
from humangazelab.temporal import (
    Grid,
    _loo_maps,
    accumulate_counts,
    alternate_history_index,
    bin_index,
    blur_counts,
    density_from_counts,
    derangement,
    evaluate_records,
    histogram_counts,
    image_role,
    impulse_library,
    information_gain,
    normalize_and_floor,
    pack_scanpaths,
)


def _scan(name, subject, split, points):
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return Scanpath(0, name, subject, split, "freeview", x, y, np.ones(len(points)), False, 1)


def _records(scans, grid):
    kwargs = {"train_name": "train", "modulus": 5, "bandwidth_bucket": 0, "mixture_bucket": 1}
    return pack_scanpaths(scans, grid, kwargs)


def test_bins_cover_the_canvas_without_wrapping():
    grid = Grid(1680, 1050, 42, 26)
    assert bin_index(np.array([0.0]), grid.width, grid.nx)[0] == 0
    assert bin_index(np.array([1679.9]), grid.width, grid.nx)[0] == 41
    assert bin_index(np.array([0.0]), grid.height, grid.ny)[0] == 0
    assert bin_index(np.array([1049.9]), grid.height, grid.ny)[0] == 25
    assert bin_index(np.array([1680.0]), grid.width, grid.nx)[0] == 41


def test_density_is_positive_and_normalized_and_uniform_ig_is_zero():
    counts = np.zeros((5, 4))
    counts[1, 1] = 3
    density = density_from_counts(counts, sigma_bins=0.8, alpha=0.05)
    assert np.isclose(density.sum(), 1.0)
    assert density.min() > 0
    uniform = normalize_and_floor(np.zeros((5, 4)), 0.05)
    ig = information_gain(np.full(4, 1.0 / 20.0), 20)
    assert np.allclose(ig, 0.0)
    assert np.allclose(uniform, 1.0 / 20.0)


def test_impulse_library_matches_histogram_blur():
    grid = Grid(40, 30, 4, 3)
    library = impulse_library(grid.ny, grid.nx, 0.7)
    iy = np.array([0, 0, 2])
    ix = np.array([1, 1, 3])
    counts = histogram_counts(iy, ix, grid.ny, grid.nx)
    from_hist = blur_counts(counts, 0.7)
    from_lib = library[0, 1] + library[0, 1] + library[2, 3]
    assert np.allclose(from_hist, from_lib)


def test_held_out_observer_does_not_change_own_reference():
    grid = Grid(100, 100, 10, 10)
    scans = [
        _scan("img", 1, "train", [(5, 5), (6, 6), (7, 7)]),
        _scan("img", 2, "train", [(80, 80), (82, 80), (84, 80)]),
        _scan("img", 3, "train", [(80, 20), (82, 20), (84, 20)]),
    ]
    records = _records(scans, grid)
    before = _loo_maps(records, grid, sigma_bins=0.4, alpha=0.05, index_min=1)
    records[0].ix[:] = 9
    records[0].iy[:] = 1
    after = _loo_maps(records, grid, sigma_bins=0.4, alpha=0.05, index_min=1)
    assert np.allclose(before[1], after[1])
    assert not np.allclose(before[2], after[2])


def test_validation_fixations_are_absent_from_the_train_count():
    grid = Grid(100, 100, 10, 10)
    scans = [
        _scan("train-img", 1, "train", [(10, 10), (12, 10)]),
        _scan("valid-img", 1, "valid", [(90, 90), (92, 90)]),
    ]
    records = _records(scans, grid)
    counts = accumulate_counts(records, lambda r: r.split == "train", 1, None, grid)
    assert counts.sum() == 1
    assert records[1].role == "eval"
    assert records[0].role in {"fit", "select_bandwidth", "select_mixture"}


def test_roles_partition_train_and_leave_validation_alone():
    roles = {image_role(f"im{i}", "train", "train", 5, 0, 1) for i in range(50)}
    assert roles <= {"fit", "select_bandwidth", "select_mixture"}
    assert image_role("anything", "valid", "train", 5, 0, 1) == "eval"


def test_image_shuffle_is_a_derangement_and_is_stable():
    names = [f"im{i}" for i in range(12)]
    first = derangement(names, 20261114)
    second = derangement(names, 20261114)
    assert first == second
    assert set(first) == set(names)
    assert all(left != right for left, right in first.items())
    assert set(first.values()) == set(names)


def test_alternate_history_is_not_the_predecessor():
    for length in (3, 5, 8):
        for target in range(1, length):
            previous = alternate_history_index(target, length)
            assert previous != target
            assert previous != target - 1
            assert 0 <= previous < length


def test_osie_loader_keeps_order_and_holds_out_a_hash_bucket():
    from humangazelab.osie import osie_split, scanpaths_from_fixations

    class Subject:
        def __init__(self, x, y):
            self.fix_x = x
            self.fix_y = y
            self.fix_duration = 40 if np.isscalar(x) else [40, 50]

    class Record:
        def __init__(self, name):
            self.img = name
            self.subjects = [Subject(10.0, 12.0), Subject([20.0, 30.0], [22.0, 32.0])]

    names = [f"{i}.jpg" for i in range(30)]
    scanpaths = scanpaths_from_fixations([Record(name) for name in names])
    assert scanpaths[0].x.tolist() == [10.0]
    assert scanpaths[1].x.tolist() == [20.0, 30.0]
    assert scanpaths[0].subject_id == 0
    assert scanpaths[1].subject_id == 1
    assert {scan.split for scan in scanpaths} <= {"train", "valid"}
    assert any(scan.split == "valid" for scan in scanpaths)
    assert osie_split(scanpaths[0].image_name) == scanpaths[0].split


def test_b3_matches_other_observers_at_the_same_index_only():
    """Index-conditioned density uses the other observers' fixation at that index."""
    grid = Grid(100, 100, 5, 5)
    scans = [
        _scan("img", 1, "valid", [(10, 10), (10, 10)]),
        _scan("img", 2, "valid", [(10, 10), (90, 90)]),
        _scan("img", 3, "valid", [(10, 10), (90, 90)]),
        _scan("other", 1, "valid", [(12, 12), (50, 50)]),
        _scan("other", 2, "valid", [(14, 14), (52, 52)]),
    ]
    records = _records(scans, grid)
    center = np.full((5, 5), 1.0 / 25.0)
    kernel = np.full((9, 9), 1.0 / 81.0)
    from humangazelab.temporal import canvas_kernel_mass, density_from_counts

    frame = evaluate_records(
        records,
        grid,
        center,
        {1: center, 2: center, 3: center},
        0.0,
        kernel,
        canvas_kernel_mass(kernel, grid),
        0.05,
        0.0,
        0.5,
        {"img": "other", "other": "img"},
        index_min=1,
    )
    alpha = 0.05
    n_bins = 25
    # Subject 1 at index 1 is alone in its bin. Sigma 0 leaves only the floor there.
    alone = frame[(frame["image"] == "img") & (frame["subject"] == 1) & (frame["fixation_index"] == 1)]
    assert len(alone) == 1
    assert alone["ig_B3_loo_index"].iloc[0] == pytest.approx(math.log2(alpha))
    # Subjects 2 and 3 share a bin. Each reference is one co-located fixation and one elsewhere.
    shared = frame[(frame["image"] == "img") & (frame["subject"] == 2) & (frame["fixation_index"] == 1)]
    expected = (1.0 - alpha) * 0.5 + alpha / n_bins
    assert shared["ig_B3_loo_index"].iloc[0] == pytest.approx(math.log2(expected * n_bins))
    # The same number comes from blurring the other observers' index-1 counts directly.
    explicit = density_from_counts(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=float) + np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float), 0.0, alpha)
    # bins: subject 1 at (0,0), subject 3 at (4,4). Subject 2 is also at (4,4), so its reference is those two.
    assert explicit[4, 4] == pytest.approx(expected)
    # Moving another observer's index 0 must not change index-1 B3.
    order = ["image", "subject", "fixation_index"]
    before = frame.sort_values(order)["ig_B3_loo_index"].to_numpy().copy()
    for record in records:
        if record.image == "img" and record.subject == 2:
            record.ix[0] = 4
            record.iy[0] = 0
    after = evaluate_records(
        records, grid, center, {1: center, 2: center, 3: center}, 0.0, kernel,
        canvas_kernel_mass(kernel, grid), alpha, 0.0, 0.5, {"img": "other", "other": "img"}, 1,
    )
    assert np.allclose(before, after.sort_values(order)["ig_B3_loo_index"].to_numpy())


def test_b3_blurred_shortcut_matches_the_normalized_map():
    grid = Grid(80, 80, 4, 4)
    scans = [
        _scan("img", 1, "train", [(10, 10), (10, 70)]),
        _scan("img", 2, "train", [(12, 12), (70, 10)]),
        _scan("img", 3, "train", [(40, 40), (70, 70)]),
        _scan("pad", 1, "train", [(8, 8), (20, 20)]),
        _scan("pad", 2, "train", [(8, 8), (22, 22)]),
    ]
    records = _records(scans, grid)
    center = np.full((4, 4), 1.0 / 16.0)
    kernel = np.full((7, 7), 1.0 / 49.0)
    from humangazelab.temporal import canvas_kernel_mass, density_from_counts, histogram_counts

    sigma = 0.8
    alpha = 0.05
    frame = evaluate_records(
        records, grid, center, {1: center, 2: center, 3: center}, sigma, kernel,
        canvas_kernel_mass(kernel, grid), alpha, 0.25, 0.25, {"img": "pad", "pad": "img"}, 1,
    )
    group = [record for record in records if record.image == "img"]
    for record in group:
        others = [other for other in group if other.subject != record.subject and other.length > 1]
        counts = np.zeros((4, 4))
        for other in others:
            counts += histogram_counts(other.iy[1:2], other.ix[1:2], 4, 4)
        density = density_from_counts(counts, sigma, alpha)
        got = frame[(frame["image"] == "img") & (frame["subject"] == record.subject) & (frame["fixation_index"] == 1)]
        probability = 2.0 ** got["ig_B3_loo_index"].iloc[0] / 16.0
        assert probability == pytest.approx(density[int(record.iy[1]), int(record.ix[1])], rel=1e-8, abs=1e-12)
        assert density.min() > 0
        assert density.sum() == pytest.approx(1.0)


def test_index_zero_has_no_history_probability():
    grid = Grid(100, 100, 5, 5)
    scans = [
        _scan("a", 1, "valid", [(10, 10), (80, 80)]),
        _scan("a", 2, "valid", [(12, 12), (82, 82)]),
        _scan("b", 1, "valid", [(10, 10), (20, 20)]),
        _scan("b", 2, "valid", [(12, 12), (22, 22)]),
    ]
    records = _records(scans, grid)
    center = np.full((5, 5), 1.0 / 25.0)
    kernel = np.full((9, 9), 1.0 / 81.0)
    from humangazelab.temporal import canvas_kernel_mass

    frame = evaluate_records(
        records, grid, center, {1: center, 2: center}, 0.0, kernel, canvas_kernel_mass(kernel, grid),
        0.05, 0.0, 0.0, {"a": "b", "b": "a"}, index_min=0,
    )
    first = frame[frame["fixation_index"] == 0]
    later = frame[frame["fixation_index"] == 1]
    assert len(first) == 4
    assert np.isnan(first["ig_B4_history"]).all()
    assert np.isfinite(first["ig_B1_center"]).all()
    assert np.isfinite(later["ig_B4_history"]).all()
    assert first["order_control_degenerate"].all()


def test_primary_evaluation_drops_index_zero_and_length_one():
    grid = Grid(100, 100, 5, 5)
    scans = [
        _scan("a", 1, "valid", [(10, 10)]),
        _scan("a", 2, "valid", [(10, 10), (80, 80), (82, 80)]),
        _scan("b", 1, "valid", [(12, 12), (20, 20), (80, 20)]),
        _scan("b", 2, "valid", [(12, 12), (20, 20), (80, 20)]),
    ]
    records = _records(scans, grid)
    center = np.full((5, 5), 1.0 / 25.0)
    kernel = np.zeros((9, 9))
    kernel[4, 5] = 1.0  # one bin to the right
    from humangazelab.temporal import canvas_kernel_mass

    mass = canvas_kernel_mass(kernel, grid)
    shuffle = {"a": "b", "b": "a"}
    frame = evaluate_records(
        records,
        grid,
        center,
        {1: center, 2: center},
        loo_sigma_bins=0.0,
        kernel=kernel,
        kernel_mass=mass,
        alpha=0.05,
        lambda_b5=0.5,
        lambda_b5c=0.5,
        shuffle_map=shuffle,
        index_min=1,
    )
    assert frame["fixation_index"].min() == 1
    assert set(frame["image"]) == {"a", "b"}
    assert frame["scan_id"].nunique() == 3
    assert np.allclose(frame["ig_B0_uniform"], 0.0)
    assert np.isfinite(frame["ig_B4_history"]).all()
    assert frame["p_B2_loo_image"].min() > 0
    # Same-image observers agree on b, so LOO should beat the shuffled-image reference.
    b_rows = frame[frame["image"] == "b"]
    assert b_rows["ig_B2_loo_image"].mean() > b_rows["ig_B6_image_shuffle"].mean()


def test_bootstrap_interval_is_reproducible():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    first = percentile_bootstrap_mean(values, 400, 20261114)
    second = percentile_bootstrap_mean(values, 400, 20261114)
    assert first == second
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]


def test_duplicate_scanpath_is_rejected():
    grid = Grid(40, 40, 4, 4)
    scans = [
        _scan("a", 1, "train", [(5, 5), (10, 10)]),
        _scan("a", 1, "train", [(6, 6), (11, 11)]),
    ]
    try:
        _records(scans, grid)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate scanpath was accepted")


def test_index0_inclusion_case_uses_the_frozen_map():
    from humangazelab.failure_atlas import index0_inclusion_cases

    grid = Grid(100, 100, 5, 5)
    scans = [
        _scan("a.jpg", 1, "valid", [(10, 10), (50, 50), (80, 80)]),
        _scan("a.jpg", 2, "valid", [(10, 10), (52, 50), (82, 80)]),
        _scan("b.jpg", 1, "valid", [(10, 10), (50, 50)]),
        _scan("b.jpg", 2, "valid", [(12, 12), (55, 55)]),
    ]
    records = _records(scans, grid)
    center = np.full((5, 5), 1.0 / 25.0)
    center[0, 0] = 0.2
    center /= center.sum()
    images = pd.DataFrame(
        {
            "image": ["a.jpg", "b.jpg"],
            "n_subjects": [2, 2],
            "n_fixations": [4, 2],
            "ig_B1_center": [0.1, 0.2],
            "ig_B2_loo_image": [0.3, 0.2],
            "ig_B4_history": [0.4, 0.2],
            "ig_B3_loo_index": [0.0, 0.0],
        }
    )
    cases = index0_inclusion_cases(records, center, grid, images, n_examples=1, min_observers=2)
    assert len(cases) == 1
    assert cases.iloc[0]["rule"] == "largest_change_when_index0_included"
    assert np.isfinite(cases.iloc[0]["delta_including_minus_primary"])


def test_failure_rule_stays_inside_the_quartile_intersection():
    from humangazelab.failure_atlas import select_cases

    rows = []
    for i in range(20):
        rows.append(
            {
                "image": f"im{i}.jpg",
                "n_subjects": 10,
                "n_fixations": 100,
                "ig_B1_center": i / 20,
                "ig_B2_loo_image": (20 - i) / 20,
                "ig_B4_history": 0.2,
                "ig_B3_loo_index": 0.0,
            }
        )
    cases = select_cases(pd.DataFrame(rows), n_examples=3, high_q=0.75, low_q=0.25, min_observers=8)
    high = cases[cases["rule"] == "high_cross_observer_low_center"]
    assert len(high) > 0
    assert (high["ig_B2_loo_image"] >= high["b2_high_quartile"]).all()
    assert (high["ig_B1_center"] <= high["b1_low_quartile"]).all()
    assert high["rank_value"].is_monotonic_decreasing


def test_coordinate_table_matches_bins_used_for_scoring():
    grid = Grid(100, 100, 10, 10)
    scans = [
        _scan("a", 1, "train", [(1, 1), (50, 50)]),
        _scan("a", 2, "train", [(1, 1), (50, 50)]),
    ]
    records = _records(scans, grid)
    assert records[0].ix[0] == bin_index(np.array([1.0]), 100, 10)[0]
    assert records[0].iy[1] == bin_index(np.array([50.0]), 100, 10)[0]
    frame = evaluate_records(
        records,
        grid,
        np.full((10, 10), 0.01),
        {1: np.full((10, 10), 0.01), 2: np.full((10, 10), 0.01)},
        0.0,
        np.full((19, 19), 1.0 / (19 * 19)),
        np.ones((10, 10)),
        0.05,
        0.0,
        0.0,
        {"a": "a"},
        1,
    )
    # A self-shuffle is not used in the experiment. Here it only checks column alignment.
    assert len(frame) == 2
    assert set(frame["fixation_ix"]) == {records[0].ix[1]}


def test_crossed_bootstrap_is_constant_when_every_cell_is_constant(tmp_path):
    from humangazelab.audit_checkpoint import assert_not_frozen, two_way_subject_bootstrap

    subjects = np.repeat([1, 2, 3], 4)
    images = np.tile(["a", "b", "c", "d"], 3)
    values = np.full(subjects.shape, 1.25)
    summary = two_way_subject_bootstrap(subjects, images, values, 40, 7)
    assert summary["estimate"] == pytest.approx(1.25)
    assert summary["ci_low_crossed"] == pytest.approx(1.25)
    assert summary["ci_high_crossed"] == pytest.approx(1.25)
    again = two_way_subject_bootstrap(subjects, images, values + np.array([0, 1, 0, 0] * 3), 30, 7)
    repeat = two_way_subject_bootstrap(subjects, images, values + np.array([0, 1, 0, 0] * 3), 30, 7)
    assert again["ci_low_crossed"] == repeat["ci_low_crossed"]
    assert again["ci_high_crossed"] == repeat["ci_high_crossed"]
    with pytest.raises(RuntimeError):
        assert_not_frozen(tmp_path, tmp_path / "outputs" / "temporal_v1")
