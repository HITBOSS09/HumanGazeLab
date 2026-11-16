"""Controls for observer count, index alignment, and canvas-fraction coordinates."""

import math

import numpy as np
import pytest

from humangazelab.adapter import Scanpath
from humangazelab.protocolshift import (
    scale_scanpaths,
    score_index_density,
    seeded_subject_subsets,
)
from humangazelab.temporal import Grid, canvas_kernel_mass, evaluate_records, pack_scanpaths


def _scan(name, subject, split, points):
    return Scanpath(
        source_index=0,
        image_name=name,
        subject_id=subject,
        split=split,
        condition="freeview",
        x=np.array([p[0] for p in points], dtype=float),
        y=np.array([p[1] for p in points], dtype=float),
        duration=np.ones(len(points)),
        fix_on_target=None,
        correct=None,
    )


def test_unit_square_scaling_is_not_applied_twice():
    scans = [_scan("a", 1, "valid", [(0, 0), (800, 600)])]
    scaled = scale_scanpaths(scans, 800, 600)
    assert scaled[0].x.tolist() == [0.0, 1.0]
    assert scaled[0].y.tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        scale_scanpaths(scans, 400, 300)


def test_seeded_subsets_are_reproducible_and_the_requested_size():
    first = seeded_subject_subsets(list(range(15)), 10, 5, 20261114)
    second = seeded_subject_subsets(list(range(15)), 10, 5, 20261114)
    assert first == second
    assert all(len(item) == 10 and len(set(item)) == 10 for item in first)


def test_index_density_matches_the_frozen_evaluator_and_ignores_other_indices():
    grid = Grid(100, 100, 5, 5)
    scans = [
        _scan("img", 1, "valid", [(10, 10), (10, 10)]),
        _scan("img", 2, "valid", [(10, 10), (90, 90)]),
        _scan("img", 3, "valid", [(90, 10), (90, 90)]),
        _scan("pad", 1, "valid", [(12, 12), (50, 50)]),
        _scan("pad", 2, "valid", [(14, 14), (52, 52)]),
    ]
    role = {"train_name": "train", "modulus": 5, "bandwidth_bucket": 0, "mixture_bucket": 1}
    records = pack_scanpaths(scans, grid, role)
    center = np.full((5, 5), 1.0 / 25.0)
    kernel = np.full((9, 9), 1.0 / 81.0)
    evaluated = evaluate_records(
        records, grid, center, {1: center, 2: center, 3: center}, 0.0, kernel,
        canvas_kernel_mass(kernel, grid), 0.05, 0.0, 0.0, {"img": "pad", "pad": "img"}, 1,
    )
    scored = score_index_density(records, grid, 0.0, 0.05, 1, None, True)
    left = evaluated.sort_values(["image", "subject", "fixation_index"])
    right = scored.sort_values(["image", "subject", "fixation_index"])
    assert np.allclose(left["ig_B3_loo_index"], right["ig"])
    # Subject 1 is alone in its index-1 bin. Sigma 0 leaves only the floor.
    alone = right[(right["image"] == "img") & (right["subject"] == 1)]
    assert alone["ig"].iloc[0] == pytest.approx(math.log2(0.05))
    full = score_index_density(records, grid, 0.0, 0.05, 1, None, False)
    reduced_scans = [scan for scan in scans if not (scan.image_name == "img" and scan.subject_id == 3)]
    reduced = score_index_density(pack_scanpaths(reduced_scans, grid, role), grid, 0.0, 0.05, 1, None, False)
    # Removing subject 3 changes subject 2's reference. It does not change subject 1, who was already missing that bin.
    assert not np.allclose(
        full[(full["image"] == "img") & (full["subject"] == 2)]["ig"],
        reduced[(reduced["image"] == "img") & (reduced["subject"] == 2)]["ig"],
    )
