"""Stimulus alignment, radius matching, and the unscored MIT1003 boundary."""

import numpy as np
import pytest

from humangazelab.image_features import canvas_radius, radius_matched_points, window_features
from humangazelab.mit1003 import FixationExtractionRefused, assign_partition, extract_fixations
from humangazelab.stimuli import align_fixations


def test_alignment_rejects_the_wrong_frame_and_a_coordinate_outside_it():
    gray = np.zeros((4, 6))
    with pytest.raises(ValueError):
        align_fixations(gray, np.array([1.0]), np.array([1.0]), 5, 6)
    with pytest.raises(ValueError):
        align_fixations(gray, np.array([6.0]), np.array([1.0]), 4, 6)
    report = align_fixations(gray, np.array([0.0, 5.9]), np.array([0.0, 3.9]), 4, 6)
    assert report["all_inside"] is True
    assert report["visual_angle_aligned"] is False


def test_radius_match_keeps_distance_from_center_and_a_radial_feature_matches():
    yy, xx = np.mgrid[0:80, 0:100]
    gray = np.hypot(xx - 50, yy - 40)
    xs = np.array([70.0, 20.0, 50.0])
    ys = np.array([40.0, 40.0, 10.0])
    rng = np.random.default_rng(20261114)
    matched_x, matched_y = radius_matched_points(xs, ys, 100, 80, rng, margin=2)
    assert np.allclose(canvas_radius(xs, ys, 100, 80), canvas_radius(matched_x, matched_y, 100, 80))
    fixation = window_features(gray, xs, ys, 2)["luminance"]
    matched = window_features(gray, matched_x, matched_y, 2)["luminance"]
    assert np.allclose(fixation, matched, atol=1.5)


def test_mit1003_partition_is_deterministic_and_extraction_is_refused():
    assert assign_partition("i05june05_static_street_boston_p1010764") == assign_partition(
        "i05june05_static_street_boston_p1010764"
    )
    labels = {assign_partition(f"image-{index}") for index in range(400)}
    assert labels == {"train", "tune", "final"}
    with pytest.raises(FixationExtractionRefused):
        extract_fixations("DATA/ajs/example.mat")
