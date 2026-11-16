"""Checks against the public train+validation file when it is present locally."""

from pathlib import Path

import pytest

from humangazelab.adapter import load_scanpaths
from humangazelab.quality import quality_report

ROOT = Path(__file__).resolve().parents[1]
FIXATION = ROOT / "data" / "raw" / "COCOFreeView_fixations_trainval.json"
EXPECTED_SHA256 = "771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730"


def test_committed_readme_states_canvas_duration_and_initial_fixation():
    text = (ROOT / "docs" / "sources" / "COCO-FreeView-readme.txt").read_text()
    assert "1680x1050" in text
    assert "fixation durations" in text
    assert "initial center fixation" in text
    assert "train" in text and "valid" in text


@pytest.mark.skipif(not FIXATION.exists(), reason="public fixation file is not downloaded")
def test_public_file_hash_and_counts():
    import hashlib

    digest = hashlib.sha256(FIXATION.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    scanpaths = load_scanpaths(FIXATION)
    report = quality_report(scanpaths, 1680, 1050)
    assert report["n_scanpaths"] == 43048
    assert report["n_fixations"] == 665005
    assert report["n_images"] == 4317
    assert report["subject_ids"] == [101, 102, 103, 104, 105, 107, 108, 109, 110, 116]
    assert report["split_scanpath_counts"] == {"train": 37038, "valid": 6010}
    assert report["split_image_counts"] == {"train": 3714, "valid": 603}
    assert report["condition_counts"] == {"freeview": 43048}
    assert report["correct_counts"] == {"1": 43048}
    assert report["fix_on_target_counts"] == {"False": 43048}
    assert report["duplicate_image_subject_split"] == 0
    assert report["n_fixations_outside_canvas"] == 0
    assert report["n_sequences_not_strictly_increasing_duration"] == 42902
    assert report["n_sequences_nondecreasing_duration"] == 147
    assert report["images_by_n_subjects"] == {9: 122, 10: 4195}
    assert report["fraction_first_fixation_within_50px"] == pytest.approx(0.8669160007433563)
    assert report["first_fixation_radius_median"] == pytest.approx(22.100565601016555)
    assert report["x_min"] >= 0
    assert report["x_max"] <= 1680
    assert report["y_min"] >= 0
    assert report["y_max"] <= 1050
