"""Annotation unions and geometry-only example selection. No network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from humangazelab.osie_semantics import (
    VERIFIED_ATTR_NAMES,
    OsieAnnotations,
    overlap_pixels,
    select_alignment_cases,
    stage_a_stems,
)


def test_overlap_counts_shared_pixels_once():
    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[:2, :2] = True
    second[:2, 1:3] = True
    assert overlap_pixels([first, second]) == 2


def test_alignment_cases_follow_area_and_do_not_need_scores():
    manifest = pd.DataFrame(
        [
            {"image": "a", "object_index": 0, "area_px": 10, "touches_border": False, "positive_labels": "face", "image_overlap_px": 0},
            {"image": "a", "object_index": 1, "area_px": 3, "touches_border": True, "positive_labels": "text", "image_overlap_px": 0},
            {"image": "b", "object_index": 0, "area_px": 50, "touches_border": False, "positive_labels": "face,text", "image_overlap_px": 9},
            {"image": "c", "object_index": 0, "area_px": 8, "touches_border": True, "positive_labels": "", "image_overlap_px": 1},
        ]
    )
    cases = select_alignment_cases(manifest)
    by_reason = dict(zip(cases["reason"], cases["image"]))
    assert by_reason["smallest_object"] == "a"
    assert by_reason["largest_object"] == "b"
    assert by_reason["smallest_face"] == "a"
    assert by_reason["largest_text"] == "b"
    assert by_reason["smallest_border_object"] == "a"
    assert by_reason["most_overlap"] == "b"
    again = select_alignment_cases(manifest)
    assert cases.to_dict("list") == again.to_dict("list")


def test_stage_a_includes_the_reproduction_image_and_the_stride():
    images = [str(1001 + index) for index in range(100)]
    manifest = pd.DataFrame(
        {
            "image": images,
            "object_index": 0,
            "area_px": 10,
            "touches_border": [index == 0 for index in range(100)],
            "positive_labels": "face,text",
            "image_overlap_px": 1,
        }
    )
    chosen = stage_a_stems(manifest)
    assert "1001" in chosen
    assert images[50] in chosen
    assert chosen == tuple(sorted(chosen))


@pytest.mark.skipif(not Path("data/raw/external/osie_attrs.mat").exists(), reason="attrs.mat is gitignored")
def test_released_annotations_match_the_verified_inventory():
    with OsieAnnotations("data/raw/external/osie_attrs.mat") as annotations:
        assert annotations.names == VERIFIED_ATTR_NAMES
        assert len(annotations.stems) == 700
        objects = annotations.objects("1001")
        assert objects
        assert all(obj.mask.shape == (600, 800) for obj in objects)
        labels = annotations.label_masks("1001")
        assert set(labels) == set(VERIFIED_ATTR_NAMES) | {"object_union"}
        face = labels["face"]
        union = labels["object_union"]
        assert face.sum() <= union.sum()
        assert not (face & ~union).any()
