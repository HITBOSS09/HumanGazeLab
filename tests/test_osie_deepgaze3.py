"""Pairing rules for the frozen DeepGaze III OSIE expansion. No network is loaded."""

from pathlib import Path

import numpy as np
import pandas as pd

from humangazelab.adapter import Scanpath
from humangazelab.osie_deepgaze import paired_targets, previously_scored_stems, scaled_pixel, summarize_paired


def _scan(image: str, subject: int, length: int, x0: float = 10.0) -> Scanpath:
    x = np.arange(length, dtype=float) + x0
    y = np.full(length, 20.0)
    return Scanpath(0, image, subject, "train", "freeview", x, y, np.ones(length), None, None)


def test_resized_pixel_matches_the_nine_image_half_pixel_case():
    # 338.4 * 875 / 600 is 493.5. NumPy rounds halves to even, which is 494.
    # Multiplying by the precomputed ratio 875/600 lands just below the half.
    assert scaled_pixel(338.4, 600, 875) == 494
    assert int(np.rint(np.float64(338.4) * (875 / 600.0))) == 493


def test_previously_scored_stems_are_the_nine_committed_images():
    stems = previously_scored_stems(Path("outputs/baseline_v1/deepgaze3_osie_fixations.csv"))
    assert stems == ("1001", "1009", "1116", "1138", "1189", "1275", "1521", "1650", "1660")


def test_partner_is_the_next_image_long_enough_and_short_histories_are_not_targets():
    scans = [
        _scan("a.jpg", 0, 6),
        _scan("b.jpg", 0, 5),
        _scan("c.jpg", 0, 8),
    ]
    rows, counts = paired_targets(scans)
    assert counts["n_fixations"] == 19
    assert counts["n_index_0"] == 3
    assert counts["n_index_below_4"] == 12
    assert counts["n_outside_canvas"] == 0
    # a index 4 partners to c (b has length 5, so index 4 exists on b too: length > 4).
    # b length 5 means indices 0..4. index 4 is the only target on b.
    targets = {(row["image"], row["target_index"]): row["partner_image"] for row in rows}
    assert targets[("a", 4)] == "b"
    assert targets[("a", 5)] == "c"
    assert targets[("b", 4)] == "c"
    assert counts["n_scored"] == len(rows)
    assert all(row["target_index"] >= 4 for row in rows)


def test_a_coordinate_outside_the_canvas_is_excluded():
    scan = _scan("a.jpg", 0, 6)
    scan.x[4] = 800.0
    other = _scan("b.jpg", 0, 6)
    rows, counts = paired_targets([scan, other])
    assert counts["n_outside_canvas"] == 1
    assert all(row["target_index"] != 4 or row["image"] != "a" for row in rows)


def test_paired_summary_bootstraps_images_rather_than_raw_fixations():
    frame = pd.DataFrame(
        {
            "image": ["a", "a", "b", "b"],
            "subject": [0, 1, 0, 1],
            "information_gain_bits": [1.0, 3.0, 1.0, 3.0],
            "wrong_image_information_gain_bits": [0.0, 0.0, 0.0, 0.0],
            "paired_difference_bits": [1.0, 3.0, 1.0, 3.0],
        }
    )
    rows = summarize_paired(frame, n_boot=20, seed=20261114, subset="toy")
    image_row = next(row for row in rows if row["column"] == "paired_difference_bits" and row["estimand"] == "image_equal_mean")
    assert image_row["estimate"] == 2.0
    assert image_row["n_groups"] == 2
    assert image_row["n_fixations"] == 4
