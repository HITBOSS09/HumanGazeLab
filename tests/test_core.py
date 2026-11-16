import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from humangazelab.adapter import Scanpath, load_scanpaths
from humangazelab.config import load_config
from humangazelab.features import (
    epoch_distance_means,
    extract_feature_table,
    mean_successive_distance,
    shuffled_mean_distance,
    temporal_early_count,
    trial_onset,
)
from humangazelab.imageselect import select_image_subset
from humangazelab.prepare import archive_match_summary, choose_entry, zero_padding_widths
from humangazelab.spatial import disk_fraction, grid_mean_radius, uniform_mean_radius
from humangazelab.summarize import percentile_bootstrap_mean, summarize_column
from humangazelab.zipremote import ZipEntry

ROOT = Path(__file__).resolve().parents[1]


def test_epoch_boundary_is_excluded():
    x = np.array([0.0, 1.0, 2.0, 3.0, 6.0])
    y = np.zeros(5)
    epochs = epoch_distance_means(x, y, 3)
    assert epochs["early"] == pytest.approx(1.0)
    assert epochs["late"] == pytest.approx(3.0)
    assert epochs["boundary"] == pytest.approx(1.0)


def test_post_initial_distance_drops_the_first_segment():
    x = np.array([0.0, 10.0, 11.0])
    y = np.zeros(3)
    assert mean_successive_distance(x, y) == pytest.approx(5.5)
    assert mean_successive_distance(x[1:], y[1:]) == pytest.approx(1.0)


def test_shuffle_is_deterministic_and_longer_on_a_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.zeros(4)
    first = shuffled_mean_distance(x, y, np.random.default_rng(0), 30)
    second = shuffled_mean_distance(x, y, np.random.default_rng(0), 30)
    assert first == second
    assert first > mean_successive_distance(x, y)


def test_shuffle_regression_value_for_three_points():
    value = shuffled_mean_distance(
        np.array([0.0, 1.0, 3.0]),
        np.zeros(3),
        np.random.default_rng(0),
        5,
    )
    assert value == pytest.approx(2.1)


def test_temporal_onset_keeps_initial_duration_in_the_clock():
    duration = np.array([300.0, 400.0, 500.0, 800.0])
    onset = trial_onset(duration)
    assert onset.tolist() == [0.0, 300.0, 700.0, 1200.0]
    assert temporal_early_count(onset[1:], 1000) == 2


def test_disk_fraction_is_exact_inside_the_canvas_and_rejects_clipping():
    width, height, radius = 1680.0, 1050.0, 200.0
    assert disk_fraction(width, height, radius) == pytest.approx(math.pi * radius**2 / (width * height))
    with pytest.raises(ValueError):
        disk_fraction(width, height, 526)


def test_uniform_mean_matches_a_fine_grid():
    mc = uniform_mean_radius(30, 20, n=250_000, seed=4)
    grid = grid_mean_radius(30, 20, step=0.5)
    assert abs(mc["mean"] - grid) < 0.05


def test_bootstrap_weights_groups_equally_when_group_sizes_differ():
    frame = pd.DataFrame({"subject_id": [1, 1, 1, 2], "value": [0.0, 0.0, 0.0, 10.0]})
    stats = summarize_column(frame, "value", "subject_id", frame["subject_id"].notna(), n_boot=400, seed=1)
    assert stats["estimate"] == pytest.approx(5.0)
    assert stats["n_groups"] == 2
    assert stats["n_scanpaths"] == 4


def test_constant_bootstrap_interval_is_the_constant():
    stats = percentile_bootstrap_mean(np.array([3.0, 3.0, 3.0]), n_boot=50, seed=0)
    assert stats["ci_low"] == pytest.approx(3.0)
    assert stats["ci_high"] == pytest.approx(3.0)
    assert stats["ci_low"] <= stats["ci_high"]


def test_adapter_rejects_a_length_mismatch(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '[{"name":"a.jpg","subject":1,"split":"train","X":[0,1],"Y":[0],"T":[10,10],'
        '"length":2,"condition":"freeview"}]'
    )
    with pytest.raises(ValueError, match="lengths"):
        load_scanpaths(path)


def test_image_subset_is_seeded_and_ignores_row_order():
    names = [f"{i:012d}.jpg" for i in range(30)]
    table = pd.DataFrame(
        {
            "image_name": names,
            "split": ["train"] * 20 + ["valid"] * 10,
            "n_subjects": [10] * 30,
        }
    )
    shuffled = table.sample(frac=1, random_state=1)
    first = select_image_subset(table, n_train=4, n_valid=2, seed=11, min_subjects=10)
    second = select_image_subset(shuffled, n_train=4, n_valid=2, seed=11, min_subjects=10)
    assert first["image_name"].tolist() == second["image_name"].tolist()
    assert (first["split"] == "train").sum() == 4
    assert (first["split"] == "valid").sum() == 2
    assert first["n_subjects"].min() >= 10


def test_archive_choice_prefers_tp_when_crc_matches():
    entry_tp = ZipEntry("images/cat/a.jpg", 8, 5, 10, 12, 0)
    entry_ta = ZipEntry("ta/a.jpg", 8, 5, 10, 12, 0)
    other = ZipEntry("ta/b.jpg", 8, 9, 10, 12, 0)
    tp = {"a.jpg": [entry_tp]}
    ta = {"a.jpg": [entry_ta], "b.jpg": [other]}
    choice = choose_entry("a.jpg", tp, ta, prefer="TP")
    assert choice["conflict"] is False
    assert choice["archive"] == "TP"
    summary = archive_match_summary({"a.jpg", "b.jpg", "c.jpg"}, tp, ta)
    assert summary["matched_both"] == 1
    assert summary["both_crc_equal"] == 1
    assert summary["matched_ta_only"] == 1
    assert summary["matched_neither"] == 1
    conflict = choose_entry("a.jpg", {"a.jpg": [entry_tp]}, {"a.jpg": [ZipEntry("x", 8, 7, 1, 1, 0)]}, "TP")
    assert conflict["conflict"] is True


def test_zero_padding_widths_count_black_borders_only():
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[1:5, 2:7] = 20
    pads = zero_padding_widths(rgb)
    assert pads["top"] == 1
    assert pads["bottom"] == 1
    assert pads["left"] == 2
    assert pads["right"] == 1


def test_feature_table_and_summary_run_on_synthetic_scanpaths():
    scanpaths = [
        Scanpath(0, "000000000001.jpg", 1, "train", "freeview",
                 np.array([840.0, 100.0, 110.0, 400.0, 900.0]),
                 np.array([525.0, 200.0, 210.0, 500.0, 600.0]),
                 np.array([200.0, 180.0, 180.0, 400.0, 900.0]), False, 1),
        Scanpath(1, "000000000002.jpg", 2, "train", "freeview",
                 np.array([840.0, 800.0, 820.0, 830.0, 100.0, 120.0]),
                 np.array([525.0, 100.0, 120.0, 140.0, 800.0, 820.0]),
                 np.array([220.0, 200.0, 200.0, 200.0, 700.0, 700.0]), False, 1),
        Scanpath(2, "000000000003.jpg", 1, "valid", "freeview",
                 np.array([830.0, 200.0, 220.0, 600.0]),
                 np.array([520.0, 300.0, 320.0, 700.0]),
                 np.array([250.0, 300.0, 300.0, 1200.0]), False, 1),
    ]
    config = load_config(ROOT / "configs" / "analysis.yaml")
    features = extract_feature_table(scanpaths, config)
    assert len(features) == 3
    assert features.loc[0, "mean_saccade_post"] == pytest.approx(
        mean_successive_distance(scanpaths[0].x[1:], scanpaths[0].y[1:])
    )
    from humangazelab.pipeline import build_summary
    from humangazelab.spatial import disk_fraction as disk
    from humangazelab.spatial import uniform_mean_radius

    mc = uniform_mean_radius(config.canvas_width_px, config.canvas_height_px, 5000, 1)
    comparisons = {
        "uniform_mean_radius_px": mc["mean"],
        "disk_fraction_200": disk(config.canvas_width_px, config.canvas_height_px, 200),
        "disk_fraction_100": disk(config.canvas_width_px, config.canvas_height_px, 100),
        "disk_fraction_400": disk(config.canvas_width_px, config.canvas_height_px, 400),
    }
    # Keep the synthetic check small: the real bootstrap count is a config value,
    # and 2000 resamples of two groups is still cheap.
    summary = build_summary(features, config, comparisons)
    primary = summary[(summary["role"] == "primary") & (summary["subset"] == "all") & (summary["estimand"] == "subject")]
    radius = primary[primary["metric"] == "mean_radial_distance_post_px"].iloc[0]
    assert radius.n_groups == 2
    assert math.isfinite(radius.estimate)
    finite = primary[primary["estimate"].notna()]
    assert (finite["ci_low"] <= finite["ci_high"]).all()


def test_plot_smoke(tmp_path):
    from humangazelab.plotting import plot_density_pair

    mean = np.zeros((4, 3))
    mean[1, 1] = 0.2
    plot_density_pair(mean, mean, 1680, 1050, tmp_path / "density.png", "synthetic")
    assert (tmp_path / "density.png").stat().st_size > 1000
