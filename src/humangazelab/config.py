"""Load configs/analysis.yaml into a typed object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AnalysisConfig:
    canvas_width_px: float
    canvas_height_px: float
    canvas_source: str
    drop_initial_fixation: bool
    early_k_primary: int
    early_k_sensitivity: tuple[int, ...]
    center_radius_primary_px: float
    center_radii_px: tuple[float, ...]
    early_time_primary_ms: float
    early_time_sensitivity_ms: tuple[float, ...]
    n_shuffles: int
    shuffle_seed: int
    n_bootstrap: int
    bootstrap_seed: int
    uniform_mc_n: int
    uniform_mc_seed: int
    density_bins_x: int
    density_bins_y: int
    fixation_path: str
    readme_path: str
    image_dir: str
    output_dir: str
    n_train_images: int
    n_valid_images: int
    image_seed: int
    min_subjects: int
    prefer_archive: str
    fixation_url: str
    readme_url: str
    test_probe_url: str
    project_page: str
    stimulus_page: str
    tp_zip_url: str
    ta_zip_url: str
    readme_page: str


def load_config(path: str | Path) -> AnalysisConfig:
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    analysis = raw["analysis"]
    canvas = raw["canvas"]
    dataset = raw["dataset"]
    images = raw["images"]
    radii = tuple(float(v) for v in analysis["center_radii_px"])
    primary_radius = float(analysis["center_radius_primary_px"])
    if primary_radius not in radii:
        raise ValueError("center_radius_primary_px must be one of center_radii_px")
    half_min = min(float(canvas["width_px"]), float(canvas["height_px"])) / 2.0
    if any(r <= 0 or r > half_min for r in radii):
        raise ValueError(
            "center radii must be positive and no larger than half the shorter canvas side "
            "so the comparison disk lies inside the canvas"
        )
    return AnalysisConfig(
        canvas_width_px=float(canvas["width_px"]),
        canvas_height_px=float(canvas["height_px"]),
        canvas_source=str(canvas["source"]),
        drop_initial_fixation=bool(analysis["drop_initial_fixation"]),
        early_k_primary=int(analysis["early_k_primary"]),
        early_k_sensitivity=tuple(int(v) for v in analysis["early_k_sensitivity"]),
        center_radius_primary_px=primary_radius,
        center_radii_px=radii,
        early_time_primary_ms=float(analysis["early_time_primary_ms"]),
        early_time_sensitivity_ms=tuple(float(v) for v in analysis["early_time_sensitivity_ms"]),
        n_shuffles=int(analysis["n_shuffles"]),
        shuffle_seed=int(analysis["shuffle_seed"]),
        n_bootstrap=int(analysis["n_bootstrap"]),
        bootstrap_seed=int(analysis["bootstrap_seed"]),
        uniform_mc_n=int(analysis["uniform_mc_n"]),
        uniform_mc_seed=int(analysis["uniform_mc_seed"]),
        density_bins_x=int(analysis["density_bins_x"]),
        density_bins_y=int(analysis["density_bins_y"]),
        fixation_path=str(dataset["fixation_path"]),
        readme_path=str(dataset["readme_path"]),
        image_dir=str(dataset["image_dir"]),
        output_dir=str(raw["output_dir"]),
        n_train_images=int(images["n_train"]),
        n_valid_images=int(images["n_valid"]),
        image_seed=int(images["seed"]),
        min_subjects=int(images["min_subjects"]),
        prefer_archive=str(images["prefer_archive"]),
        fixation_url=str(dataset["fixation_url"]),
        readme_url=str(dataset["readme_url"]),
        test_probe_url=str(dataset["test_probe_url"]),
        project_page=str(dataset["project_page"]),
        stimulus_page=str(dataset["stimulus_page"]),
        tp_zip_url=str(dataset["tp_zip_url"]),
        ta_zip_url=str(dataset["ta_zip_url"]),
        readme_page=str(dataset["readme_page"]),
    )
