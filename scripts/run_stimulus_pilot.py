"""Descriptive contrast at fixations on the images already extracted locally.

This is not a feature model. It does not estimate center bias plus features,
and it does not use a pretrained embedding. The 18 COCO-FreeView images are
the descriptive subset. The 9 OSIE images are the overlay sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from humangazelab.adapter import load_scanpaths
from humangazelab.osie import load_osie_scanpaths
from humangazelab.protocolshift import assert_new_output

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "protocolshift_b3_v1"
RADIUS = 8
SEED = 20261114


def _gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def _window_means(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(gray)
    energy = np.hypot(gx, gy)
    height, width = gray.shape
    luminance = np.empty(len(xs))
    gradient = np.empty(len(xs))
    for index, (x_value, y_value) in enumerate(zip(xs, ys)):
        x_int = int(np.clip(round(float(x_value)), 0, width - 1))
        y_int = int(np.clip(round(float(y_value)), 0, height - 1))
        x0, x1 = max(0, x_int - radius), min(width, x_int + radius + 1)
        y0, y1 = max(0, y_int - radius), min(height, y_int + radius + 1)
        luminance[index] = float(gray[y0:y1, x0:x1].mean())
        gradient[index] = float(energy[y0:y1, x0:x1].mean())
    return luminance, gradient


def _summarize(name: str, image_dir: Path, scanpaths, width: float, height: float) -> dict:
    files = {path.name: path for path in image_dir.glob("*.jpg")}
    fixation_l, fixation_g, random_l, random_g = [], [], [], []
    fixation_radius, random_radius = [], []
    used_images = 0
    rng = np.random.default_rng(SEED)
    for image_name, path in sorted(files.items()):
        gray = _gray(path)
        if gray.shape != (int(height), int(width)):
            raise ValueError(f"{image_name} is {gray.shape}, expected {(int(height), int(width))}")
        points = [(float(x), float(y)) for scan in scanpaths if scan.image_name == image_name for x, y in zip(scan.x, scan.y)]
        if not points:
            continue
        used_images += 1
        xs = np.array([point[0] for point in points])
        ys = np.array([point[1] for point in points])
        lum, grad = _window_means(gray, xs, ys, RADIUS)
        fixation_l.append(lum)
        fixation_g.append(grad)
        fixation_radius.append(np.hypot(xs - width / 2.0, ys - height / 2.0))
        rx = rng.uniform(RADIUS, width - RADIUS, size=len(xs))
        ry = rng.uniform(RADIUS, height - RADIUS, size=len(ys))
        rlum, rgrad = _window_means(gray, rx, ry, RADIUS)
        random_l.append(rlum)
        random_g.append(rgrad)
        random_radius.append(np.hypot(rx - width / 2.0, ry - height / 2.0))
    fl = np.concatenate(fixation_l)
    fg = np.concatenate(fixation_g)
    rl = np.concatenate(random_l)
    rg = np.concatenate(random_g)
    return {
        "dataset": name,
        "n_images": used_images,
        "n_fixations": int(fl.size),
        "window_radius_px": RADIUS,
        "mean_luminance_at_fixations": float(fl.mean()),
        "mean_luminance_at_random_pixels": float(rl.mean()),
        "mean_gradient_at_fixations": float(fg.mean()),
        "mean_gradient_at_random_pixels": float(rg.mean()),
        "mean_radius_fixations_px": float(np.concatenate(fixation_radius).mean()),
        "mean_radius_random_px": float(np.concatenate(random_radius).mean()),
        "fitted_model": False,
        "pretrained_embedding": False,
        "object_masks": False,
        "note": "Random pixels are a within-image descriptive comparison. They are not a center-bias control.",
    }


def main() -> None:
    assert_new_output(ROOT, OUT)
    coco = load_scanpaths(ROOT / "data" / "raw" / "COCOFreeView_fixations_trainval.json")
    osie = load_osie_scanpaths(ROOT / "data" / "raw" / "external" / "osie_fixations.mat")
    rows = [
        _summarize("COCO-FreeView", ROOT / "data" / "raw" / "images_subset", coco, 1680, 1050),
        _summarize("OSIE", ROOT / "data" / "raw" / "external" / "osie_stimuli", osie, 800, 600),
    ]
    pd.DataFrame(rows).to_csv(OUT / "stimulus_pilot_contrast.csv", index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
