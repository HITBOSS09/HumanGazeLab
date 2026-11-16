"""Score official DeepGaze III where the pixel scale is stated.

OSIE's 2014 paper states about 24 pixels per degree in the released 800 by 600
grid. The DeepGaze README states that models before MSDB were trained at 35
pixels per degree on MIT1003. This script rescales the nine local OSIE images
by 35/24 and scores the natural-log density against the MIT1003 center-bias
template, using the last four fixations before the target.

That information gain is not the uniform-bin information gain in the frozen
experiments. COCO-FreeView is not scored: its dataset paper does not state
pixels per degree. MIT1003 is not scored: those images are the training set,
including any local final split.

DeepGaze MSDB is not run. Its constructor loads CLIP RN50x64 and DINOv2, and
the released checkpoint is the head only.
"""

from __future__ import annotations

import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp

from humangazelab.osie import load_osie_scanpaths, osie_split
from humangazelab.protocolshift import assert_new_output

ROOT = Path(__file__).resolve().parents[1]
DEEPGAZE = Path("/tmp/DeepGaze")
WEIGHTS = Path("/tmp/deepgaze-weights/deepgaze3.pth")
CENTERBIAS = Path("/tmp/deepgaze-weights/centerbias_mit1003.npy")
OSIE_FIXATIONS = ROOT / "data" / "raw" / "external" / "osie_fixations.mat"
OSIE_IMAGES = ROOT / "data" / "raw" / "external" / "osie_stimuli"
OUTPUT = ROOT / "outputs" / "baseline_v1"
OSIE_CANVAS = (800, 600)
OSIE_PX_PER_DEG = 24.0
TRAINING_PX_PER_DEG = 35.0
DEEPGAZE_COMMIT = "c7db17e2d1d7ea6468ffdee2cfaddf141095dcff"


def _centerbias(template: np.ndarray, height: int, width: int) -> np.ndarray:
    scaled = zoom(
        template,
        (height / template.shape[0], width / template.shape[1]),
        order=0,
        mode="nearest",
    )
    scaled = np.asarray(scaled, dtype=np.float64)
    scaled -= logsumexp(scaled)
    return scaled


def _backbone(model, image_tensor: torch.Tensor):
    orig_shape = image_tensor.shape
    resized = F.interpolate(image_tensor, scale_factor=1 / model.downsample, recompute_scale_factor=False)
    features = model.features(resized)
    readout_shape = [
        math.ceil(orig_shape[2] / model.downsample / model.readout_factor),
        math.ceil(orig_shape[3] / model.downsample / model.readout_factor),
    ]
    features = [F.interpolate(item, readout_shape) for item in features]
    return torch.cat(features, dim=1), orig_shape


def _density(model, readout, orig_shape, centerbias: torch.Tensor, x_hist: torch.Tensor, y_hist: torch.Tensor):
    from deepgaze_pytorch.modules import encode_scanpath_features

    readout_shape = [
        math.ceil(orig_shape[2] / model.downsample / model.readout_factor),
        math.ceil(orig_shape[3] / model.downsample / model.readout_factor),
    ]
    predictions = []
    for saliency_network, scanpath_network, selection_network, finalizer in zip(
        model.saliency_networks,
        model.scanpath_networks,
        model.fixation_selection_networks,
        model.finalizers,
    ):
        saliency = saliency_network(readout)
        scanpath = encode_scanpath_features(
            x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=saliency.device
        )
        scanpath = F.interpolate(scanpath, readout_shape)
        selected = selection_network((saliency, scanpath_network(scanpath)))
        predictions.append(finalizer(selected, centerbias)[:, np.newaxis, :, :])
    stacked = torch.cat(predictions, dim=1) - math.log(len(model.saliency_networks))
    return stacked.logsumexp(dim=1, keepdim=True)


def _history(x: np.ndarray, y: np.ndarray, target_index: int) -> tuple[np.ndarray, np.ndarray] | None:
    if target_index < 4:
        return None
    previous_x = np.asarray(x[:target_index], dtype=np.float64)
    previous_y = np.asarray(y[:target_index], dtype=np.float64)
    order = np.array([-1, -2, -3, -4])
    return previous_x[order], previous_y[order]


def _msdb_blocker() -> dict:
    module = sys.modules.get("clip")
    installed = module is not None and bool(getattr(module, "__file__", None))
    return {
        "executed": False,
        "clip_installed": installed,
        "checkpoint_bytes": Path("/tmp/deepgaze-weights/deepgazemsdb.pth").stat().st_size,
        "blocker": "CLIP is not installed." if not installed else "CLIP imported, and the RN50x64 forward was not started.",
        "reason": "The v1.2.0 file is the head. DeepGazeMSDB.__init__ calls clip.load('RN50x64') and a DINOv2 hub load before the head is used. Stub modules for clip and einops were inserted only so the package could import DeepGaze III. No MSDB information gain was computed.",
    }


def main() -> None:
    assert_new_output(ROOT, OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DEEPGAZE))
    # The package __init__ imports DeepGaze MSDB, and that file imports CLIP at
    # import time. DeepGaze III does not call CLIP. A stub lets III load. It is
    # not a CLIP model, and MSDB is still not executed.
    if "clip" not in sys.modules:
        clip_stub = types.ModuleType("clip")

        def _clip_load(*_args, **_kwargs):
            raise RuntimeError("CLIP RN50x64 was not loaded")

        clip_stub.load = _clip_load
        sys.modules["clip"] = clip_stub
    if "einops" not in sys.modules:
        einops_stub = types.ModuleType("einops")

        def _rearrange(*_args, **_kwargs):
            raise RuntimeError("einops.rearrange was not used")

        einops_stub.rearrange = _rearrange
        sys.modules["einops"] = einops_stub
    from deepgaze_pytorch.deepgaze3 import DeepGazeIII

    started = time.time()
    model = DeepGazeIII(pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    head_missing = [key for key in missing if not key.startswith("features.")]
    if head_missing or unexpected:
        raise RuntimeError(
            f"DeepGaze III head did not load cleanly: {len(head_missing)} missing head keys, {len(unexpected)} unexpected"
        )
    model.eval()
    template = np.load(CENTERBIAS)
    scale = TRAINING_PX_PER_DEG / OSIE_PX_PER_DEG
    width = int(round(OSIE_CANVAS[0] * scale))
    height = int(round(OSIE_CANVAS[1] * scale))
    scanpaths = load_osie_scanpaths(OSIE_FIXATIONS)
    available = {path.stem: path for path in OSIE_IMAGES.glob("*.jpg")}
    by_image: dict[str, list] = {}
    for scan in scanpaths:
        stem = Path(scan.image_name).stem
        if stem in available:
            by_image.setdefault(stem, []).append(scan)
    if not by_image:
        raise RuntimeError("none of the local OSIE images match a scanpath name")
    locations = {
        (stem, int(scan.subject_id), index): (float(scan.x[index]), float(scan.y[index]))
        for stem, scans in by_image.items()
        for scan in scans
        for index in range(len(scan.x))
    }
    stems = sorted(by_image)
    wrong_stem = {stem: stems[(index + 1) % len(stems)] for index, stem in enumerate(stems)}

    rows: list[dict] = []
    agreement = None
    with torch.no_grad():
        for image_index, (stem, scans) in enumerate(sorted(by_image.items())):
            image = Image.open(available[stem]).convert("RGB")
            if image.size != OSIE_CANVAS:
                raise RuntimeError(f"{stem} is {image.size}, expected {OSIE_CANVAS}")
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32)
            image_tensor = torch.tensor(array.transpose(2, 0, 1)[None, ...])
            center = _centerbias(template, height, width)
            center_tensor = torch.tensor(center[None, ...], dtype=torch.float32)
            readout, orig_shape = _backbone(model, image_tensor)
            image_started = time.time()
            n_scored = 0
            for scan in scans:
                x = np.asarray(scan.x, dtype=np.float64)
                y = np.asarray(scan.y, dtype=np.float64)
                for target in range(len(x)):
                    history = _history(x, y, target)
                    if history is None:
                        continue
                    hist_x = torch.tensor(history[0][None, :] * width / OSIE_CANVAS[0], dtype=torch.float32)
                    hist_y = torch.tensor(history[1][None, :] * height / OSIE_CANVAS[1], dtype=torch.float32)
                    density = _density(model, readout, orig_shape, center_tensor, hist_x, hist_y)
                    if agreement is None:
                        official = model(image_tensor, center_tensor, hist_x, hist_y)
                        agreement = float((density - official).abs().max())
                        print(f"cached-versus-official max abs {agreement}", flush=True)
                        if agreement > 1e-4:
                            raise RuntimeError("cached readout does not match DeepGazeIII.forward")
                    pixel_x = int(np.clip(np.rint(x[target] * width / OSIE_CANVAS[0]), 0, width - 1))
                    pixel_y = int(np.clip(np.rint(y[target] * height / OSIE_CANVAS[1]), 0, height - 1))
                    log_model = float(density[0, 0, pixel_y, pixel_x])
                    log_center = float(center[pixel_y, pixel_x])
                    other = locations.get((wrong_stem[stem], int(scan.subject_id), target))
                    wrong_bits = math.nan
                    if other is not None:
                        other_x = int(np.clip(np.rint(other[0] * width / OSIE_CANVAS[0]), 0, width - 1))
                        other_y = int(np.clip(np.rint(other[1] * height / OSIE_CANVAS[1]), 0, height - 1))
                        wrong_bits = (float(density[0, 0, other_y, other_x]) - float(center[other_y, other_x])) / math.log(2.0)
                    rows.append(
                        {
                            "image": stem,
                            "split": osie_split(scan.image_name),
                            "subject": int(scan.subject_id),
                            "target_index": int(target),
                            "log_density_nats": log_model,
                            "centerbias_log_density_nats": log_center,
                            "information_gain_bits": (log_model - log_center) / math.log(2.0),
                            "wrong_image_information_gain_bits": wrong_bits,
                            "history_fixations": 4,
                        }
                    )
                    n_scored += 1
            print(
                f"{stem} scored {n_scored} in {time.time() - image_started:.1f}s image {image_index + 1}/{len(by_image)}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "deepgaze3_osie_fixations.csv", index=False)
    image_means = frame.groupby("image")["information_gain_bits"].mean()
    subject_means = frame.groupby("subject")["information_gain_bits"].mean()
    summary = {
        "deepgaze_commit": DEEPGAZE_COMMIT,
        "weights": str(WEIGHTS),
        "state_dict_missing": len(missing),
        "state_dict_unexpected": len(unexpected),
        "centerbias": "MIT1003 1024 template, nearest-neighbor zoom, logsumexp renormalized",
        "osie_pixels_per_degree_source": "Xu et al., Journal of Vision, 2014, about 24 px/deg in the 800x600 image",
        "training_pixels_per_degree_source": "DeepGaze README: MIT1003 at 35 px/deg",
        "scale_35_over_24": scale,
        "resized_width": width,
        "resized_height": height,
        "resize": "PIL bilinear",
        "image_tensor": "float32, values in 0 to 255",
        "history": "last four fixations strictly before the target, order [-1, -2, -3, -4], indices 0 through 3 required",
        "metric": "mean of (natural log model density minus natural log center-bias density) divided by ln(2), at the rounded resized pixel",
        "not_comparable_to": "uniform-bin information gain in outputs/temporal_v1, outputs/osie_v1, and outputs/protocolshift_b3_v1",
        "cached_versus_official_max_abs": agreement,
        "n_fixations": int(len(frame)),
        "n_images": int(frame["image"].nunique()),
        "n_subjects": int(frame["subject"].nunique()),
        "mean_information_gain_bits": float(frame["information_gain_bits"].mean()),
        "image_mean_bits": float(image_means.mean()),
        "image_sd_bits": float(image_means.std(ddof=1)),
        "subject_mean_bits": float(subject_means.mean()),
        "subject_sd_bits": float(subject_means.std(ddof=1)),
        "mean_log_density_nats": float(frame["log_density_nats"].mean()),
        "mean_centerbias_log_density_nats": float(frame["centerbias_log_density_nats"].mean()),
        "wrong_image_image_mean_bits": float(frame.groupby("image")["wrong_image_information_gain_bits"].mean().mean()),
        "wrong_image_image_sd_bits": float(frame.groupby("image")["wrong_image_information_gain_bits"].mean().std(ddof=1)),
        "wrong_image_n": int(frame["wrong_image_information_gain_bits"].notna().sum()),
        "seconds": time.time() - started,
        "coco_freeview": "not scored; Chen et al. 2022 does not state pixels per degree",
        "mit1003": "not scored; DeepGaze III was trained on MIT1003, so a local split is not a model-independent test",
        "exploratory": "These nine images were already used in earlier inspected analyses. This is not a confirmatory test.",
        "msdb": _msdb_blocker(),
    }
    (OUTPUT / "deepgaze3_osie_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("n_fixations", "image_mean_bits", "image_sd_bits", "subject_mean_bits", "subject_sd_bits", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
