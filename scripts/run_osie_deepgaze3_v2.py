"""Score frozen DeepGaze III on every eligible OSIE image.

The checkpoint, 35/24 scale, four-fixation history, center-bias template, and
information-gain definition are the ones already used on nine images. Nothing
is tuned on the new scores. Backbone features are cached per image. Scanpath
maps are computed with the same arithmetic as encode_scanpath_features, in
bounded batches. Prediction values are checked against DeepGazeIII.forward.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp

from humangazelab.osie import load_osie_scanpaths, osie_split
from humangazelab.osie_deepgaze import (
    paired_targets,
    previously_scored_stems,
    scaled_coordinate,
    scaled_pixel,
    summarize_paired,
)
from humangazelab.protocolshift import FROZEN_OUTPUTS, assert_new_output

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "osie_deepgaze3_v2.yaml"
DEEPGAZE = Path("/tmp/DeepGaze")
WEIGHTS = Path("/tmp/deepgaze-weights/deepgaze3.pth")
CENTERBIAS = Path("/tmp/deepgaze-weights/centerbias_mit1003.npy")
BATCH = 16
EXTRA_FROZEN = (
    "outputs/baseline_v1",
    "outputs/protocolshift_b3_v1",
    "outputs/protocolshift_prep_v1",
)


def _refuse(output: Path) -> None:
    assert_new_output(ROOT, output)
    for relative in (*FROZEN_OUTPUTS, *EXTRA_FROZEN):
        if output.resolve() == (ROOT / relative).resolve():
            raise RuntimeError(f"refusing to write into {relative}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _nearest_source(height: int, width: int, readout_shape: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Source coordinates selected by the model's default nearest interpolation.

    encode_scanpath_features builds full-resolution maps and DeepGazeIII
    downsamples them with F.interpolate's default nearest mode. These two maps
    are that sampling, computed once because every OSIE image uses the same
    resized canvas.
    """
    probe_x = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width).expand(1, 1, height, width)
    probe_y = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1).expand(1, 1, height, width)
    return F.interpolate(probe_x, readout_shape), F.interpolate(probe_y, readout_shape)


def _load_model():
    sys.path.insert(0, str(DEEPGAZE))
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

    model = DeepGazeIII(pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    head_missing = [key for key in missing if not key.startswith("features.")]
    if head_missing or unexpected:
        raise RuntimeError(f"head load failed: {len(head_missing)} missing, {len(unexpected)} unexpected")
    model.eval()
    return model, len(missing), len(unexpected)


def _backbone(model, image_tensor: torch.Tensor):
    orig = image_tensor.shape
    resized = F.interpolate(image_tensor, scale_factor=1 / model.downsample, recompute_scale_factor=False)
    features = model.features(resized)
    readout_shape = [
        math.ceil(orig[2] / model.downsample / model.readout_factor),
        math.ceil(orig[3] / model.downsample / model.readout_factor),
    ]
    features = [F.interpolate(item, readout_shape) for item in features]
    return torch.cat(features, dim=1), orig, readout_shape


def _density_batch(model, readout_shape, centerbias, x_hist, y_hist, saliency_cache, src_x, src_y):
    dx = src_x - x_hist[:, :, None, None]
    dy = src_y - y_hist[:, :, None, None]
    scan = torch.cat((dx, dy, torch.sqrt(dx.square() + dy.square())), dim=1)
    batch = scan.shape[0]
    center = centerbias.expand(batch, -1, -1)
    predictions = []
    for index, (scan_net, selection, finalizer) in enumerate(
        zip(model.scanpath_networks, model.fixation_selection_networks, model.finalizers)
    ):
        saliency = saliency_cache[index].expand(batch, -1, -1, -1)
        selected = selection((saliency, scan_net(scan)))
        predictions.append(finalizer(selected, center)[:, None])
    stacked = torch.cat(predictions, dim=1) - math.log(len(model.saliency_networks))
    return stacked.logsumexp(dim=1, keepdim=True)


def _stimulus_manifest(image_dir: Path, output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(image_dir.glob("*.jpg")):
        with Image.open(path) as image:
            size = image.size
        rows.append({"image": path.stem, "sha256": _sha256(path), "width": size[0], "height": size[1]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "stimulus_manifest.csv", index=False)
    if not frame["width"].eq(800).all() or not frame["height"].eq(600).all():
        raise RuntimeError("an OSIE JPEG is not 800 by 600")
    return frame


def _score_image(model, stem, rows, image_path, width, height, center, center_tensor, src_x, src_y, checked: dict) -> list[dict]:
    image = Image.open(image_path).convert("RGB")
    if image.size != (800, 600):
        raise RuntimeError(f"{stem} is {image.size}")
    resized = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    image_tensor = torch.tensor(resized.transpose(2, 0, 1)[None, ...])
    readout, orig_shape, readout_shape = _backbone(model, image_tensor)
    saliency_cache = [network(readout) for network in model.saliency_networks]
    scored = []
    order = np.array([-1, -2, -3, -4])
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start : start + BATCH]
            hist_x = []
            hist_y = []
            for row in chunk:
                hist_x.append(scaled_coordinate(row["history_x"][order], 800, width))
                hist_y.append(scaled_coordinate(row["history_y"][order], 600, height))
            x_hist = torch.tensor(np.stack(hist_x), dtype=torch.float32)
            y_hist = torch.tensor(np.stack(hist_y), dtype=torch.float32)
            density = _density_batch(
                model, readout_shape, center_tensor, x_hist, y_hist, saliency_cache, src_x, src_y
            )
            if not checked["done"]:
                official = model(image_tensor, center_tensor, x_hist[:1], y_hist[:1])
                gap = float((density[:1] - official).abs().max())
                checked["gap"] = gap
                checked["done"] = True
                print(f"cached-versus-official max abs {gap}", flush=True)
                if gap > 1e-4:
                    raise RuntimeError("batched maps do not match DeepGazeIII.forward")
            density_np = density[:, 0].numpy()
            for row_index, row in enumerate(chunk):
                pixel_x = scaled_pixel(row["x"], 800, width)
                pixel_y = scaled_pixel(row["y"], 600, height)
                other_x = scaled_pixel(row["partner_x"], 800, width)
                other_y = scaled_pixel(row["partner_y"], 600, height)
                log_model = float(density_np[row_index, pixel_y, pixel_x])
                log_wrong = float(density_np[row_index, other_y, other_x])
                log_center = float(center[pixel_y, pixel_x])
                log_center_wrong = float(center[other_y, other_x])
                true_bits = (log_model - log_center) / math.log(2.0)
                wrong_bits = (log_wrong - log_center_wrong) / math.log(2.0)
                scored.append(
                    {
                        "image": stem,
                        "split": row["split"],
                        "previously_scored": row["previously_scored"],
                        "subject": row["subject"],
                        "target_index": row["target_index"],
                        "partner_image": row["partner_image"],
                        "log_density_nats": log_model,
                        "centerbias_log_density_nats": log_center,
                        "information_gain_bits": true_bits,
                        "wrong_image_information_gain_bits": wrong_bits,
                        "paired_difference_bits": true_bits - wrong_bits,
                    }
                )
    return scored


def _attach_history(scanpaths, targets: list[dict]) -> list[dict]:
    lookup = {(str(scan.image_name), int(scan.subject_id)): scan for scan in scanpaths}
    attached = []
    for row in targets:
        scan = lookup[(row["image_name"], row["subject"])]
        index = row["target_index"]
        attached.append(
            {
                **row,
                "split": osie_split(row["image_name"]),
                "history_x": np.asarray(scan.x[:index], dtype=np.float64),
                "history_y": np.asarray(scan.y[:index], dtype=np.float64),
            }
        )
    return attached


def _figures(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    expansion = frame.loc[~frame["previously_scored"].astype(bool)]
    image_means = expansion.groupby("image")[["information_gain_bits", "wrong_image_information_gain_bits", "paired_difference_bits"]].mean()
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].hist(image_means["paired_difference_bits"], bins=30, color="#1f4e79")
    axes[0].axvline(float(image_means["paired_difference_bits"].mean()), color="#b85c38", linewidth=1.2)
    axes[0].set_xlabel("Image-mean paired difference (bits)")
    axes[0].set_ylabel("Images")
    axes[0].set_title("True image minus wrong image")
    axes[1].errorbar(
        [0, 1],
        [image_means["information_gain_bits"].mean(), image_means["wrong_image_information_gain_bits"].mean()],
        yerr=[image_means["information_gain_bits"].std(ddof=1), image_means["wrong_image_information_gain_bits"].std(ddof=1)],
        fmt="o",
        color="#1f4e79",
        capsize=4,
    )
    axes[1].axhline(0.0, color="#666666", linewidth=0.8)
    axes[1].set_xticks([0, 1], ["True image", "Wrong image"])
    axes[1].set_ylabel("Image-mean information gain (bits)")
    axes[1].set_title("Expansion images, image SD")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    _refuse(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    image_dir = ROOT / config["stimuli"]
    manifest = _stimulus_manifest(image_dir, output)
    print(f"stimuli {len(manifest)} sha256 written", flush=True)
    scanpaths = load_osie_scanpaths(ROOT / config["fixations"])
    targets, counts = paired_targets(scanpaths)
    prior = set(previously_scored_stems(ROOT / config["previously_scored_output"]))
    if len(prior) != 9:
        raise RuntimeError(f"expected 9 previously scored images, found {len(prior)}")
    for row in targets:
        row["previously_scored"] = row["image"] in prior
    targets = _attach_history(scanpaths, targets)
    by_image: dict[str, list] = {}
    for row in targets:
        by_image.setdefault(row["image"], []).append(row)
    missing_files = sorted(set(by_image) - set(manifest["image"]))
    if missing_files:
        raise RuntimeError(f"{len(missing_files)} scored images have no JPEG")
    (output / "eligibility.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)

    done_path = output / "completed_images.txt"
    done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
    rows_path = output / "deepgaze3_osie_fixations.csv"
    model, missing_keys, unexpected_keys = _load_model()
    template = np.load(CENTERBIAS)
    scale = 35.0 / 24.0
    width = int(round(800 * scale))
    height = int(round(600 * scale))
    center = _centerbias(template, height, width)
    center_tensor = torch.tensor(center[None, ...], dtype=torch.float32)
    readout_shape = [
        math.ceil(height / model.downsample / model.readout_factor),
        math.ceil(width / model.downsample / model.readout_factor),
    ]
    src_x, src_y = _nearest_source(height, width, readout_shape)
    checked = {"done": False, "gap": None}
    started = time.time()
    image_times = []
    timing_path = output / "image_seconds.csv"
    if not timing_path.exists():
        timing_path.write_text("image,n_fixations,seconds\n", encoding="utf-8")
    pending = [stem for stem in sorted(by_image) if stem not in done]
    limit = int(os.environ.get("OSIE_DG3_LIMIT", "0"))
    if limit:
        pending = pending[:limit]
    print(f"pending {len(pending)} done {len(done)}", flush=True)
    for image_index, stem in enumerate(pending, start=1):
        image_started = time.time()
        scored = _score_image(
            model,
            stem,
            by_image[stem],
            image_dir / f"{stem}.jpg",
            width,
            height,
            center,
            center_tensor,
            src_x,
            src_y,
            checked,
        )
        block = pd.DataFrame(scored)
        block.to_csv(rows_path, mode="a", header=not rows_path.exists(), index=False)
        with done_path.open("a", encoding="utf-8") as handle:
            handle.write(stem + "\n")
        elapsed = time.time() - image_started
        image_times.append(elapsed)
        with timing_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stem},{len(scored)},{elapsed:.3f}\n")
        print(f"{stem} scored {len(scored)} in {elapsed:.1f}s {image_index}/{len(pending)}", flush=True)

    frame = pd.read_csv(rows_path)
    if frame.duplicated(["image", "subject", "target_index"]).any():
        raise RuntimeError("duplicate scored targets")
    if limit:
        print(f"limit {limit}; full summary skipped", flush=True)
        return
    prior_frame = pd.read_csv(ROOT / config["previously_scored_output"])
    merged = prior_frame.merge(
        frame,
        on=["image", "subject", "target_index"],
        suffixes=("_previous", "_new"),
    )
    if len(merged) != len(prior_frame):
        raise RuntimeError(f"reproduction matched {len(merged)} of {len(prior_frame)} previous rows")
    reproduction_gap = float((merged["information_gain_bits_previous"] - merged["information_gain_bits_new"]).abs().max())
    print(f"reproduction max abs information gain {reproduction_gap}", flush=True)
    if reproduction_gap > 1e-3:
        raise RuntimeError("the nine-image information gains moved")

    n_boot = int(config["uncertainty"]["bootstrap_draws"])
    seed = int(config["uncertainty"]["bootstrap_seed"])
    expansion = frame.loc[~frame["previously_scored"].astype(bool)].copy()
    summaries = summarize_paired(expansion, n_boot, seed, "expansion_excluding_nine")
    summaries.extend(summarize_paired(frame, n_boot, seed, "all_eligible_including_nine"))
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "summary.csv", index=False)
    value_columns = ["information_gain_bits", "wrong_image_information_gain_bits", "paired_difference_bits"]
    image_means = frame.groupby(["image", "previously_scored"], sort=True)[value_columns].mean().reset_index()
    image_means.to_csv(output / "image_means.csv", index=False)
    subject_means = frame.groupby("subject", sort=True)[value_columns].mean().reset_index()
    subject_means.to_csv(output / "subject_means.csv", index=False)
    expansion.groupby("subject", sort=True)[value_columns].mean().reset_index().to_csv(
        output / "subject_means_expansion.csv", index=False
    )
    _figures(frame, output / "figures" / "paired_difference.png")
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "exploratory": True,
        "reason": "The nine-image outcome, the 35/24 scale, the center-bias template, and this metric were already chosen.",
        "config": str(CONFIG.relative_to(ROOT)),
        "deepgaze_commit": config["checkpoint"]["commit"],
        "weights_sha256": _sha256(WEIGHTS),
        "centerbias_sha256": _sha256(CENTERBIAS),
        "state_dict_missing": missing_keys,
        "state_dict_unexpected": unexpected_keys,
        "cached_versus_official_max_abs": checked["gap"],
        "reproduction_max_abs_information_gain": reproduction_gap,
        "n_stimuli": int(len(manifest)),
        "stimulus_sha256_file": "stimulus_manifest.csv",
        "eligibility": counts,
        "eligibility_note": "n_index_0 is included in n_index_below_4. Eligible before pairing is n_fixations minus n_index_below_4 minus n_outside_canvas.",
        "n_scored_rows": int(len(frame)),
        "n_expansion_images": int(expansion["image"].nunique()),
        "n_expansion_fixations": int(len(expansion)),
        "n_previously_scored_images": 9,
        "cuda": bool(torch.cuda.is_available()),
        "gpu_memory_bytes": None,
        "peak_rss_kb": int(peak_kb),
        "batch_size": BATCH,
        "seconds_this_process": time.time() - started,
        "seconds_per_image_mean": float(pd.read_csv(timing_path)["seconds"].mean()),
        "n_images_timed": int(len(pd.read_csv(timing_path))),
        "frozen_b3": "uniform-bin information gain in outputs/osie_v1 and outputs/protocolshift_b3_v1; a different probability reference",
        "training_separation": "DeepGaze III readout was trained on MIT1003. OSIE is not named as a DeepGaze III training set. The DenseNet-201 backbone is the torchvision ImageNet-1K checkpoint. Overlap between these OSIE photographs and ImageNet was not tested.",
        "backbone_overlap": "unverified",
    }
    (output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_scored_rows", "reproduction_max_abs_information_gain", "seconds_this_process", "peak_rss_kb")}, indent=2))


if __name__ == "__main__":
    main()
