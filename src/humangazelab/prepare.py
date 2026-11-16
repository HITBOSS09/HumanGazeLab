"""Index the official stimulus archives and extract a seeded image subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from humangazelab.adapter import Scanpath, load_scanpaths
from humangazelab.config import AnalysisConfig, load_config
from humangazelab.download import FALLBACK
from humangazelab.imageselect import select_image_subset
from humangazelab.zipremote import RangeReader, ZipEntry, extract_entry, jpeg_index, parse_central_directory


def archive_match_summary(needed: set[str], tp: dict[str, list[ZipEntry]], ta: dict[str, list[ZipEntry]]) -> dict:
    tp_same, tp_bad = _duplicate_crc_counts(tp)
    ta_same, ta_bad = _duplicate_crc_counts(ta)
    both = needed & set(tp) & set(ta)
    crc_equal = 0
    crc_conflict = 0
    for name in both:
        crcs = {entry.crc for entry in tp[name]} | {entry.crc for entry in ta[name]}
        if len(crcs) == 1:
            crc_equal += 1
        else:
            crc_conflict += 1
    return {
        "n_needed": len(needed),
        "tp_jpeg_entries": sum(len(entries) for entries in tp.values()),
        "ta_jpeg_entries": sum(len(entries) for entries in ta.values()),
        "tp_unique_basenames": len(tp),
        "ta_unique_basenames": len(ta),
        "matched_tp_only": len((needed & set(tp)) - set(ta)),
        "matched_ta_only": len((needed & set(ta)) - set(tp)),
        "matched_both": len(both),
        "matched_neither": len(needed - set(tp) - set(ta)),
        "both_crc_equal": crc_equal,
        "both_crc_conflict": crc_conflict,
        "tp_duplicate_basename_same_crc": tp_same,
        "tp_duplicate_basename_crc_conflict": tp_bad,
        "ta_duplicate_basename_same_crc": ta_same,
        "ta_duplicate_basename_crc_conflict": ta_bad,
    }


def choose_entry(name: str, tp: dict[str, list[ZipEntry]], ta: dict[str, list[ZipEntry]], prefer: str) -> dict | None:
    sources: list[tuple[str, list[ZipEntry]]] = []
    if name in tp:
        sources.append(("TP", tp[name]))
    if name in ta:
        sources.append(("TA", ta[name]))
    if not sources:
        return None
    crcs = {entry.crc for _archive, entries in sources for entry in entries}
    paths = [entry.filename for _archive, entries in sources for entry in entries]
    if len(crcs) != 1:
        return {"conflict": True, "paths": paths, "crcs": sorted(crcs)}
    ordered = sources if prefer == "TP" else list(reversed(sources))
    archive, entries = ordered[0]
    entry = sorted(entries, key=lambda item: item.filename)[0]
    return {
        "conflict": False,
        "archive": archive,
        "entry": entry,
        "crc": entry.crc,
        "paths": sorted(paths),
    }


def zero_padding_widths(rgb: np.ndarray) -> dict[str, int]:
    """Count leading and trailing rows and columns whose pixels are all (0, 0, 0)."""
    black = rgb.max(axis=2) == 0
    height, width = black.shape

    def leading(mask: np.ndarray) -> int:
        count = 0
        for row in mask:
            if bool(row.all()):
                count += 1
            else:
                break
        return count

    return {
        "top": leading(black),
        "bottom": leading(black[::-1]),
        "left": leading(black.T),
        "right": leading(black.T[::-1]),
        "height": int(height),
        "width": int(width),
    }


def prepare(
    root: Path,
    config: AnalysisConfig,
    tp_source: str | None = None,
    ta_source: str | None = None,
) -> dict:
    root = Path(root)
    scanpaths = load_scanpaths(root / config.fixation_path)
    needed = {sp.image_name for sp in scanpaths}
    tp_source = tp_source or config.tp_zip_url
    ta_source = ta_source or config.ta_zip_url
    try:
        tp_reader = RangeReader(tp_source)
        ta_reader = RangeReader(ta_source)
        tp_index = jpeg_index(parse_central_directory(tp_reader))
        ta_index = jpeg_index(parse_central_directory(ta_reader))
    except Exception as exc:
        raise RuntimeError(f"could not index stimulus archives\n{FALLBACK}") from exc

    summary = archive_match_summary(needed, tp_index, ta_index)
    image_table = _image_table(scanpaths)
    selected = select_image_subset(
        image_table,
        n_train=config.n_train_images,
        n_valid=config.n_valid_images,
        seed=config.image_seed,
        min_subjects=config.min_subjects,
    )
    image_dir = root / config.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[str, list[Scanpath]] = {}
    for scanpath in scanpaths:
        if scanpath.image_name in set(selected["image_name"]):
            by_image.setdefault(scanpath.image_name, []).append(scanpath)

    rows = []
    readers = {"TP": tp_reader, "TA": ta_reader}
    for record in selected.itertuples(index=False):
        choice = choose_entry(record.image_name, tp_index, ta_index, config.prefer_archive)
        row = {
            "image_name": record.image_name,
            "split": record.split,
            "n_subjects": int(record.n_subjects),
            "selection": record.selection,
        }
        if choice is None or choice["conflict"]:
            row.update({"extract_status": "conflict_or_missing", "archive": "", "zip_path": "", "crc32": ""})
            rows.append(row)
            continue
        raw = extract_entry(readers[choice["archive"]], choice["entry"])
        dest = image_dir / record.image_name
        dest.write_bytes(raw)
        with Image.open(dest) as image:
            width, height = image.size
            rgb = np.asarray(image.convert("RGB"))
        pads = zero_padding_widths(rgb)
        n_post, n_pad = _padding_fixations(by_image[record.image_name], width, height, pads)
        row.update(
            {
                "extract_status": "ok",
                "archive": choice["archive"],
                "zip_path": choice["entry"].filename,
                "crc32": choice["crc"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "width_px": width,
                "height_px": height,
                "matches_readme_canvas": width == config.canvas_width_px and height == config.canvas_height_px,
                "pad_left": pads["left"],
                "pad_right": pads["right"],
                "pad_top": pads["top"],
                "pad_bottom": pads["bottom"],
                "n_post_fixations": n_post,
                "n_post_fixations_in_all_zero_border": n_pad,
                "duplicate_paths": "|".join(choice["paths"]),
            }
        )
        rows.append(row)

    output = root / config.output_dir
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(table_dir / "image_subset_manifest.csv", index=False)
    summary["n_subset_extracted"] = int((manifest["extract_status"] == "ok").sum()) if "extract_status" in manifest else 0
    summary["n_subset_canvas_match"] = int(manifest["matches_readme_canvas"].sum()) if "matches_readme_canvas" in manifest else 0
    summary["image_seed"] = config.image_seed
    summary["n_train_requested"] = config.n_train_images
    summary["n_valid_requested"] = config.n_valid_images
    summary["min_subjects"] = config.min_subjects
    summary["prefer_archive"] = config.prefer_archive
    summary["padding_note"] = (
        "pad_* counts contiguous all-zero rows or columns at the image border. "
        "A photograph can contain real black pixels, so this is a border heuristic, and it is computed only for the extracted subset."
    )
    (output / "image_archive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _image_table(scanpaths: list[Scanpath]) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for scanpath in scanpaths:
        row = rows.setdefault(
            scanpath.image_name,
            {"image_name": scanpath.image_name, "split": scanpath.split, "subjects": set()},
        )
        if row["split"] != scanpath.split:
            raise ValueError(f"{scanpath.image_name} appears in splits {row['split']} and {scanpath.split}")
        row["subjects"].add(scanpath.subject_id)
    return pd.DataFrame(
        [{"image_name": key, "split": value["split"], "n_subjects": len(value["subjects"])} for key, value in rows.items()]
    )


def _padding_fixations(scanpaths: list[Scanpath], width: int, height: int, pads: dict[str, int]) -> tuple[int, int]:
    n_post = 0
    n_pad = 0
    right = width - pads["right"]
    bottom = height - pads["bottom"]
    for scanpath in scanpaths:
        x = scanpath.x[1:]
        y = scanpath.y[1:]
        n_post += int(len(x))
        if len(x) == 0:
            continue
        inside = (x < pads["left"]) | (x >= right) | (y < pads["top"]) | (y >= bottom)
        n_pad += int(np.sum(inside))
    return n_post, n_pad


def _duplicate_crc_counts(index: dict[str, list[ZipEntry]]) -> tuple[int, int]:
    same = 0
    conflict = 0
    for entries in index.values():
        if len(entries) < 2:
            continue
        if len({entry.crc for entry in entries}) == 1:
            same += 1
        else:
            conflict += 1
    return same, conflict


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract a seeded subset of COCO-Search18 stimuli used by COCO-FreeView.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--tp-zip", default=None, help="Local TP zip. Defaults to the official URL.")
    parser.add_argument("--ta-zip", default=None, help="Local TA zip. Defaults to the official URL.")
    args = parser.parse_args(argv)
    summary = prepare(Path(args.root), load_config(args.config), tp_source=args.tp_zip, ta_source=args.ta_zip)
    print(json.dumps(summary, indent=2))
