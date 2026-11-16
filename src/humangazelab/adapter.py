"""Load COCO-FreeView scanpaths into the columns this file actually contains."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Scanpath:
    source_index: int
    image_name: str
    subject_id: int
    split: str
    condition: str
    x: np.ndarray
    y: np.ndarray
    duration: np.ndarray
    fix_on_target: bool | None
    correct: int | None


_REQUIRED = ("name", "subject", "split", "X", "Y", "T", "length")


def load_scanpaths(path: str | Path) -> list[Scanpath]:
    """Load the released JSON list.

    Array position is kept as fixation order. The official readme says ``X``,
    ``Y``, and ``T`` run from the first fixation to the last. ``T`` is stored
    as released and is not rescaled here.
    """
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list of scanpaths, found {type(payload).__name__}")

    scanpaths: list[Scanpath] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is {type(record).__name__}, expected an object")
        missing = [key for key in _REQUIRED if key not in record]
        if missing:
            raise ValueError(f"record {index} is missing {missing}")
        x = _vector(record["X"], index, "X")
        y = _vector(record["Y"], index, "Y")
        duration = _vector(record["T"], index, "T")
        if not (len(x) == len(y) == len(duration)):
            raise ValueError(
                f"record {index} has lengths X={len(x)}, Y={len(y)}, T={len(duration)}"
            )
        length = int(record["length"])
        if length != len(x):
            raise ValueError(f"record {index} length field is {length} but X has {len(x)} values")
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(duration).all()):
            raise ValueError(f"record {index} contains a non-finite coordinate or duration")
        fix_on_target = record.get("fixOnTarget")
        correct = record.get("correct")
        scanpaths.append(
            Scanpath(
                source_index=index,
                image_name=str(record["name"]),
                subject_id=int(record["subject"]),
                split=str(record["split"]),
                condition=str(record["condition"]) if "condition" in record else "",
                x=x,
                y=y,
                duration=duration,
                fix_on_target=bool(fix_on_target) if isinstance(fix_on_target, bool) else None,
                correct=int(correct) if correct is not None and not isinstance(correct, bool) else None,
            )
        )
    return scanpaths


def _vector(values, index: int, field: str) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError(f"record {index} field {field} is {type(values).__name__}, expected a list")
    return np.asarray(values, dtype=np.float64)
