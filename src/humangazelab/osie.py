"""OSIE adapter. Coordinates are taken from the released MATLAB struct.

Subject ids are the position of each record in ``subjects`` (0 through 14).
The file does not contain the original participant codes. The image split is
assigned here by a hash of the image name. It is not an author-provided split.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import scipy.io

from humangazelab.adapter import Scanpath


def osie_split(image_name: str) -> str:
    """Held-out images are hash bucket 4. Buckets 0-3 stay available for fitting and selection."""
    if zlib.crc32(image_name.encode("utf-8")) % 5 == 4:
        return "valid"
    return "train"


def scanpaths_from_fixations(fixations) -> list[Scanpath]:
    scanpaths: list[Scanpath] = []
    source_index = 0
    for record in fixations:
        image_name = str(record.img)
        subjects = np.atleast_1d(record.subjects)
        for subject_index, subject in enumerate(subjects):
            x = np.atleast_1d(np.asarray(subject.fix_x, dtype=float)).ravel()
            y = np.atleast_1d(np.asarray(subject.fix_y, dtype=float)).ravel()
            duration = np.atleast_1d(np.asarray(subject.fix_duration, dtype=float)).ravel()
            if not (x.size == y.size == duration.size):
                raise ValueError(f"OSIE fixation arrays differ in length for {image_name}")
            if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
                raise ValueError(f"non-finite OSIE coordinate for {image_name}")
            scanpaths.append(
                Scanpath(
                    source_index=source_index,
                    image_name=image_name,
                    subject_id=int(subject_index),
                    split=osie_split(image_name),
                    condition="freeview",
                    x=x,
                    y=y,
                    duration=duration,
                    fix_on_target=None,
                    correct=None,
                )
            )
            source_index += 1
    return scanpaths


def load_osie_scanpaths(path: str | Path) -> list[Scanpath]:
    payload = scipy.io.loadmat(Path(path), squeeze_me=True, struct_as_record=False)
    if "fixations" not in payload:
        raise ValueError("OSIE file has no 'fixations' field")
    return scanpaths_from_fixations(payload["fixations"])
