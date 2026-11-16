"""Inventory the MIT1003 raw-sample archive.

The authors' readme says the database is publicly available with the ICCV 2009
paper. That readme does not contain a license grant. ``extract_fixations``
still refuses. The conversion that was checked against ``checkFixations.m``
lives in ``check_fixations.py`` and calls that file under Octave. It does not
choose among the three pixels-per-degree statements, and it does not read the
reserved final partition.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections import Counter
from pathlib import Path

README_USERS = (
    "CNG", "ajs", "emb", "ems", "ff", "hp", "jcw", "jw", "kae", "krl", "po", "tmj", "tu", "ya", "zb",
)


class FixationExtractionRefused(RuntimeError):
    """Raised when code asks this module to turn raw samples into fixations."""


def extract_fixations(*_args, **_kwargs):
    raise FixationExtractionRefused(
        "MIT1003 fixations are not extracted here. Use the authors' checkFixations.m "
        "and reconcile its 34 px/deg header with the paper's display description and "
        "the DeepGaze README's 35 px/deg note before any conversion."
    )


def assign_partition(image_stem: str) -> str:
    """Frozen rule for a future MIT1003 split. Membership is not a gaze score.

    The first byte of SHA-256(stem) selects the partition. Bytes 0 through 31
    are tune, bytes 224 through 255 are final, and the rest are train. This
    rule was written before any MIT1003 likelihood was computed. It is not an
    approved confirmatory analysis.
    """
    first = hashlib.sha256(image_stem.encode("utf-8")).digest()[0]
    if first < 32:
        return "tune"
    if first >= 224:
        return "final"
    return "train"


def image_stems(path: str | Path) -> list[str]:
    stems: set[str] = set()
    with zipfile.ZipFile(path) as handle:
        for name in handle.namelist():
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == "DATA" and parts[1] in README_USERS and parts[2].endswith(".mat"):
                stems.add(Path(parts[2]).stem)
    return sorted(stems)


def inventory_zip(path: str | Path) -> dict:
    """Count raw per-image files. Does not read gaze coordinates into a result."""
    archive = Path(path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    per_user: Counter[str] = Counter()
    stems: set[str] = set()
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            parts = name.split("/")
            if len(parts) < 3 or parts[0] != "DATA" or not parts[2].endswith(".mat"):
                continue
            if parts[1].startswith(".") or parts[1] in {".svn"}:
                continue
            per_user[parts[1]] += 1
            stems.add(Path(parts[2]).stem)
    missing = sorted(set(README_USERS) - set(per_user))
    unexpected = sorted(set(per_user) - set(README_USERS))
    return {
        "sha256": digest,
        "n_users": len(per_user),
        "files_per_user": dict(sorted(per_user.items())),
        "users_match_readme": not missing and not unexpected and all(count == 1003 for count in per_user.values()),
        "missing_readme_users": missing,
        "unexpected_users": unexpected,
        "n_image_stems": len(stems),
        "fixations_extracted": False,
        "license_file_in_readme": False,
        "authors_state_publicly_available": True,
    }


def partition_stems(stems: list[str]) -> list[dict]:
    rows = []
    for stem in sorted(stems):
        rows.append({"image_stem": stem, "partition": assign_partition(stem), "scored": False, "fixations_extracted": False})
    return rows
