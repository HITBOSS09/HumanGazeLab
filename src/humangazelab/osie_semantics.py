"""Official OSIE object masks and semantic attributes.

attrs.mat is MATLAB v7.3. Each image has a list of binary masks and a
12-value attribute vector. The vector order is the order stored in
attrNames, which is not the order in the comment inside computeSemanticMaps.m.
A positive value means the attribute is present. Face and text use values
above 1; those values are kept in the manifest and collapsed with >0 for the
binary region, matching the released Matlab feature maps.

Masks overlap. A pixel may belong to more than one object and more than one
semantic attribute. Background is the complement of the object union. The
2014 paper did not treat that complement as proof that no object was there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
VERIFIED_ATTR_NAMES = (
    "text",
    "face",
    "emotion",
    "sound",
    "smell",
    "taste",
    "touch",
    "motion",
    "operability",
    "watchability",
    "touched",
    "gazed",
)
PRIMARY_LABELS = ("object_union", "face", "text")
COVERAGE_MIN_IMAGES = 100


def _chars(array) -> str:
    return "".join(chr(int(value)) for value in np.asarray(array).reshape(-1))


def mask_from_hdf5(array) -> np.ndarray:
    """Return a boolean mask in original-image coordinates, shape (600, 800).

    h5py reads a MATLAB 600 by 800 mask as shape (800, 600). The transpose
    puts row 0 at the top of the JPEG. Face masks in that orientation contain
    far more fixations than a horizontal or vertical flip.
    """
    stored = np.asarray(array)
    if stored.shape != (CANVAS_WIDTH, CANVAS_HEIGHT):
        raise ValueError(f"mask shape {stored.shape}, expected {(CANVAS_WIDTH, CANVAS_HEIGHT)} before transpose")
    mask = stored.T.astype(bool)
    if mask.shape != (CANVAS_HEIGHT, CANVAS_WIDTH):
        raise ValueError(f"oriented mask shape {mask.shape}")
    return mask


@dataclass(frozen=True)
class ObjectAnnotation:
    index: int
    features: np.ndarray
    mask: np.ndarray

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def positive_labels(self, names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name, value in zip(names, self.features) if float(value) > 0)


class OsieAnnotations:
    """Random-access reader for data/attrs.mat. The file is not copied into git."""

    def __init__(self, path: str | Path):
        import h5py

        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        names = tuple(_chars(self._file[ref]) for ref in self._file["attrNames"][()].reshape(-1))
        if names != VERIFIED_ATTR_NAMES:
            raise ValueError(f"attrNames {names} does not match the verified order")
        self.names = names
        self._by_stem: dict[str, object] = {}
        for ref in self._file["attrs"][()].reshape(-1):
            group = self._file[ref]
            stem = Path(_chars(group["img"][()])).stem
            if stem in self._by_stem:
                raise ValueError(f"duplicate annotation for {stem}")
            self._by_stem[stem] = ref

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "OsieAnnotations":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_stem))

    def objects(self, stem: str) -> list[ObjectAnnotation]:
        group = self._file[self._by_stem[stem]]
        objects = []
        for index, ref in enumerate(group["objs"][()].reshape(-1)):
            obj = self._file[ref]
            features = np.asarray(obj["features"][()], dtype=np.float64).reshape(-1)
            if features.shape != (len(self.names),):
                raise ValueError(f"{stem} object {index} has {features.shape[0]} features")
            values = set(np.unique(np.asarray(obj["map"][()])).tolist())
            if not values.issubset({0, 1}):
                raise ValueError(f"{stem} object {index} is not a binary mask")
            objects.append(ObjectAnnotation(index, features, mask_from_hdf5(obj["map"][()])))
        return objects

    def label_masks(self, stem: str) -> dict[str, np.ndarray]:
        """Union mask for every attribute, plus the union of all objects.

        Attributes are not a partition. Object union is not one of the twelve names.
        """
        objects = self.objects(stem)
        labels = {name: np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool) for name in self.names}
        union = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=bool)
        for obj in objects:
            union |= obj.mask
            for name, value in zip(self.names, obj.features):
                if float(value) > 0:
                    labels[name] |= obj.mask
        labels["object_union"] = union
        return labels


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def overlap_pixels(masks: list[np.ndarray]) -> int:
    if not masks:
        return 0
    count = np.zeros(masks[0].shape, dtype=np.uint16)
    for mask in masks:
        count += mask.astype(np.uint16)
    return int((count > 1).sum())


def manifest_frame(annotations: OsieAnnotations) -> pd.DataFrame:
    """One row per object. Built from the masks, not from model scores."""
    rows = []
    for stem in annotations.stems:
        objects = annotations.objects(stem)
        overlap = overlap_pixels([obj.mask for obj in objects])
        for obj in objects:
            labels = obj.positive_labels(annotations.names)
            rows.append(
                {
                    "image": stem,
                    "object_index": obj.index,
                    "area_px": obj.area,
                    "touches_border": touches_border(obj.mask),
                    "n_positive_labels": len(labels),
                    "positive_labels": ",".join(labels),
                    "image_overlap_px": overlap,
                    "image_n_objects": len(objects),
                    **{f"raw_{name}": float(value) for name, value in zip(annotations.names, obj.features)},
                }
            )
    return pd.DataFrame(rows)


def select_alignment_cases(manifest: pd.DataFrame) -> pd.DataFrame:
    """Geometry-only examples. The rule does not read a model score."""
    if manifest.empty:
        raise ValueError("empty annotation manifest")
    picks = [
        ("smallest_object", _extreme(manifest, "area_px", False)),
        ("largest_object", _extreme(manifest, "area_px", True)),
        ("smallest_face", _extreme(manifest[manifest["positive_labels"].str.contains("face")], "area_px", False)),
        ("largest_text", _extreme(manifest[manifest["positive_labels"].str.contains("text")], "area_px", True)),
        ("smallest_border_object", _extreme(manifest[manifest["touches_border"]], "area_px", False)),
        ("most_overlap", _extreme(manifest, "image_overlap_px", True)),
    ]
    rows = []
    for reason, row in picks:
        rows.append(
            {
                "reason": reason,
                "image": row["image"],
                "object_index": int(row["object_index"]),
                "area_px": int(row["area_px"]),
                "image_overlap_px": int(row["image_overlap_px"]),
            }
        )
    return pd.DataFrame(rows)


def _extreme(frame: pd.DataFrame, column: str, largest: bool) -> pd.Series:
    if frame.empty:
        raise ValueError(f"no rows to select {column}")
    ordered = frame.sort_values([column, "image", "object_index"], ascending=[not largest, True, True])
    return ordered.iloc[0]


def stage_a_stems(manifest: pd.DataFrame) -> tuple[str, ...]:
    """Fifteen stride images plus the geometry cases and the reproduction image.

    The stride is every 50th sorted stem. It was fixed from the file names and
    the mask table, before any DeepGaze region score.
    """
    stems = sorted(manifest["image"].astype(str).unique())
    stride = tuple(stems[::50])
    cases = tuple(select_alignment_cases(manifest)["image"].astype(str))
    return tuple(sorted(set(stride) | set(cases) | {"1001"}))
