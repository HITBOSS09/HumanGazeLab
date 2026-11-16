"""Check that a stimulus file and the released fixations share one frame.

This module does not download images and does not invent missing files.
Canvas-fraction coordinates are not degrees of visual angle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: str | Path) -> np.ndarray:
    """Return a float image in [0, 1], shape (height, width)."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def align_fixations(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray, expected_height: int, expected_width: int) -> dict:
    """Require the file size and every fixation to match the released canvas.

    y increases downward, in the same direction used for the OSIE overlay.
    """
    height, width = gray.shape
    if (height, width) != (expected_height, expected_width):
        raise ValueError(f"image is {height} by {width}, expected {expected_height} by {expected_width}")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError("x and y lengths differ")
    if xs.size == 0:
        raise ValueError("no fixations were supplied")
    if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
        raise ValueError("a fixation coordinate is not finite")
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.all(inside):
        raise ValueError(f"{int((~inside).sum())} fixations fall outside the image")
    return {
        "height": height,
        "width": width,
        "n_fixations": int(xs.size),
        "all_inside": True,
        "visual_angle_aligned": False,
    }
