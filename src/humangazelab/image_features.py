"""Low-level window features and a radial center-distance control.

The control asks whether a fixation-versus-random difference remains after
the comparison pixel is placed at the same distance from the canvas center.
It does not remove the tendency of objects to lie toward the middle of a scene,
and it is not a fitted gaze model.
"""

from __future__ import annotations

import numpy as np


def gradient_energy(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(gray, dtype=np.float64))
    return np.hypot(gx, gy)


def window_features(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: int) -> dict[str, np.ndarray]:
    """Mean luminance, gradient energy, and variance inside a square window."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    gray = np.asarray(gray, dtype=np.float64)
    energy = gradient_energy(gray)
    height, width = gray.shape
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    luminance = np.full(xs.shape, np.nan)
    gradient = np.full(xs.shape, np.nan)
    variance = np.full(xs.shape, np.nan)
    for index, (x_value, y_value) in enumerate(zip(xs, ys)):
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        x_int = int(np.clip(round(float(x_value)), 0, width - 1))
        y_int = int(np.clip(round(float(y_value)), 0, height - 1))
        x0, x1 = max(0, x_int - radius), min(width, x_int + radius + 1)
        y0, y1 = max(0, y_int - radius), min(height, y_int + radius + 1)
        patch = gray[y0:y1, x0:x1]
        luminance[index] = float(patch.mean())
        gradient[index] = float(energy[y0:y1, x0:x1].mean())
        variance[index] = float(patch.var())
    return {"luminance": luminance, "gradient": gradient, "variance": variance}


def directional_energy(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute neighbor differences along four pixel directions.

    ``horizontal`` is the left-right difference. It responds to vertical structure.
    ``vertical`` is the up-down difference. The two diagonal maps use the
    lower-right and upper-right neighbors. Edge rows and columns that have no
    neighbor are zero. These are not Gabor filters, and the window is in pixels.
    """
    values = np.asarray(gray, dtype=np.float64)
    horizontal = np.zeros_like(values)
    vertical = np.zeros_like(values)
    diagonal_down = np.zeros_like(values)
    diagonal_up = np.zeros_like(values)
    horizontal[:, 1:] = np.abs(np.diff(values, axis=1))
    vertical[1:, :] = np.abs(np.diff(values, axis=0))
    diagonal_down[1:, 1:] = np.abs(values[1:, 1:] - values[:-1, :-1])
    diagonal_up[:-1, 1:] = np.abs(values[:-1, 1:] - values[1:, :-1])
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "diagonal_down": diagonal_down,
        "diagonal_up": diagonal_up,
    }


def window_mean(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: int) -> np.ndarray:
    """Mean of one map inside the same square window used for luminance."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    field = np.asarray(field, dtype=np.float64)
    height, width = field.shape
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    out = np.full(xs.shape, np.nan)
    for index, (x_value, y_value) in enumerate(zip(xs, ys)):
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        x_int = int(np.clip(round(float(x_value)), 0, width - 1))
        y_int = int(np.clip(round(float(y_value)), 0, height - 1))
        x0, x1 = max(0, x_int - radius), min(width, x_int + radius + 1)
        y0, y1 = max(0, y_int - radius), min(height, y_int + radius + 1)
        out[index] = float(field[y0:y1, x0:x1].mean())
    return out


def center_antipode(xs: np.ndarray, ys: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Point opposite a fixation through the canvas center. The radius is unchanged."""
    return width - np.asarray(xs, dtype=float), height - np.asarray(ys, dtype=float)


def canvas_radius(xs: np.ndarray, ys: np.ndarray, width: float, height: float) -> np.ndarray:
    """Distance from the canvas center used in the descriptive report, (width/2, height/2)."""
    return np.hypot(np.asarray(xs, dtype=float) - width / 2.0, np.asarray(ys, dtype=float) - height / 2.0)


def radius_matched_points(xs: np.ndarray, ys: np.ndarray, width: float, height: float, rng: np.random.Generator, margin: int) -> tuple[np.ndarray, np.ndarray]:
    """Place one point on the same circle about the canvas center, inside the frame.

    A point that cannot be placed after repeated draws is left non-finite.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    radii = canvas_radius(xs, ys, width, height)
    out_x = np.full(xs.shape, np.nan)
    out_y = np.full(ys.shape, np.nan)
    for index, radius in enumerate(radii):
        if not np.isfinite(radius):
            continue
        for _ in range(80):
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            candidate_x = width / 2.0 + radius * np.cos(theta)
            candidate_y = height / 2.0 + radius * np.sin(theta)
            if margin <= candidate_x < width - margin and margin <= candidate_y < height - margin:
                out_x[index] = candidate_x
                out_y[index] = candidate_y
                break
    return out_x, out_y
