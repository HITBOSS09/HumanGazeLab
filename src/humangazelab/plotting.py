"""Figure writers. Titles name the quantity that was computed."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def plot_density_pair(mean: np.ndarray, sd: np.ndarray, width: float, height: float, path: Path, title: str) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    extent = [0, width, height, 0]
    show_map(axes[0], mean.T, extent, "Subject-mean share")
    show_map(axes[1], sd.T, extent, "SD across subjects")
    fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def plot_early_late(early: np.ndarray, late: np.ndarray, width: float, height: float, path: Path, title: str) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    extent = [0, width, height, 0]
    vmax = float(max(early.max(), late.max()))
    show_map(axes[0], early.T, extent, "Early share", vmax=vmax)
    show_map(axes[1], late.T, extent, "Late share", vmax=vmax)
    fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def show_map(ax, image: np.ndarray, extent, label: str, vmax: float | None = None) -> None:
    field = ax.imshow(image, origin="upper", extent=extent, cmap="magma", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.figure.colorbar(field, ax=ax, fraction=0.046, label=label)


def plot_radius_hist(observed: np.ndarray, uniform: np.ndarray, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    hi = float(np.quantile(observed, 0.995))
    bins = np.linspace(0, hi, 41)
    ax.hist(observed, bins=bins, density=True, histtype="step", linewidth=1.6, label="Post-initial fixations")
    ax.hist(uniform, bins=bins, density=True, histtype="step", linewidth=1.6, label="Uniform draws on the canvas")
    ax.set_xlim(0, hi)
    ax.set_xlabel("Distance to canvas center (px)")
    ax.set_ylabel("Density")
    ax.set_title("Post-initial fixation radius. Axis ends at the observed 99.5 percentile.")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_saccade_hist(real: np.ndarray, shuffled: np.ndarray, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    hi = float(np.quantile(real, 0.99))
    bins = np.linspace(0, hi, 41)
    ax.hist(real, bins=bins, density=True, histtype="step", linewidth=1.6, label="Released order")
    ax.hist(shuffled, bins=bins, density=True, histtype="step", linewidth=1.6, label="One within-scanpath shuffle")
    ax.set_xlim(0, hi)
    ax.set_xlabel("Successive distance (px)")
    ax.set_ylabel("Density")
    ax.set_title("Post-initial successive distance. Axis ends at the released-order 99th percentile.")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_subject_radius(subject_ids: np.ndarray, means: np.ndarray, grand: float, uniform: float, path: Path) -> None:
    _style()
    order = np.argsort(subject_ids)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(subject_ids))
    ax.scatter(means[order], y, color="#222222", zorder=3)
    ax.axvline(grand, color="#0072B2", linewidth=1.4, label="Unweighted mean of subjects")
    ax.axvline(uniform, color="#D55E00", linewidth=1.4, linestyle="--", label="1 px grid mean on the canvas")
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(v)) for v in subject_ids[order]])
    ax.set_xlabel("Scanpath-mean radius, post-initial fixations (px)")
    ax.set_ylabel("Subject id")
    ax.set_title("Subject means. Each scanpath has equal weight inside a subject.")
    ax.legend(frameon=False, loc="best")
    fig.savefig(path)
    plt.close(fig)


def plot_scanpath_grid(panels: list[dict], path: Path) -> None:
    """Each panel dict has image_path, x, y, duration, title."""
    _style()
    n = len(panels)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, panel in zip(axes, panels):
        _draw_scanpath(ax, panel)
    fig.suptitle("Released fixation order on the stimulus. Open red circle: fixation index 0.")
    fig.savefig(path)
    plt.close(fig)


def plot_all_subjects(panel: dict, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = Image.open(panel["image_path"]).convert("RGB")
    width, height = image.size
    ax.imshow(image, extent=[0, width, height, 0])
    for series in panel["series"]:
        ax.plot(series["x"], series["y"], linewidth=1.0, alpha=0.85, label=str(series["subject_id"]))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(panel["title"])
    ax.legend(
        frameon=False,
        ncol=5,
        fontsize=8,
        title="Subject id",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.savefig(path)
    plt.close(fig)


def _draw_scanpath(ax, panel: dict) -> None:
    image = Image.open(panel["image_path"]).convert("RGB")
    width, height = image.size
    x = np.asarray(panel["x"], dtype=float)
    y = np.asarray(panel["y"], dtype=float)
    duration = np.asarray(panel["duration"], dtype=float)
    ax.imshow(image, extent=[0, width, height, 0])
    ax.plot(x, y, color="white", linewidth=1.15, solid_capstyle="round")
    scale = duration / duration.max() if len(duration) and duration.max() > 0 else np.ones_like(duration)
    ax.scatter(x, y, c=np.arange(len(x)), cmap="viridis", s=16 + 40 * scale, zorder=3)
    ax.scatter([x[0]], [y[0]], s=70, facecolors="none", edgecolors="#E41A1C", linewidths=1.4, zorder=4)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(panel["title"], fontsize=8)
    ax.axis("on")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.tick_params(labelsize=7)
