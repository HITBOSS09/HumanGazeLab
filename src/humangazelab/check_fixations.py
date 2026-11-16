"""Run the authors' MIT1003 fixation detector. Do not reimplement it.

``checkFixations.m`` (Barbara Hidalgo-Sotelo) is the procedure named by the
database readme. This module writes the raw samples and calls that file under
Octave. It does not replace ``extract_fixations``, which still refuses, so an
unvalidated path cannot silently emit coordinates.

The reserved final partition is rejected before a sample file is opened.
Negative tracker values are left as they are: the m-file looks for NaNs, and
the readme describes negatives as blinks. Those two statements are not the
same operation.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from humangazelab.mit1003 import assign_partition

COMPAT = Path(__file__).resolve().parent / "octave_compat"
OFFICIAL_SHA256 = "c3721a971a754a149109e751d4739ef0e989983961a82b1a70d0b91454545664"
OFFICIAL_URL = (
    "https://people.csail.mit.edu/tjudd/WherePeopleLook/Code/DatabaseCode/checkFixations.m"
)


class FinalPartitionSealed(RuntimeError):
    """Raised when a caller asks to convert a reserved final-partition image."""


class CheckFixationsUnavailable(RuntimeError):
    """Raised when Octave or the authors' m-file cannot produce a fixation list."""


def assert_open_partition(image_stem: str) -> str:
    """Return train or tune. Never open the reserved final partition."""
    label = assign_partition(image_stem)
    if label == "final":
        raise FinalPartitionSealed(
            f"{image_stem} is in the reserved final partition. Its samples are not read."
        )
    if label not in {"train", "tune"}:
        raise FinalPartitionSealed(f"unrecognized partition {label} for {image_stem}")
    return label


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_official_mfile(path: str | Path) -> Path:
    """Accept only the downloaded m-file whose bytes match the copy that was run."""
    mfile = Path(path)
    if not mfile.is_file():
        raise CheckFixationsUnavailable(
            f"checkFixations.m is not at {mfile}. Download {OFFICIAL_URL} and keep it outside the repository."
        )
    found = file_sha256(mfile)
    if found != OFFICIAL_SHA256:
        raise CheckFixationsUnavailable(
            f"checkFixations.m sha256 is {found}, expected {OFFICIAL_SHA256}"
        )
    return mfile


def official_check_fixations(
    eye: np.ndarray,
    mfile: str | Path,
    octave: str = "octave",
) -> list[dict[str, float]]:
    """Return median x, median y, and duration in milliseconds for one trial.

    The list is empty only when the m-file returns no fixation struct. An
    Octave error is raised and is not turned into coordinates.
    """
    samples = np.asarray(eye, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] < 2:
        raise ValueError("eye samples must be an N by 2 array with N >= 2")
    if not np.isfinite(samples).all() and np.isnan(samples).any():
        # NaNs are part of the m-file's own missing-data path. Inf is not.
        if not np.isfinite(np.nan_to_num(samples, nan=0.0)).all():
            raise ValueError("eye samples contain a non-finite value other than NaN")
    checked = require_official_mfile(mfile)
    if shutil.which(octave) is None:
        raise CheckFixationsUnavailable(f"{octave} is not on PATH")

    with tempfile.TemporaryDirectory(prefix="checkfix-") as directory:
        directory_path = Path(directory)
        csv_path = directory_path / "eye.csv"
        np.savetxt(csv_path, samples, delimiter=",")
        script = "\n".join(
            [
                "pkg load statistics;",
                f"addpath('{COMPAT.as_posix()}', '-begin');",
                f"addpath('{checked.parent.as_posix()}', '-begin');",
                "warning('off', 'all');",
                f"d = dlmread('{csv_path.as_posix()}');",
                "try",
                "  [~, Fix, ~] = checkFixations(d);",
                "  if ~isfield(Fix, 'duration')",
                "    printf('EMPTY\\n');",
                "  else",
                "    for k = 1:numel(Fix.duration)",
                "      printf('FIX %.10f %.10f %.10f\\n', Fix.medianXY(k, 1), Fix.medianXY(k, 2), Fix.duration(k));",
                "    end",
                "  end",
                "catch err",
                "  printf('ERROR %s\\n', err.message);",
                "end",
            ]
        )
        completed = subprocess.run(
            [octave, "--no-gui", "--quiet", "--eval", script],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise CheckFixationsUnavailable(detail[-500:] or f"octave exited {completed.returncode}")
    rows: list[dict[str, float]] = []
    for line in completed.stdout.splitlines():
        if line.startswith("ERROR"):
            raise CheckFixationsUnavailable(line[6:].strip() or "checkFixations.m failed")
        if line.startswith("EMPTY"):
            return []
        if not line.startswith("FIX "):
            continue
        x_text, y_text, duration_text = line.split()[1:]
        rows.append({"x": float(x_text), "y": float(y_text), "duration_ms": float(duration_text)})
    if not rows:
        raise CheckFixationsUnavailable("octave returned no fixation rows and no error")
    return rows


def official_check_fixations_batch(
    trials: list[tuple[str, np.ndarray]],
    mfile: str | Path,
    octave: str = "octave",
) -> dict[str, list[dict[str, float]] | str]:
    """Convert many trials in one Octave process.

    The result maps each trial id to a fixation list, or to an error string
    that begins with ``error:``. Ids must be unique and must not contain commas.
    """
    if any("," in trial_id or "\n" in trial_id for trial_id, _ in trials):
        raise ValueError("trial ids cannot contain commas or newlines")
    checked = require_official_mfile(mfile)
    if shutil.which(octave) is None:
        raise CheckFixationsUnavailable(f"{octave} is not on PATH")
    with tempfile.TemporaryDirectory(prefix="checkfix-batch-") as directory:
        directory_path = Path(directory)
        list_path = directory_path / "trials.txt"
        lines = []
        for trial_id, eye in trials:
            samples = np.asarray(eye, dtype=np.float64)
            if samples.ndim != 2 or samples.shape[1] != 2 or samples.shape[0] < 2:
                raise ValueError(f"{trial_id} is not an N by 2 sample array")
            csv_path = directory_path / f"{trial_id}.csv"
            if csv_path.exists():
                raise ValueError(f"duplicate trial id {trial_id}")
            np.savetxt(csv_path, samples, delimiter=",")
            lines.append(f"{trial_id},{csv_path.as_posix()}")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out_path = directory_path / "fixations.csv"
        script = "\n".join(
            [
                "pkg load statistics;",
                f"addpath('{COMPAT.as_posix()}', '-begin');",
                f"addpath('{checked.parent.as_posix()}', '-begin');",
                "warning('off', 'all');",
                f"fid = fopen('{list_path.as_posix()}', 'r');",
                f"out = fopen('{out_path.as_posix()}', 'w');",
                "while true",
                "  line = fgetl(fid);",
                "  if ~ischar(line), break; end",
                "  parts = strsplit(line, ',');",
                "  trial_id = parts{1};",
                "  csv_path = parts{2};",
                "  try",
                "    d = dlmread(csv_path);",
                "    [~, Fix, ~] = checkFixations(d);",
                "    if ~isfield(Fix, 'duration')",
                "      fprintf(out, '%s,empty\\n', trial_id);",
                "    else",
                "      for k = 1:numel(Fix.duration)",
                "        fprintf(out, '%s,ok,%d,%.10f,%.10f,%.10f\\n', trial_id, k, Fix.medianXY(k, 1), Fix.medianXY(k, 2), Fix.duration(k));",
                "      end",
                "    end",
                "  catch err",
                "    fprintf(out, '%s,error,%s\\n', trial_id, strrep(err.message, sprintf('\\n'), ' '));",
                "  end",
                "end",
                "fclose(fid); fclose(out);",
            ]
        )
        completed = subprocess.run(
            [octave, "--no-gui", "--quiet", "--eval", script],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise CheckFixationsUnavailable(detail[-500:] or f"octave exited {completed.returncode}")
        grouped: dict[str, list[dict[str, float]] | str] = {trial_id: [] for trial_id, _ in trials}
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            parts = line.split(",")
            trial_id = parts[0]
            if trial_id not in grouped:
                raise CheckFixationsUnavailable(f"unexpected trial id {trial_id}")
            if parts[1] == "empty":
                grouped[trial_id] = []
            elif parts[1] == "error":
                grouped[trial_id] = "error: " + ",".join(parts[2:])
            elif parts[1] == "ok":
                current = grouped[trial_id]
                if isinstance(current, str):
                    raise CheckFixationsUnavailable(f"{trial_id} mixed an error with a fixation row")
                current.append(
                    {
                        "index": float(parts[2]),
                        "x": float(parts[3]),
                        "y": float(parts[4]),
                        "duration_ms": float(parts[5]),
                    }
                )
            else:
                raise CheckFixationsUnavailable(f"unrecognized octave row {line[:120]}")
    return grouped
