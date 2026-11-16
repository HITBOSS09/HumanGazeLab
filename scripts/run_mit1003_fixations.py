"""Convert MIT1003 raw samples with the authors' checkFixations.m.

Train and tune partitions only. The reserved final partition is not opened.
Outputs go to outputs/baseline_v1 and do not replace the frozen experiments.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from humangazelab.check_fixations import (
    OFFICIAL_SHA256,
    assert_open_partition,
    official_check_fixations_batch,
    require_official_mfile,
)
from humangazelab.mit1003 import README_USERS, assign_partition
from humangazelab.protocolshift import assert_new_output

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "external" / "mit1003_DATA.zip"
MANIFEST = ROOT / "outputs" / "protocolshift_prep_v1" / "mit1003_split_manifest.csv"
MFILE = Path("/tmp/checkFixations.m")
OUTPUT = ROOT / "outputs" / "baseline_v1"
CHUNK = 400

# Independent Octave session, before this wrapper existed. User ajs, three train stems.
EXPECTED = {
    "i05june05_static_street_boston_p1010764": (516.17721519, 373.44000000, 266.66666667),
    "i05june05_static_street_boston_p1010785": (499.46835443, 362.88000000, 387.50000000),
    "i05june05_static_street_boston_p1010800": (521.74683544, 411.45600000, 187.50000000),
}


def _open_stems() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    labels = manifest["partition"].map(str)
    if set(labels) != {"train", "tune", "final"}:
        raise RuntimeError("the frozen manifest does not contain train, tune, and final")
    opened = manifest.loc[labels.isin(["train", "tune"])].copy()
    for stem in opened["image_stem"].astype(str):
        assert_open_partition(stem)
    final_stems = set(manifest.loc[labels.eq("final"), "image_stem"].astype(str))
    if final_stems & set(opened["image_stem"].astype(str)):
        raise RuntimeError("a final-partition stem entered the conversion list")
    return opened


def _record(payload: dict, stem: str):
    if stem in payload:
        return payload[stem]
    keys = [key for key in payload if not str(key).startswith("__")]
    if len(keys) == 1:
        return payload[keys[0]]
    raise KeyError(stem)


def _read_eye(archive: zipfile.ZipFile, user: str, stem: str) -> np.ndarray:
    payload = loadmat(
        io.BytesIO(archive.read(f"DATA/{user}/{stem}.mat")),
        squeeze_me=True,
        struct_as_record=False,
    )
    eye = np.asarray(_record(payload, stem).DATA.eyeData, dtype=np.float64)
    if eye.ndim != 2 or eye.shape[1] < 2:
        raise ValueError(f"{user}/{stem} eyeData has shape {eye.shape}")
    # Column 3, when present, holds the 0-3 labels written by checkFixations.m.
    # The detector reads positions from the first two columns.
    return eye[:, :2]


def _chunked(rows: list[dict]) -> None:
    archive = zipfile.ZipFile(ARCHIVE)
    trials: list[tuple[str, np.ndarray]] = []
    meta: dict[str, dict] = {}
    summaries: list[dict] = []
    tune_rows: list[dict] = []
    nan_trials: list[tuple[str, np.ndarray]] = []
    nan_meta: dict[str, str] = {}
    train_sample = set(
        pd.read_csv(MANIFEST)
        .query("partition == 'train'")
        .sort_values("image_stem")["image_stem"]
        .astype(str)
        .head(40)
    )

    def flush(batch: list[tuple[str, np.ndarray]], mapping: dict[str, dict], kind: str) -> None:
        if not batch:
            return
        converted = official_check_fixations_batch(batch, MFILE)
        for trial_id, _eye in batch:
            result = converted[trial_id]
            info = mapping[trial_id]
            if isinstance(result, str):
                summaries.append({**info, "n_fixations": 0, "mean_duration_ms": np.nan, "status": result, "conversion": kind})
                continue
            durations = [row["duration_ms"] for row in result]
            summaries.append(
                {
                    **info,
                    "n_fixations": len(result),
                    "mean_duration_ms": float(np.mean(durations)) if durations else np.nan,
                    "status": "ok",
                    "conversion": kind,
                }
            )
            if kind == "official" and info["partition"] == "tune":
                for row in result:
                    tune_rows.append(
                        {
                            "user": info["user"],
                            "image_stem": info["image_stem"],
                            "fixation_index": int(row["index"]),
                            "x": row["x"],
                            "y": row["y"],
                            "duration_ms": row["duration_ms"],
                        }
                    )

    try:
        for user in README_USERS:
            for record in rows:
                stem = str(record["image_stem"])
                partition = assert_open_partition(stem)
                if partition != str(record["partition"]):
                    raise RuntimeError(f"manifest partition disagrees with the hash rule for {stem}")
                trial_id = f"{user}__{stem}"
                try:
                    eye = _read_eye(archive, user, stem)
                except Exception as exc:
                    summaries.append(
                        {
                            "user": user,
                            "image_stem": stem,
                            "partition": partition,
                            "n_samples": 0,
                            "n_negative_samples": 0,
                            "n_nan_samples": 0,
                            "samples": "as_released",
                            "n_fixations": 0,
                            "mean_duration_ms": np.nan,
                            "status": f"error: {type(exc).__name__}: {exc}",
                            "conversion": "official",
                        }
                    )
                    continue
                meta[trial_id] = {
                    "user": user,
                    "image_stem": stem,
                    "partition": partition,
                    "n_samples": int(eye.shape[0]),
                    "n_negative_samples": int(np.sum(eye < 0)),
                    "n_nan_samples": int(np.sum(~np.isfinite(eye))),
                    "samples": "as_released",
                }
                trials.append((trial_id, eye))
                if partition == "tune" or stem in train_sample:
                    mapped = eye.copy()
                    mapped[mapped < 0] = np.nan
                    nan_id = f"nan__{trial_id}"
                    nan_meta[nan_id] = {**meta[trial_id], "samples": "negatives_set_to_nan"}
                    nan_trials.append((nan_id, mapped))
                if len(trials) >= CHUNK:
                    flush(trials, meta, "official")
                    print(f"official {len(summaries)}", flush=True)
                    trials, meta = [], {}
        flush(trials, meta, "official")
        print(f"official done {len(summaries)}", flush=True)
        for start in range(0, len(nan_trials), CHUNK):
            batch = nan_trials[start : start + CHUNK]
            mapping = {trial_id: nan_meta[trial_id] for trial_id, _ in batch}
            flush(batch, mapping, "negatives_as_nan")
            print(f"nan-mapped {start + len(batch)}", flush=True)
    finally:
        archive.close()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "mit1003_fixation_counts.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(OUTPUT / "mit1003_tune_fixations.csv", index=False)
    _validate(summary)


def _validate(summary: pd.DataFrame) -> None:
    official = summary.loc[summary["conversion"].eq("official")]
    if (official["partition"] == "final").any():
        raise RuntimeError("final partition appeared in the conversion output")
    archive = zipfile.ZipFile(ARCHIVE)
    checks = []
    try:
        for stem, expected in EXPECTED.items():
            assert_open_partition(stem)
            eye = _read_eye(archive, "ajs", stem)
            from humangazelab.check_fixations import official_check_fixations

            fresh = official_check_fixations(eye, MFILE)
            got = (fresh[0]["x"], fresh[0]["y"], fresh[0]["duration_ms"])
            checks.append(
                {
                    "user": "ajs",
                    "image_stem": stem,
                    "expected_x": expected[0],
                    "expected_y": expected[1],
                    "expected_duration_ms": expected[2],
                    "got_x": got[0],
                    "got_y": got[1],
                    "got_duration_ms": got[2],
                    "abs_dx": abs(got[0] - expected[0]),
                    "abs_dy": abs(got[1] - expected[1]),
                    "abs_duration_ms": abs(got[2] - expected[2]),
                    "n_fixations": len(fresh),
                }
            )
    finally:
        archive.close()
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(OUTPUT / "mit1003_checkfixations_agreement.csv", index=False)
    if check_frame[["abs_dx", "abs_dy", "abs_duration_ms"]].max().max() > 1e-6:
        raise RuntimeError("checkFixations.m disagreed with the independent Octave session")
    grouped = (
        official.groupby("partition")
        .agg(n_trials=("status", "size"), n_ok=("status", lambda s: int((s == "ok").sum())), mean_n_fixations=("n_fixations", "mean"), mean_duration_ms=("mean_duration_ms", "mean"))
        .reset_index()
    )
    nan_mapped = summary.loc[summary["conversion"].eq("negatives_as_nan")]
    report = {
        "mfile_sha256": OFFICIAL_SHA256,
        "mfile_url": "https://people.csail.mit.edu/tjudd/WherePeopleLook/Code/DatabaseCode/checkFixations.m",
        "final_partition_opened": False,
        "n_official_trials": int(len(official)),
        "n_official_errors": int((official["status"] != "ok").sum()),
        "by_partition": grouped.to_dict(orient="records"),
        "negatives_as_nan_trials": int(len(nan_mapped)),
        "negatives_as_nan_errors": int((nan_mapped["status"] != "ok").sum()) if len(nan_mapped) else 0,
        "max_agreement_abs_error": float(check_frame[["abs_dx", "abs_dy", "abs_duration_ms"]].max().max()),
        "note": "Negatives stay in the official call. The readme calls them blinks. The m-file only special-cases NaN, and its no-saccade branch calls nanmedian, which this Octave statistics package does not implement.",
    }
    (OUTPUT / "mit1003_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def _stored_fixations(archive: zipfile.ZipFile, user: str, stem: str):
    payload = loadmat(
        io.BytesIO(archive.read(f"DATA/{user}/{stem}.mat")),
        squeeze_me=True,
        struct_as_record=False,
    )
    data = _record(payload, stem).DATA
    if not hasattr(data, "fixationData"):
        return None
    stored = data.fixationData
    if not hasattr(stored, "medianXY") or not hasattr(stored, "duration"):
        return None
    xy = np.atleast_2d(np.asarray(stored.medianXY, dtype=np.float64))
    if xy.shape[0] == 2 and xy.shape[1] != 2:
        xy = xy.T
    duration = np.atleast_1d(np.asarray(stored.duration, dtype=np.float64)).ravel()
    return xy, duration


def retry_failed() -> None:
    """Rerun trials the first pass could not read. Do not touch successes or the final partition."""
    require_official_mfile(MFILE)
    counts = pd.read_csv(OUTPUT / "mit1003_fixation_counts.csv")
    failed = counts.loc[counts["conversion"].eq("official") & ~counts["status"].eq("ok")].copy()
    for stem in failed["image_stem"].astype(str):
        assert_open_partition(stem)
    print(f"retry {len(failed)}", flush=True)
    archive = zipfile.ZipFile(ARCHIVE)
    tune_extra: list[dict] = []
    comparisons: list[dict] = []
    updated: dict[tuple[str, str], dict] = {}
    trials: list[tuple[str, np.ndarray]] = []
    meta: dict[str, dict] = {}

    def flush() -> None:
        if not trials:
            return
        converted = official_check_fixations_batch(trials, MFILE)
        for trial_id, _eye in trials:
            info = meta[trial_id]
            result = converted[trial_id]
            if isinstance(result, str):
                updated[(info["user"], info["image_stem"])] = {
                    "n_fixations": 0,
                    "mean_duration_ms": np.nan,
                    "status": result,
                }
                continue
            durations = [row["duration_ms"] for row in result]
            updated[(info["user"], info["image_stem"])] = {
                "n_fixations": len(result),
                "mean_duration_ms": float(np.mean(durations)) if durations else np.nan,
                "status": "ok",
            }
            if info["partition"] == "tune":
                for row in result:
                    tune_extra.append(
                        {
                            "user": info["user"],
                            "image_stem": info["image_stem"],
                            "fixation_index": int(row["index"]),
                            "x": row["x"],
                            "y": row["y"],
                            "duration_ms": row["duration_ms"],
                        }
                    )
            stored = info.get("stored")
            if stored is not None and result:
                xy, duration = stored
                n = min(len(result), len(duration), len(xy))
                dx = max(abs(result[i]["x"] - float(xy[i, 0])) for i in range(n)) if n else np.nan
                dy = max(abs(result[i]["y"] - float(xy[i, 1])) for i in range(n)) if n else np.nan
                dd = max(abs(result[i]["duration_ms"] - float(duration[i])) for i in range(n)) if n else np.nan
                comparisons.append(
                    {
                        "user": info["user"],
                        "image_stem": info["image_stem"],
                        "n_fresh": len(result),
                        "n_stored": int(len(duration)),
                        "max_abs_dx": dx,
                        "max_abs_dy": dy,
                        "max_abs_duration_ms": dd,
                    }
                )

    try:
        for record in failed.to_dict(orient="records"):
            user = str(record["user"])
            stem = str(record["image_stem"])
            partition = assert_open_partition(stem)
            trial_id = f"{user}__{stem}"
            eye = _read_eye(archive, user, stem)
            stored = _stored_fixations(archive, user, stem)
            meta[trial_id] = {"user": user, "image_stem": stem, "partition": partition, "stored": stored}
            trials.append((trial_id, eye))
            if len(trials) >= CHUNK:
                flush()
                print(f"retried {len(updated)}", flush=True)
                trials, meta = [], {}
        flush()
    finally:
        archive.close()
    for (user, stem), values in updated.items():
        mask = counts["conversion"].eq("official") & counts["user"].eq(user) & counts["image_stem"].eq(stem)
        for column, value in values.items():
            counts.loc[mask, column] = value
    counts.to_csv(OUTPUT / "mit1003_fixation_counts.csv", index=False)
    if tune_extra:
        previous = pd.read_csv(OUTPUT / "mit1003_tune_fixations.csv")
        pd.concat([previous, pd.DataFrame(tune_extra)], ignore_index=True).to_csv(
            OUTPUT / "mit1003_tune_fixations.csv", index=False
        )
    compared = pd.DataFrame(comparisons)
    if not compared.empty:
        compared.to_csv(OUTPUT / "mit1003_stored_fixation_agreement.csv", index=False)
    official = counts.loc[counts["conversion"].eq("official")]
    report_path = OUTPUT / "mit1003_conversion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["retry_trials"] = int(len(failed))
    report["retry_still_failing"] = int((~official["status"].eq("ok")).sum())
    report["official_ok"] = int(official["status"].eq("ok").sum())
    ok = official.loc[official["status"].eq("ok")]
    report["ok_mean_n_fixations"] = float(ok["n_fixations"].mean())
    report["ok_mean_duration_ms"] = float(ok["mean_duration_ms"].mean())
    if not compared.empty:
        report["stored_fixation_rows"] = int(len(compared))
        report["stored_count_mismatches"] = int((compared["n_fresh"] != compared["n_stored"]).sum())
        report["stored_max_abs_dx"] = float(compared["max_abs_dx"].max())
        report["stored_max_abs_dy"] = float(compared["max_abs_dy"].max())
        report["stored_max_abs_duration_ms"] = float(compared["max_abs_duration_ms"].max())
    report["position_columns"] = "first two columns; a third column of 0-3 labels is not passed to the detector"
    report["nan_functions"] = "nanmedian and nanvar are local two-line definitions because Octave statistics 1.6.3 does not implement them. checkFixations.m was not edited."
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key.startswith("retry") or key.startswith("stored") or key.startswith("ok_") or key.startswith("official")}, indent=2))


def main() -> None:
    assert_new_output(ROOT, OUTPUT)
    require_official_mfile(MFILE)
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    if "--retry" in sys.argv:
        retry_failed()
        return
    opened = _open_stems()
    print(f"open stems {len(opened)} final excluded", flush=True)
    _chunked(opened.to_dict(orient="records"))


if __name__ == "__main__":
    main()
