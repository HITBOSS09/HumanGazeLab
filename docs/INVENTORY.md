# Technical inventory

This file is a factual list of what this repository contains and what was run. It is written so the study description and any application answers can be drafted from the repository rather than from memory. It does not state a scientific conclusion.

## Specification

The requested project was a reproducible analysis of gaze trajectories on photographs, aimed at the topic of Max Planck CaCTüS Project 10, using the official COCO-FreeView release. The requested analyses were fixation and scanpath visualization, fixation density and center-bias summaries, fixation-position and saccade-distance statistics, early versus late gaze, a within-scanpath shuffled-order baseline, and group-aware uncertainty where the metadata allows it. The specification also required schema inspection before assuming a format, real public records, synthetic tests, no invented results, and no scientific conclusion written on the applicant's behalf.

A search for a public page that spells out "CaCTüS Project 10" did not return a project brief beyond the program description of the CaCTüS internship at the Max Planck Institute for Biological Cybernetics and neighboring Tübingen institutes. The analyses follow the written list in the specification, not a separate brief.

## Who wrote what

Hitanshu Vaidya specified the dataset, the analysis list, the deadline priority (a working reproducible study ahead of extra features), and the limits on interpretation and invented metadata.

The Python package, scripts, tests, figures, tables, and these notes are in this repository. The initial empty commit on the repository was made by Hitanshu Vaidya.

This inventory is part of that agent-written snapshot. Decisions in [ASSUMPTIONS.md](ASSUMPTIONS.md) were not resolved by editing the results after the fact. They are recorded so they can be accepted, rejected, or replaced.

## Implemented

Package `src/humangazelab/`:

| Module | Role |
| --- | --- |
| `adapter.py` | Loads the JSON list and rejects unequal `X`/`Y`/`T` lengths |
| `quality.py` | Counts, schema comparison against readme v1.0, subject and split tables |
| `features.py` | Per-scanpath radius, successive distance, shuffle, ordinal and temporal contrasts |
| `summarize.py` | Equal-group means and percentile bootstrap |
| `spatial.py` | Density maps, disk fraction, grid and Monte Carlo radius reference |
| `zipremote.py` | ZIP central directory and CRC-checked extraction over HTTP ranges or a local file |
| `prepare.py` | Archive index, seeded 18-image subset, zero-border measurement |
| `download.py` | Fixation and readme download, hash, test-URL probe, disk space |
| `pipeline.py` | Writes tables, figures, and `outputs/verification_report.txt` |
| `plotting.py` | Figure writers |
| `config.py` | Loads `configs/analysis.yaml` |

Scripts: `scripts/download_data.py`, `scripts/prepare_images.py`, `scripts/run_analysis.py`, `scripts/run_all.py`.

## Executed

1. `scripts/download_data.py`. Fixation hash matched the file already retrieved from the official URL. Readme hash matched the committed copy. Test URL returned HTTP 404. Record: `outputs/download_verification.json`.
2. `scripts/prepare_images.py`. Indexed both stimulus zips by range request, extracted 18 images, all 1680×1050. Record: `outputs/image_archive_summary.json` and `outputs/tables/image_subset_manifest.csv`.
3. `scripts/run_analysis.py`. The files in `outputs/` are from the run stamped `2026-11-14T05:07:54.837138+00:00`. Runtime was about 35 seconds after the data were local.
4. `.venv/bin/pytest`. 18 tests passed, including the hash and count checks against the public JSON.

The six-panel overlay was inspected. Marks fall on scene content with y increasing downward, and index 0 is near the canvas center. That inspection is described in [DATASET.md](DATASET.md).

## Primary subject-level rows

Authoritative values, intervals, train/validation splits, image-level rows, and sensitivity rows are in `outputs/tables/metrics_summary.csv`. The verification report's subject-equal, all-scanpath rows from this run are:

| Metric | Estimate | 2.5% | 97.5% | Scanpaths | Comparison |
| --- | --- | --- | --- | --- | --- |
| Mean radius, index 0 excluded (px) | 314.922 | 298.566 | 330.126 | 42,997 | 530.685 (1 px grid) |
| Share within 200 px | 0.287822 | 0.255425 | 0.324492 | 42,997 | 0.071238 (disk area) |
| Mean successive distance (px) | 270.212 | 240.670 | 296.873 | 42,932 | |
| Shuffled minus released successive distance (px) | 122.554 | 100.747 | 137.996 | 42,932 | |
| Late minus early radius, k=3 (px) | 36.921 | 21.317 | 52.098 | 42,731 | |
| Late minus early successive distance, k=3 (px) | 17.073 | −0.778 | 34.859 | 42,610 | |
| Late minus early radius, onset 1,500 (px) | 13.508 | −0.399 | 27.056 | 42,580 | |
| Late minus early successive distance, onset 1,500 (px) | 12.125 | −3.991 | 27.391 | 42,068 | |

Per-subject values are in `outputs/tables/subject_metrics.csv`. Fixation-weighted quantiles are in `outputs/tables/descriptive_marginals.csv`. The fixation-weighted mean post-initial radius there is 317.678 px, which is a different weighting from the 314.922 px subject-equal estimate.

## Not implemented

- No deep-learning training, no saliency model, no scanpath predictor.
- No evaluation on a held-out test set. The test file was not available.
- No conversion to degrees of visual angle.
- No use of `task`, `correct`, or `fixOnTarget` as outcomes.
- No mixed-effects model. Uncertainty is the cluster bootstrap described in the methodology.
- No dashboard.

## Files a reader can trust for a number

Use `outputs/tables/metrics_summary.csv` and `outputs/verification.json` rather than a rounded number from a notebook or from this prose. If this prose and a CSV disagree, the CSV from the stamped run is the one that was computed.
