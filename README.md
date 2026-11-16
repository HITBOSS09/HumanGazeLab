# HumanGazeLab

Reproducible descriptive analysis of free-viewing gaze trajectories on photographs, using the public train and validation release of [COCO-FreeView](https://sites.google.com/view/cocosearch/coco-freeview).

The pipeline downloads that release, checks it against the authors' readme, and writes fixation statistics, a within-scanpath shuffle baseline, subject-level and image-level uncertainty, tables, and figures. It does not train a model, and it does not write a scientific conclusion. The numbers to use are the files in `outputs/`.

## Run

From this directory, with Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_all.py
.venv/bin/pytest
```

If `python3 -m venv` fails because `ensurepip` is missing, create the environment with `python3 -m venv --without-pip .venv` and install pip with `curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python`. That is the path used in this environment.

`scripts/run_all.py` downloads the fixation file and the format readme, extracts an 18-image stimulus subset, and runs the analysis. The three steps are also separate:

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/prepare_images.py
.venv/bin/python scripts/run_analysis.py
```

If a host blocks the download, the fallback is in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## What you get

| Path | Contents |
| --- | --- |
| `outputs/verification_report.txt` | Counts, hash, and the primary subject-level estimates |
| `outputs/tables/metrics_summary.csv` | Estimates, subject or image bootstrap intervals, and sensitivity rows |
| `outputs/tables/subject_metrics.csv` | One row per subject id |
| `outputs/tables/schema_comparison.csv` | Readme fields versus fields present in the JSON |
| `outputs/figures/` | Density maps, radius and saccade histograms, scanpath overlays |
| `docs/METHODOLOGY.md` | Definitions used to compute those files |
| `docs/ASSUMPTIONS.md` | Choices and checks that still need a human decision |
| `docs/INVENTORY.md` | What was implemented, executed, and left undone |

Raw fixation records and stimulus images stay in `data/raw/` and are gitignored. `outputs/tables/scanpath_level.csv` is regenerated locally and is also gitignored (about 35 MB).

## Data

The fixation file used here is the authors' train+validation JSON, last modified 23 June 2023, SHA-256 `771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730`. It contains 43,048 scanpaths, 665,005 fixations, 4,317 images, and 10 subject ids. A probe of the obvious test-file URL returned HTTP 404. The project page says the testing set was taken down. Details, including where the file disagrees with readme v1.0, are in [docs/DATASET.md](docs/DATASET.md).

Stimuli are the COCO-Search18 images named on the [COCO-Search18 page](https://sites.google.com/view/cocosearch/home). The analysis of positions uses every public fixation. Overlays use 18 images chosen by a seeded sample that does not look at gaze values.

## Citation

COCO-FreeView asks users to cite Chen et al. 2022. The project page calls that paper a CVPR workshop paper. The bundled readme's BibTeX names the venue as CVPR. Both wordings are recorded in [docs/DATASET.md](docs/DATASET.md). The stimulus images come from COCO-Search18 (Chen et al., Scientific Reports, 2021).

## Authorship

[docs/INVENTORY.md](docs/INVENTORY.md) separates the implementation from the decisions the specification left open. This repository does not add a software license; dataset terms remain with the original authors.

## Temporal predictability

A second, versioned experiment asks how much information about a held-out fixation is available from other observers and from the preceding fixation. It does not replace the descriptive analysis above, and it does not adopt a hypothesis. The command is:

```bash
.venv/bin/python scripts/run_temporal_v1.py
```

Tables and figures go to `outputs/temporal_v1/`. An OSIE within-dataset run, which is not pooled with COCO-FreeView, is:

```bash
.venv/bin/python scripts/run_osie_v1.py
```

That command expects `data/raw/external/osie_fixations.mat`. The file is not committed. The protocol, the open decisions, and the publication audit are in `docs/EXPERIMENT_PROTOCOL.md`, `docs/HYPOTHESIS_DECISION_WORKSHEET.md`, and `docs/PUBLICATION_READINESS_AUDIT.md`.

The validation checkpoint leaves those frozen outputs in place. It writes `outputs/audit_checkpoint/` and a train-fold-only bandwidth curve at `outputs/temporal_v2/`. The measured checks, including what was not verified, are in `docs/TECHNICAL_EVIDENCE_REPORT.md`.

A later sensitivity analysis asks whether the index-conditioned COCO-FreeView and OSIE scores survive observer-count, index, bandwidth, and canvas-fraction controls. It writes `outputs/protocolshift_b3_v1/` and does not replace the frozen experiments. The novelty audit and the publication gate are `docs/NOVELTY_AUDIT.md` and `docs/PUBLICATION_GATE.md`. The gate's decision is NO GO.

A further pass compared the official DeepGaze, DeepGaze MSDB, DeepGaze3.5-VL, and HAT implementations and stopped before training a model. The assessment is `docs/NOVELTY_GAP_ASSESSMENT.md`. Its model-development decision is also NO GO.

The baseline-reproduction pass starts from commit `fd68b38e82626d50e6a6a9d4779d02f5056e0e1d` and writes `outputs/baseline_v1/`. It runs official DeepGaze III on the nine local OSIE images, converts MIT1003 train and tune samples with the authors' `checkFixations.m`, and measures low-level image features on the tuning images only. The reserved final partition is not read. Commands, from this directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mit1003_fixations.py
PYTHONPATH=src .venv/bin/python scripts/run_deepgaze3_baseline.py
PYTHONPATH=src .venv/bin/python scripts/run_visual_features.py
```

Those commands need the gitignored archives, the authors' m-file, Octave, and the DeepGaze checkout and weights outside this repository. The measured results and the decision are in `docs/BASELINE_REPRODUCTION.md` and `docs/RESEARCH_DECISION_MEMO.md`. The publication gate remains NO GO.

The same frozen DeepGaze III checkpoint was then scored on all 700 released OSIE images, with a wrong-image control on the same eligible targets. It writes `outputs/osie_deepgaze3_v2/` and does not replace `outputs/baseline_v1/`. The result is exploratory.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_osie_deepgaze3_v2.py
```

The command needs the gitignored OSIE JPEGs and `osie_fixations.mat`, plus the DeepGaze checkout and weights outside this repository. The protocol, the measured intervals, and the decision not to begin an object-contour study are in `docs/OSIE_DEEPGAZE3_V2.md` and `docs/OBJECT_REGION_STUDY_DECISION.md`.

SemanticGaze-Cal asks whether that same frozen density, summed inside the official OSIE object masks, matches how often fixations land in those masks. It writes `outputs/semantic_calibration_v1/` and does not replace the 700-image table. The journal gate stays NO GO.

```bash
PYTHONPATH=src TORCH_HOME=/tmp/torch SEMANTIC_STAGE=A .venv/bin/python scripts/run_semantic_calibration_v1.py
PYTHONPATH=src TORCH_HOME=/tmp/torch .venv/bin/python scripts/run_semantic_calibration_v1.py
```

The first command is the geometry check on a predeclared image subset. The second resumes and scores the remaining images. Both need `data/raw/external/osie_attrs.mat` as well as the stimuli, the fixations, and the DeepGaze checkout. The protocol is `docs/SEMANTIC_CALIBRATION_PROTOCOL.md`. A result file from the first command is not the full measurement. The measured tables and the decision are `docs/SEMANTIC_RESULTS.md` and `docs/SEMANTIC_NEXT_DECISION.md`. The journal gate remains NO GO.
